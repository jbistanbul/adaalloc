"""Video probing, frame-index sampling, frame decoding and resizing to a pixel budget."""

from __future__ import annotations

import ctypes
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .data import VideoMetadata

_VIDEO_BACKENDS = ("decord", "torchcodec", "pyav", "torchvision", "opencv")
_DECODER_IMPORT_ERRORS: dict[str, str] = {}


def _preload_conda_libwebp_for_decord() -> None:
    """Load conda's libwebp stack globally before decord.

    The pip torchvision wheel bundles its own libwebp on some platforms. When torchvision is imported
    before decord, FFmpeg's libwebpmux may bind against that bundled libwebp and fail to resolve
    WebPMalloc. Loading conda's matching libwebp, mux, and demux libraries with RTLD_GLOBAL keeps
    decord/FFmpeg bound to one consistent conda libwebp stack.
    """
    prefix_candidates = [Path(sys.prefix)]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix_candidates.append(Path(conda_prefix))
    load_mode = ctypes.RTLD_GLOBAL | getattr(os, "RTLD_NOW", 0)
    seen_dirs: set[Path] = set()
    for prefix in prefix_candidates:
        lib_dir = (prefix / "lib").resolve()
        if lib_dir in seen_dirs:
            continue
        seen_dirs.add(lib_dir)
        for sonames in (
            ("libsharpyuv.so.0", "libsharpyuv.so"),
            ("libwebp.so.7", "libwebp.so"),
            ("libwebpdemux.so.2", "libwebpdemux.so"),
            ("libwebpmux.so.3", "libwebpmux.so"),
        ):
            for soname in sonames:
                lib_path = lib_dir / soname
                if not lib_path.exists():
                    continue
                try:
                    ctypes.CDLL(str(lib_path), mode=load_mode)
                except Exception:
                    pass
                break


def _backend_import_error(backend: str) -> str | None:
    try:
        if backend == "torchcodec":
            from torchcodec.decoders import VideoDecoder  # noqa: F401
        elif backend == "decord":
            _preload_conda_libwebp_for_decord()
            import decord  # noqa: F401
        elif backend == "pyav":
            import av  # noqa: F401
        elif backend == "torchvision":
            import torchvision.io  # noqa: F401
        elif backend == "opencv":
            import cv2  # noqa: F401
        else:  # pragma: no cover
            raise ValueError(f"unknown video backend: {backend}")
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _video_backend_available(backend: str) -> bool:
    error = _backend_import_error(backend)
    if error is not None:
        _DECODER_IMPORT_ERRORS[backend] = error
        return False
    _DECODER_IMPORT_ERRORS.pop(backend, None)
    return True


def _preferred_video_backends() -> tuple[str, ...]:
    healthy = tuple(backend for backend in _VIDEO_BACKENDS if _video_backend_available(backend))
    if healthy:
        return healthy
    return ("opencv",)


def preferred_qwen_video_reader_backend() -> str:
    """Pick a qwen-vl-utils reader that is actually importable in this process."""
    for backend in ("decord", "torchcodec", "torchvision"):
        if _video_backend_available(backend):
            return backend
    return "torchvision"


def _open_video_reader(video_path: str | Path):
    _preload_conda_libwebp_for_decord()
    from decord import VideoReader, cpu

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    return VideoReader(str(path), ctx=cpu(0))


def _open_opencv_capture(video_path: str | Path):
    import cv2

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"Unable to open video with OpenCV: {path}")
    return capture


def _probe_video_metadata_torchcodec(video_path: str | Path, fallback_fps: float | None = None) -> VideoMetadata:
    from torchcodec.decoders import VideoDecoder

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    decoder = VideoDecoder(str(path))
    metadata = decoder.metadata
    fps = float(metadata.average_fps) if metadata.average_fps else fallback_fps
    total_frames = int(metadata.num_frames) if metadata.num_frames else None
    duration_sec = float(metadata.duration_seconds) if metadata.duration_seconds else None
    if duration_sec is None and total_frames and fps:
        duration_sec = total_frames / fps
    return VideoMetadata(total_frames=total_frames, fps=fps, duration_sec=duration_sec, source="video-probe:torchcodec")


def _probe_video_metadata_decord(video_path: str | Path, fallback_fps: float | None = None) -> VideoMetadata:
    reader = _open_video_reader(video_path)
    fps = float(reader.get_avg_fps()) if reader.get_avg_fps() else fallback_fps
    total_frames = int(len(reader)) or None
    duration_sec = None
    if total_frames and fps:
        duration_sec = total_frames / fps
    return VideoMetadata(
        total_frames=total_frames,
        fps=float(fps) if fps else fallback_fps,
        duration_sec=duration_sec,
        source="video-probe:decord",
    )


def _probe_video_metadata_pyav(video_path: str | Path, fallback_fps: float | None = None) -> VideoMetadata:
    import av

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else fallback_fps
        total_frames = int(stream.frames) if stream.frames else None
        duration_sec = None
        if stream.duration is not None and stream.time_base is not None:
            duration_sec = float(stream.duration * stream.time_base)
        if total_frames is None:
            total_frames = sum(1 for _ in container.decode(stream)) or None
        if duration_sec is None and total_frames and fps:
            duration_sec = total_frames / fps
    return VideoMetadata(total_frames=total_frames, fps=fps, duration_sec=duration_sec, source="video-probe:pyav")


def _probe_video_metadata_torchvision(video_path: str | Path, fallback_fps: float | None = None) -> VideoMetadata:
    import torch
    from torchvision.io import read_video

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    video, _, info = read_video(str(path), pts_unit="sec", output_format="TCHW")
    total_frames = int(video.shape[0]) if isinstance(video, torch.Tensor) else None
    fps = float(info.get("video_fps") or 0.0) or fallback_fps
    duration_sec = total_frames / fps if total_frames and fps else None
    return VideoMetadata(total_frames=total_frames, fps=fps, duration_sec=duration_sec, source="video-probe:torchvision")


def _probe_video_metadata_opencv(video_path: str | Path, fallback_fps: float | None = None) -> VideoMetadata:
    import cv2

    capture = _open_opencv_capture(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or fallback_fps
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    finally:
        capture.release()

    duration_sec = None
    if total_frames and fps:
        duration_sec = total_frames / fps
    return VideoMetadata(
        total_frames=total_frames,
        fps=float(fps) if fps else fallback_fps,
        duration_sec=duration_sec,
        source="video-probe:opencv",
    )


def _run_with_video_backend_fallbacks(operation: Callable[[str], object], failure_message: str):
    last_error: Exception | None = None
    attempted: list[str] = []
    for backend in _preferred_video_backends():
        attempted.append(backend)
        try:
            return operation(backend)
        except Exception as exc:
            last_error = exc
    errors = "; ".join(f"{name} import failed: {error}" for name, error in _DECODER_IMPORT_ERRORS.items())
    suffix = f" attempted={attempted}"
    if errors:
        suffix += f"; import_errors=({errors})"
    raise FileNotFoundError(f"{failure_message}{suffix}") from last_error


def probe_video_metadata(video_path: str, fallback_fps: float | None = None) -> VideoMetadata:
    """Frame count, fps and duration from the first healthy decoder backend."""
    path = Path(video_path)
    if not path.exists():
        return VideoMetadata(fps=fallback_fps, source="missing-path")

    def _probe(backend: str) -> VideoMetadata:
        if backend == "torchcodec":
            return _probe_video_metadata_torchcodec(path, fallback_fps=fallback_fps)
        if backend == "decord":
            return _probe_video_metadata_decord(path, fallback_fps=fallback_fps)
        if backend == "pyav":
            return _probe_video_metadata_pyav(path, fallback_fps=fallback_fps)
        if backend == "torchvision":
            return _probe_video_metadata_torchvision(path, fallback_fps=fallback_fps)
        return _probe_video_metadata_opencv(path, fallback_fps=fallback_fps)

    try:
        return _run_with_video_backend_fallbacks(_probe, f"Unable to probe video metadata from: {video_path}")
    except FileNotFoundError:
        return VideoMetadata(fps=fallback_fps, source="probe-failed")


def video_metadata_complete(metadata: VideoMetadata) -> bool:
    return metadata.total_frames is not None and metadata.fps is not None and metadata.duration_sec is not None


@lru_cache(maxsize=4096)
def probe_video_dimensions(video_path: str) -> tuple[int, int] | None:
    """Return the coded ``(height, width)`` without decoding frames, or None when unavailable."""

    path = Path(video_path)
    if not path.exists():
        return None
    try:
        import cv2

        capture = _open_opencv_capture(path)
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            capture.release()
    except Exception:
        return None
    if height <= 0 or width <= 0:
        return None
    return height, width


def uniform_frame_indices(total_frames: int, budget: int) -> list[int]:
    """Up to ``budget`` frame indices spread evenly from the first to the last frame."""
    if total_frames <= 0 or budget <= 0:
        return []
    if budget == 1:
        return [0 if total_frames == 1 else total_frames // 2]
    positions = np.linspace(0, max(0, total_frames - 1), num=min(budget, total_frames))
    return np.unique(np.round(positions).astype(int)).tolist()


def qwen_fps_frame_indices(
    total_frames: int,
    native_fps: float,
    sample_fps: float,
    max_frames: int,
    min_frames: int = 4,
) -> list[int]:
    """Frame indices sampled at ``sample_fps`` like qwen-vl-utils: an even count in [min_frames, max_frames]."""
    if total_frames <= 0 or native_fps <= 0 or sample_fps <= 0 or max_frames <= 0:
        return []

    if total_frames == 1:
        return [0]

    target_frames = total_frames / native_fps * sample_fps
    target_frames = min(max(target_frames, min_frames), max_frames, total_frames)
    target_frames = int(math.floor(target_frames / 2.0) * 2)

    if target_frames <= 0:
        target_frames = min(total_frames, max(1, max_frames))
    if target_frames == 1:
        return [0 if total_frames == 1 else total_frames // 2]

    positions = np.linspace(0, max(0, total_frames - 1), num=target_frames)
    return np.unique(np.round(positions).astype(int)).tolist()


def _decode_frames_at_indices_torchcodec(video_path: str | Path, indices: list[int]) -> list[np.ndarray]:
    from torchcodec.decoders import VideoDecoder

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    decoder = VideoDecoder(str(path))
    total_frames = int(decoder.metadata.num_frames)
    unique_sorted = sorted({min(max(0, int(index)), max(0, total_frames - 1)) for index in indices})
    if not unique_sorted:
        return []
    batch = decoder.get_frames_at(indices=unique_sorted).data
    if hasattr(batch, "detach"):
        batch = batch.detach().cpu()
    # TorchCodec returns TCHW; convert to RGB HWC numpy frames.
    if getattr(batch, "ndim", 0) == 4 and int(batch.shape[1]) == 3:
        batch = batch.permute(0, 2, 3, 1)
    return [np.array(frame, dtype=np.uint8, copy=True, order="C") for frame in batch.numpy()]


def _decode_frames_at_indices_decord(video_path: str | Path, indices: list[int]) -> list[np.ndarray]:
    reader = _open_video_reader(video_path)
    total_frames = len(reader)
    unique_sorted = sorted({min(max(0, int(index)), max(0, total_frames - 1)) for index in indices})
    if not unique_sorted:
        return []

    try:
        batch = reader.get_batch(unique_sorted).asnumpy()
        return [np.array(frame, dtype=np.uint8, copy=True, order="C") for frame in batch]
    except Exception:
        frames: list[np.ndarray] = []
        for index in unique_sorted:
            frames.append(np.array(reader[index].asnumpy(), dtype=np.uint8, copy=True, order="C"))
        return frames


def _decode_frames_at_indices_pyav(video_path: str | Path, indices: list[int]) -> list[np.ndarray]:
    import av

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    requested = sorted({max(0, int(index)) for index in indices})
    if not requested:
        return []
    requested_set = set(requested)
    max_requested = requested[-1]
    frames_by_index: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index in requested_set:
                frames_by_index[frame_index] = np.array(frame.to_ndarray(format="rgb24"), dtype=np.uint8, copy=True, order="C")
                if len(frames_by_index) == len(requested_set):
                    break
            if frame_index >= max_requested and len(frames_by_index) == len(requested_set):
                break
    if not frames_by_index:
        raise ValueError(f"PyAV failed to decode requested frames for {video_path}")
    return [frames_by_index[index] for index in requested if index in frames_by_index]


def _decode_frames_at_indices_torchvision(video_path: str | Path, indices: list[int]) -> list[np.ndarray]:
    from torchvision.io import read_video

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    video, _, _ = read_video(str(path), pts_unit="sec", output_format="TCHW")
    total_frames = int(video.shape[0])
    unique_sorted = sorted({min(max(0, int(index)), max(0, total_frames - 1)) for index in indices})
    if not unique_sorted:
        return []
    selected = video[unique_sorted].permute(0, 2, 3, 1).detach().cpu().numpy()
    return [np.array(frame, dtype=np.uint8, copy=True, order="C") for frame in selected]


def _decode_frames_at_indices_opencv(video_path: str | Path, indices: list[int]) -> list[np.ndarray]:
    import cv2

    capture = _open_opencv_capture(video_path)
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            raise ValueError(f"Unable to determine total frame count for {video_path}")

        unique_sorted = sorted({min(max(0, int(index)), max(0, total_frames - 1)) for index in indices})
        if not unique_sorted:
            return []

        frames: list[np.ndarray] = []
        for index in unique_sorted:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(index)):
                raise ValueError(f"OpenCV failed to seek to frame {index} for {video_path}")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"OpenCV failed to decode frame {index} for {video_path}")
            frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8, copy=True, order="C"))
        return frames
    finally:
        capture.release()


def decode_frames_at_indices(video_path: str, indices: list[int]) -> list[np.ndarray]:
    """Decode the given frames (deduplicated, sorted, clamped to the video) as RGB HWC uint8 arrays."""
    if not indices:
        return []

    def _decode(backend: str) -> list[np.ndarray]:
        if backend == "torchcodec":
            return _decode_frames_at_indices_torchcodec(video_path, indices)
        if backend == "decord":
            return _decode_frames_at_indices_decord(video_path, indices)
        if backend == "pyav":
            return _decode_frames_at_indices_pyav(video_path, indices)
        if backend == "torchvision":
            return _decode_frames_at_indices_torchvision(video_path, indices)
        return _decode_frames_at_indices_opencv(video_path, indices)

    return _run_with_video_backend_fallbacks(_decode, f"Unable to decode video frames from: {video_path}")


def resize_frames_to_total_pixels(
    frames: Sequence[np.ndarray],
    *,
    max_total_pixels: int,
    temporal_factor: int = 1,
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> list[np.ndarray]:
    """Resize every frame to one Qwen-compatible size so the clip fits ``max_total_pixels``."""
    if not frames:
        return []

    from PIL import Image

    from .qwen import resize_size_for_profile, resolve_qwen_profile

    profile = resolve_qwen_profile(model_name)
    max_total_pixels = max(profile.min_frame_pixels, int(max_total_pixels))
    first = np.asarray(frames[0], dtype=np.uint8)
    height, width = first.shape[:2]
    min_total_pixels = min(max_total_pixels, max(profile.min_frame_pixels, profile.min_frame_pixels * len(frames)))
    resized_height, resized_width = resize_size_for_profile(
        profile,
        frame_count=len(frames),
        height=height,
        width=width,
        temporal_factor=temporal_factor,
        min_total_pixels=min_total_pixels,
        max_total_pixels=max_total_pixels,
    )
    resized: list[np.ndarray] = []
    for frame in frames:
        array = np.array(frame, dtype=np.uint8, copy=True, order="C")
        if resized_height != array.shape[0] or resized_width != array.shape[1]:
            image = Image.fromarray(array)
            array = np.array(image.resize((resized_width, resized_height), Image.Resampling.BICUBIC), dtype=np.uint8, copy=True)
        resized.append(array)
    return resized

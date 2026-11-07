"""``budget.num_frames`` resolves every frame cap from the multiples in the config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from adaalloc.config import BudgetConfig, Config, apply_budget, load_config
from adaalloc.planning import AdaptiveBudgetPolicy, AdaptiveOverviewPlan, force_allocation_policy

ROOT = Path(__file__).resolve().parents[1]


def _base(**overrides) -> Config:
    return load_config(ROOT / "configs" / "adaalloc_128f.yaml", overrides=overrides)


@pytest.mark.parametrize(("name", "num_frames", "visual_tokens"), [("adaalloc_128f.yaml", 128, 32768), ("adaalloc_64f.yaml", 64, 16384)])
def test_num_frames_names_the_paper_budgets(name, num_frames, visual_tokens):
    config = load_config(ROOT / "configs" / name)
    assert config.budget.num_frames == num_frames
    assert config.budget.visual_tokens == visual_tokens
    assert config.budget.total_pixels == visual_tokens * 32 * 32
    assert config.adaptive_allocation is True


def test_64f_config_only_changes_the_budget():
    raw = yaml.safe_load((ROOT / "configs" / "adaalloc_64f.yaml").read_text())
    assert raw == {"base": "adaalloc_128f.yaml", "name": "adaalloc_64f", "budget": {"num_frames": 64}}


@pytest.mark.parametrize("num_frames", [32, 64, 96, 128, 256])
def test_frame_caps_follow_num_frames(num_frames):
    config = _base(**{"budget.num_frames": num_frames})
    assert config.budget.visual_tokens == num_frames * 256
    assert (config.caps.planner, config.caps.refiner, config.caps.local) == (num_frames, 2 * num_frames, num_frames)
    assert config.caps.overview == {"GLOBAL_ONLY": 2 * num_frames, "GLOBAL_HEAVY": 2 * num_frames, "BALANCED": 2 * num_frames, "LOCAL_HEAVY": num_frames}
    # per-frame resolution floors and everything else do not depend on the budget
    reference = _base()
    assert config.frames == reference.frames
    assert (config.tokens_per_frame, config.fps, config.retrieval, config.memory, config.refiner) == (
        reference.tokens_per_frame, reference.fps, reference.retrieval, reference.memory, reference.refiner
    )


@pytest.mark.parametrize("num_frames", [0, -64, 64.5, "64", True])
def test_invalid_num_frames_is_rejected(num_frames):
    with pytest.raises(ValueError):
        apply_budget(Config(budget=BudgetConfig(num_frames=num_frames)))


@pytest.mark.parametrize(("overrides", "cap"), [({"frames.local": 1.5}, 96), ({"frames.refiner": 0.5}, 32)])
def test_fractional_multiples_resolve_when_whole(overrides, cap):
    config = _base(**{"budget.num_frames": 64}, **overrides)
    assert getattr(config.caps, next(iter(overrides)).split(".")[1]) == cap


@pytest.mark.parametrize("overrides", [{"budget.num_frames": 65, "frames.local": 1.5}, {"frames.planner": 0}, {"frames.planner": "1"}])
def test_multiples_that_break_whole_frame_counts_are_rejected(overrides):
    with pytest.raises(ValueError):
        _base(**overrides)


def test_caps_cannot_be_set_directly():
    with pytest.raises(ValueError):
        _base(**{"caps.planner": 100})


def test_fixed_allocation_config():
    config = _base(adaptive_allocation=False)
    reference = _base()
    assert (config.adaptive_allocation, config.fixed_policy) == (False, "BALANCED")
    assert (config.caps, config.tokens_per_frame) == (reference.caps, reference.tokens_per_frame)
    with pytest.raises(ValueError):
        _base(adaptive_allocation=False, fixed_policy="HALF")
    with pytest.raises(ValueError):
        _base(adaptive_allocation="no")


@pytest.mark.parametrize(("policy", "ratios", "local_needed"), [("BALANCED", (0.5, 0.5), True), ("LOCAL_ONLY", (0.0, 1.0), True), ("GLOBAL_ONLY", (1.0, 0.0), False)])
def test_fixed_policy_replaces_only_the_split(policy, ratios, local_needed):
    plan = AdaptiveOverviewPlan(
        queries=["the chef returns the pan to the stove"],
        budget_policy=AdaptiveBudgetPolicy(policy_name="LOCAL_HEAVY", overview_ratio=0.3, local_ratio=0.7, overview_min_ratio=0.3, local_budget_ratio=0.7),
        local_retrieval_needed=False,
    )
    metadata = force_allocation_policy(plan, policy)
    budget = plan.budget_policy
    assert budget.policy_name == policy
    assert (budget.overview_ratio, budget.local_ratio) == ratios == (budget.overview_min_ratio, budget.local_budget_ratio)
    assert budget.local_evidence_needed is local_needed
    # the plan's queries and retrieval decision are kept
    assert plan.queries == ["the chef returns the pan to the stove"] and plan.local_retrieval_needed is False
    assert metadata["planner_policy_name"] == "LOCAL_HEAVY"

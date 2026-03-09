# Copyright Sierra
# Unit tests for planner module: plan(), build_planner_guidance_text(), run_loop integration.

import pytest
from unittest.mock import MagicMock, patch

from tau_bench.types import Task, Action, RESPOND_ACTION_NAME
from tau_bench.orchestration.task_state import create_initial_task_state
from tau_bench.orchestration.recovery import RecoveryState
from tau_bench.orchestration.planner import (
    PlanResult,
    StepSpec,
    StepType,
    plan,
    build_planner_guidance_text,
)


def test_plan_from_task_state_returns_plan_result():
    """Given a TaskState (and optional RecoveryState), plan() returns PlanResult with required fields."""
    task = Task(user_id="u1", actions=[], instruction="Book a flight for tomorrow", outputs=[])
    state = create_initial_task_state(domain="airline", task=task)
    result = plan(state, None, step_index=1, max_num_steps=30)
    assert isinstance(result, PlanResult)
    assert result.current_goal == "Book a flight for tomorrow"
    assert isinstance(result.subgoal, str)
    assert 1 <= len(result.next_steps) <= 3
    assert all(isinstance(s, StepSpec) for s in result.next_steps)
    assert result.preferred_next_action_type in ("tool_call", "ask_user", "respond", "wait_terminate_safe")
    assert len(result.replan_triggers) > 0
    assert len(result.anti_patterns_to_avoid) > 0


def test_plan_surfaces_constraints():
    """TaskState with unresolved_slots / required_prerequisites -> plan's active_constraints includes them."""
    task = Task(user_id="u1", actions=[], instruction="Cancel my reservation", outputs=[])
    state = create_initial_task_state(domain="airline", task=task)
    state.intent.unresolved_slots = ["reservation_id", "confirmation"]
    state.checklist.required_prerequisites = ["get_user_details"]
    result = plan(state, None, step_index=1, max_num_steps=30)
    assert any("reservation_id" in c for c in result.active_constraints)
    assert any("confirmation" in c for c in result.active_constraints)
    assert any("get_user_details" in c or "Prerequisite" in c for c in result.active_constraints)


def test_plan_prefers_tool_call_when_progress_required():
    """State with attempted_mutating_tools and empty successful_mutations -> preferred_next_action_type is tool_call."""
    task = Task(user_id="u1", actions=[], instruction="Book a flight", outputs=[])
    state = create_initial_task_state(domain="airline", task=task)
    state.attempted_mutating_tools.add("book_reservation")
    assert len(state.successful_mutations) == 0
    result = plan(state, None, step_index=2, max_num_steps=30)
    assert result.preferred_next_action_type == "tool_call"


def test_plan_no_drift_after_tool_history():
    """State with last_tool_action and successful_mutations -> subgoal/next_steps consistent with continuing from last success."""
    task = Task(user_id="u1", actions=[], instruction="Book a flight", outputs=[])
    state = create_initial_task_state(domain="airline", task=task)
    state.last_tool_action = "get_user_details"
    state.successful_mutations.append("get_user_details")
    state.grounded["user_id"] = "mia_li_3668"
    state.identity.profile_grounded = True
    result = plan(state, None, step_index=3, max_num_steps=30)
    assert result.current_goal == "Book a flight"
    assert "Book" in result.current_goal or "flight" in result.current_goal.lower()
    assert 1 <= len(result.next_steps) <= 3


def test_plan_replan_trigger_after_no_progress():
    """RecoveryState with recent_last_actions indicating no progress -> replan_triggers or anti_patterns mention repeated action."""
    task = Task(user_id="u1", actions=[], instruction="Do something", outputs=[])
    state = create_initial_task_state(domain="retail", task=task)
    recovery = RecoveryState(recent_last_actions=["get_user_details", "get_user_details", "get_user_details"])
    result = plan(state, recovery, step_index=5, max_num_steps=30)
    assert any(
        "repeated" in t or "no_progress" in t or "same" in t.lower()
        for t in result.replan_triggers
    ) or any(
        "repeat" in a.lower() or "progress" in a.lower() or "same" in a.lower()
        for a in result.anti_patterns_to_avoid
    )


def test_plan_blocked_prerequisite_produces_targeted_subgoal_and_tool_call_preference():
    """When recovery_state has blocked_goal_action + missing_prerequisites, subgoal and next_steps target satisfying prerequisite."""
    from tau_bench.types import Action

    task = Task(user_id="u1", actions=[], instruction="Book a flight", outputs=[])
    state = create_initial_task_state(domain="airline", task=task)
    blocked_action = Action(name="book_reservation", kwargs={"user_id": "u1"})
    recovery = RecoveryState(
        blocked_goal_action=blocked_action,
        missing_prerequisites=["profile_grounded"],
        resume_intent_after_prereq=True,
    )
    result = plan(state, recovery, step_index=2, max_num_steps=30)
    assert "profile" in result.subgoal.lower() or "prerequisite" in result.subgoal.lower()
    assert "book_reservation" in result.subgoal
    assert result.preferred_next_action_type == "tool_call"
    assert any("prerequisite" in s.rationale.lower() or "satisfy" in s.rationale.lower() for s in result.next_steps)
    assert any("Do not retry" in a or "prerequisite" in a.lower() for a in result.anti_patterns_to_avoid)


def test_plan_prefers_ask_user_when_subject_ambiguous():
    """When subject_resolution_status is ambiguous, preferred_next_action_type is ask_user."""
    task = Task(user_id="u1", actions=[], instruction="Change my booking", outputs=[])
    state = create_initial_task_state(domain="airline", task=task)
    state.subject_resolution_status = "ambiguous"
    result = plan(state, None, step_index=2, max_num_steps=30)
    assert result.preferred_next_action_type == "ask_user"


def test_build_planner_guidance_text_contains_subgoal_and_preferred():
    """build_planner_guidance_text produces string containing subgoal and preferred next."""
    task = Task(user_id="u1", actions=[], instruction="Book flight", outputs=[])
    state = create_initial_task_state(domain="airline", task=task)
    plan_result = plan(state, None, step_index=1, max_num_steps=30)
    text = build_planner_guidance_text(plan_result)
    assert "Plan:" in text
    assert plan_result.subgoal[:50] in text or plan_result.subgoal in text
    assert plan_result.preferred_next_action_type in text


def test_proposer_integration_planner_guidance_in_messages():
    """Run loop: planner is called and message passed to proposer contains planner guidance (e.g. Plan: or subgoal)."""
    from tau_bench.orchestration.run_loop import run_orchestrated_loop
    from tau_bench.types import SolveResult

    captured_messages = []

    class SpyProposer:
        def generate_next_step(self, messages):
            captured_messages.append(list(messages))
            return (
                {"role": "assistant", "content": "I will help. ###STOP###"},
                Action(name=RESPOND_ACTION_NAME, kwargs={"content": "I will help. ###STOP###"}),
                0.0,
            )

    mock_env = MagicMock()
    mock_env.wiki = "# Policy"
    mock_env.task = Task(user_id="u1", actions=[], instruction="Book a flight", outputs=[])
    mock_env.tools_map = {}
    mock_env.tools_info = []
    mock_env.reset.return_value = MagicMock(
        observation="I want to book a flight",
        info=MagicMock(model_dump=lambda: {}),
    )
    mock_env.step.return_value = MagicMock(
        observation="###STOP###",
        reward=1.0,
        done=True,
        info=MagicMock(model_dump=lambda: {}),
    )

    mock_logger = MagicMock()
    result = run_orchestrated_loop(
        env=mock_env,
        proposer=SpyProposer(),
        run_logger=mock_logger,
        task_index=0,
        max_num_steps=5,
        domain="airline",
    )

    assert isinstance(result, SolveResult)
    assert len(captured_messages) >= 1
    last_messages = captured_messages[-1]
    content_parts = []
    for m in last_messages:
        c = m.get("content")
        if isinstance(c, str):
            content_parts.append(c)
    full_content = " ".join(content_parts)
    assert "Plan:" in full_content
    assert "Goal:" in full_content or "Subgoal:" in full_content

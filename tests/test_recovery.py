# Copyright Sierra
# Unit tests for Recovery module (Phase 3 orchestration).

import pytest
from tau_bench.types import Action
from tau_bench.orchestration.recovery import (
    FailureCategory,
    RecoveryStrategy,
    RecoveryState,
    RecoveryInput,
    RecoveryDecision,
    RecoveryConfig,
    action_retry_key,
    default_recovery_config,
    decide_recovery,
    detect_confirmation_satisfied,
)


def test_recovery_state_defaults():
    """RecoveryState has expected default fields."""
    state = RecoveryState()
    assert state.retry_counts == {}
    assert state.failure_type_counts == {}
    assert state.pending_side_effect_action is None
    assert state.pending_confirmation_key is None
    assert state.pending_since_step == 0
    assert state.recovery_count_this_run == 0
    assert state.last_blocked_retry_key is None


def test_decide_recovery_validation_error_returns_replan():
    """For validation_error, decide_recovery returns REPLAN_FROM_STATE (Phase A stub)."""
    action = Action(name="unknown_tool", kwargs={})
    rec_input = RecoveryInput(
        failure_type=FailureCategory.validation_error.value,
        action=action,
        step_index=1,
        max_num_steps=30,
        source_code="tool_not_found",
        source_message="unknown tool: unknown_tool",
    )
    config = default_recovery_config("airline")
    decision = decide_recovery(rec_input, config)
    assert isinstance(decision, RecoveryDecision)
    assert decision.failure_type == FailureCategory.validation_error.value
    assert decision.proposed_strategy == RecoveryStrategy.REPLAN_FROM_STATE.value
    assert decision.recoverable is True
    assert "Validation failed" in decision.diagnosis
    assert decision.retry_key == action_retry_key(action)
    assert "retry_key" in decision.trace_metadata
    assert "failure_code" in decision.trace_metadata


def test_decide_recovery_policy_block_missing_confirmation_returns_ask_user_confirmation():
    """For policy_block with missing_confirmation + side-effecting action, returns ASK_USER_CONFIRMATION (Phase B)."""
    action = Action(name="book_reservation", kwargs={"user_id": "u1"})
    rec_input = RecoveryInput(
        failure_type=FailureCategory.policy_block.value,
        action=action,
        step_index=2,
        max_num_steps=30,
        source_code="missing_confirmation",
        source_message="explicit user confirmation required before booking",
        missing_prerequisites=["booking_confirmed"],
        domain="airline",
    )
    config = default_recovery_config("airline")
    decision = decide_recovery(rec_input, config)
    assert decision.proposed_strategy == RecoveryStrategy.ASK_USER_CONFIRMATION.value
    assert "set_pending_side_effect_action" in decision.state_updates
    assert decision.state_updates.get("pending_confirmation_key") == "booking_confirmed"
    assert decision.message_to_user is not None
    assert decision.retry_allowed is True


def test_decide_recovery_policy_block_missing_user_id_returns_replan():
    """For policy_block with missing_user_id (not confirmation), returns REPLAN_FROM_STATE."""
    action = Action(name="book_reservation", kwargs={"user_id": "u1"})
    rec_input = RecoveryInput(
        failure_type=FailureCategory.policy_block.value,
        action=action,
        step_index=2,
        max_num_steps=30,
        source_code="missing_user_id",
        source_message="user_id must be established",
        missing_prerequisites=["user_id"],
        domain="airline",
    )
    config = default_recovery_config("airline")
    decision = decide_recovery(rec_input, config)
    assert decision.proposed_strategy == RecoveryStrategy.REPLAN_FROM_STATE.value


def test_decide_recovery_repeated_same_action_blocked_again_returns_replan():
    """When same retry_key blocked again and pending is set, return REPLAN (repeated_same_action)."""
    action = Action(name="book_reservation", kwargs={"user_id": "u1"})
    pending = Action(name="book_reservation", kwargs={"user_id": "u1"})
    state = RecoveryState(pending_side_effect_action=pending, pending_confirmation_key="booking_confirmed")
    rec_input = RecoveryInput(
        failure_type=FailureCategory.policy_block.value,
        action=action,
        step_index=3,
        max_num_steps=30,
        source_code="missing_confirmation",
        source_message="confirmation required",
        missing_prerequisites=["booking_confirmed"],
        domain="airline",
        recovery_state=state,
    )
    config = default_recovery_config("airline")
    decision = decide_recovery(rec_input, config)
    assert decision.proposed_strategy == RecoveryStrategy.REPLAN_FROM_STATE.value
    assert decision.failure_type == FailureCategory.repeated_same_action.value


def test_recovery_decision_schema_usable_by_run_loop():
    """RecoveryDecision has all fields needed for trace and orchestrator."""
    decision = RecoveryDecision(
        failure_type="validation_error",
        diagnosis="test",
        confidence=0.8,
        recoverable=True,
        proposed_strategy=RecoveryStrategy.REPLAN_FROM_STATE.value,
        retry_key="book_reservation|{}",
        trace_metadata={"failure_code": "schema_mismatch"},
    )
    assert decision.terminal_reason is None
    assert decision.retry_budget_cost == 1
    # Trace event built from decision should have expected keys
    trace = {
        "failure_trigger": decision.failure_type,
        "diagnosis": decision.diagnosis,
        "chosen_strategy": decision.proposed_strategy,
        "retry_key": decision.retry_key,
        **decision.trace_metadata,
    }
    assert "failure_trigger" in trace
    assert "chosen_strategy" in trace
    assert trace["failure_code"] == "schema_mismatch"


def test_action_retry_key_same_action_same_key():
    """Same action produces same retry_key."""
    a1 = Action(name="book_reservation", kwargs={"user_id": "u1"})
    a2 = Action(name="book_reservation", kwargs={"user_id": "u1"})
    assert action_retry_key(a1) == action_retry_key(a2)


def test_action_retry_key_different_kwargs_different_key():
    """Different kwargs produce different retry_key."""
    a1 = Action(name="book_reservation", kwargs={"user_id": "u1"})
    a2 = Action(name="book_reservation", kwargs={"user_id": "u2"})
    assert action_retry_key(a1) != action_retry_key(a2)


def test_budget_exhausted_returns_safe_terminate():
    """When recovery_count_this_run >= max_recovery_per_run, return SAFE_TERMINATE."""
    state = RecoveryState(recovery_count_this_run=10)
    action = Action(name="book_reservation", kwargs={"user_id": "u1"})
    rec_input = RecoveryInput(
        failure_type=FailureCategory.policy_block.value,
        action=action,
        step_index=5,
        max_num_steps=30,
        source_code="missing_confirmation",
        source_message="confirmation required",
        recovery_state=state,
    )
    config = RecoveryConfig(max_recovery_per_run=10)
    decision = decide_recovery(rec_input, config)
    assert decision.proposed_strategy == RecoveryStrategy.SAFE_TERMINATE.value
    assert decision.recoverable is False
    assert decision.terminal_reason == "max_recovery_per_run"


def test_turn_limit_risk_returns_safe_terminate():
    """When step_index >= max_num_steps - 1, return SAFE_TERMINATE."""
    action = Action(name="respond", kwargs={"content": "ok"})
    rec_input = RecoveryInput(
        failure_type=FailureCategory.validation_error.value,
        action=action,
        step_index=29,
        max_num_steps=30,
        source_code="schema_mismatch",
        source_message="missing content",
    )
    config = default_recovery_config("airline")
    decision = decide_recovery(rec_input, config)
    assert decision.proposed_strategy == RecoveryStrategy.SAFE_TERMINATE.value
    assert decision.terminal_reason == "turn_limit_risk"


def test_detect_confirmation_satisfied_yes():
    """User saying yes is detected as confirmation."""
    assert detect_confirmation_satisfied("yes", "booking_confirmed") is True
    assert detect_confirmation_satisfied("Yes please", "booking_confirmed") is True
    assert detect_confirmation_satisfied("go ahead", "booking_confirmed") is True
    assert detect_confirmation_satisfied("please do", "booking_confirmed") is True


def test_detect_confirmation_satisfied_no():
    """User saying no is not confirmation."""
    assert detect_confirmation_satisfied("no", "booking_confirmed") is False
    assert detect_confirmation_satisfied("don't", "booking_confirmed") is False
    assert detect_confirmation_satisfied("cancel", "booking_confirmed") is False


def test_default_recovery_config_domain_airline():
    """default_recovery_config(airline) has airline side-effecting tools."""
    config = default_recovery_config("airline")
    assert "book_reservation" in config.side_effecting_tools
    assert "cancel_reservation" in config.side_effecting_tools


def test_default_recovery_config_domain_retail():
    """default_recovery_config(retail) has retail side-effecting tools."""
    config = default_recovery_config("retail")
    assert "cancel_pending_order" in config.side_effecting_tools


def test_is_no_progress_true_when_last_n_same():
    """is_no_progress True when last NO_PROGRESS_WINDOW actions are the same."""
    from tau_bench.orchestration.recovery import is_no_progress
    state = RecoveryState(recent_last_actions=["book_reservation", "book_reservation", "book_reservation"])
    assert is_no_progress(state) is True


def test_is_no_progress_false_when_insufficient_steps():
    """is_no_progress False when fewer than window steps."""
    from tau_bench.orchestration.recovery import is_no_progress
    state = RecoveryState(recent_last_actions=["book_reservation", "book_reservation"])
    assert is_no_progress(state, window=3) is False


def test_decide_recovery_tool_execution_error_returns_replan():
    """For tool_execution_error, decide_recovery returns REPLAN_FROM_STATE (Phase D)."""
    action = Action(name="book_reservation", kwargs={"user_id": "u1"})
    rec_input = RecoveryInput(
        failure_type="tool_execution_error",
        action=action,
        step_index=2,
        max_num_steps=30,
        tool_observation_summary="Error: payment failed",
        source_message="Error: payment failed",
    )
    config = default_recovery_config("airline")
    decision = decide_recovery(rec_input, config)
    assert decision.proposed_strategy == RecoveryStrategy.REPLAN_FROM_STATE.value
    assert "Tool execution failed" in decision.diagnosis


def test_run_loop_recovery_trace_has_recovery_decision_on_policy_block():
    """Integration: run_loop with use_recovery=True logs recovery_decision on policy block."""
    from unittest.mock import MagicMock, patch
    from tau_bench.types import Task
    from tau_bench.orchestration.run_loop import run_orchestrated_loop
    from tau_bench.orchestration.task_state import create_initial_task_state

    trace_events = []
    mock_logger = MagicMock()
    mock_logger.write_trace_event = lambda e: trace_events.append(e)
    mock_logger.log_run_start = MagicMock()
    mock_logger.log_step_stage = MagicMock()
    mock_logger.finish_run = MagicMock()

    mock_env = MagicMock()
    mock_env.wiki = "# Policy"
    mock_env.task = Task(user_id="u1", actions=[], instruction="Book flight", outputs=[])
    mock_env.tools_map = {"book_reservation": None}
    mock_env.tools_info = [
        {"type": "function", "function": {"name": "book_reservation", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}},
    ]
    mock_env.reset.return_value = MagicMock(observation="Book a flight", info=MagicMock(model_dump=lambda: {}))

    class ProposeBookProposer:
        def generate_next_step(self, messages):
            return (
                {"role": "assistant", "tool_calls": [{"id": "tc_1", "function": {"name": "book_reservation", "arguments": "{\"user_id\":\"sara_doe_496\"}"}}]},
                Action(name="book_reservation", kwargs={"user_id": "sara_doe_496"}),
                0.0,
            )

    run_orchestrated_loop(env=mock_env, proposer=ProposeBookProposer(), run_logger=mock_logger, task_index=0, max_num_steps=5, domain="airline", use_recovery=True)
    recovery_events = [e for e in trace_events if e.get("module") == "recovery" and e.get("event_type") == "recovery_decision"]
    assert len(recovery_events) >= 1
    # With max_num_steps=5, step 1 does not hit turn_limit_risk, so we get policy-block strategy (REPLAN for missing_user_id)
    assert recovery_events[0].get("chosen_strategy") in ("REPLAN_FROM_STATE", "ASK_USER_CONFIRMATION")

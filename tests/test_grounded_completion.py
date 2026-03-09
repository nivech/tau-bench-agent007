# Copyright Sierra
# Tests for grounded completion guard and subject-resolution (Phase 3).

import pytest
from unittest.mock import MagicMock

from tau_bench.types import Action, Task, RESPOND_ACTION_NAME
from tau_bench.orchestration.task_state import TaskState, create_initial_task_state
from tau_bench.orchestration.tool_outcomes import (
    ToolOutcome,
    get_mutating_tools,
    is_mutating_tool,
    classify_observation,
    is_success_style_respond,
)
from tau_bench.orchestration.policy_guard import (
    check_policy,
    CODE_SUBJECT_AMBIGUITY,
)
from tau_bench.orchestration.recovery import (
    get_completion_guard_recovery_message,
    decide_recovery,
    RecoveryInput,
    RecoveryConfig,
    FailureCategory,
    RecoveryStrategy,
)


# ---- TaskState ----
def test_task_state_record_mutating_attempt_and_success():
    """record_mutating_attempt and record_successful_mutation update state."""
    state = TaskState(domain="airline")
    assert len(state.attempted_mutating_tools) == 0
    assert len(state.successful_mutations) == 0
    state.record_mutating_attempt("book_reservation")
    assert "book_reservation" in state.attempted_mutating_tools
    state.record_successful_mutation("book_reservation")
    assert "book_reservation" in state.successful_mutations
    assert state.requires_grounded_completion(None) is False


def test_requires_grounded_completion_true_when_attempted_no_success():
    """requires_grounded_completion is True when mutating attempted but no success."""
    state = TaskState(domain="airline")
    state.record_mutating_attempt("book_reservation")
    assert state.requires_grounded_completion(None) is True
    state.record_successful_mutation("book_reservation")
    assert state.requires_grounded_completion(None) is False


def test_requires_grounded_completion_true_when_pending_side_effect():
    """requires_grounded_completion is True when pending_side_effect_action is set."""
    state = TaskState(domain="airline")
    pending = Action(name="book_reservation", kwargs={"user_id": "u1"})
    assert state.requires_grounded_completion(pending) is True


# ---- tool_outcomes ----
def test_is_mutating_tool_airline():
    assert is_mutating_tool("airline", "book_reservation") is True
    assert is_mutating_tool("airline", "get_user_details") is False


def test_is_mutating_tool_retail():
    assert is_mutating_tool("retail", "cancel_pending_order") is True
    assert is_mutating_tool("retail", "get_order_details") is False


def test_classify_observation_success():
    outcome = classify_observation("airline", "book_reservation", '{"reservation_id": "ABC"}')
    assert outcome.is_mutating is True
    assert outcome.execution_succeeded is True
    assert outcome.state_change_confirmed is True


def test_classify_observation_error():
    outcome = classify_observation("airline", "book_reservation", "Error: payment not found")
    assert outcome.is_mutating is True
    assert outcome.execution_succeeded is False
    assert outcome.state_change_confirmed is False


def test_synthetic_tool_content_does_not_count_as_grounded_success():
    """Only real env.step(tool) results update successful_mutations; grounding layer does not."""
    from tau_bench.orchestration.grounding import apply_grounding

    mock_env = MagicMock()
    mock_env.tools_map = {"get_user_details": None}  # not a mutating tool; use one that has extractor
    state = TaskState(domain="airline")
    action = Action(name="get_user_details", kwargs={"user_id": "u1"})
    # Even if we had a mutating tool and success-like observation, apply_grounding never calls
    # record_successful_mutation - only run_loop does after env.step. So synthetic tool content
    # in messages (e.g. "Your reservation is confirmed") never adds to successful_mutations.
    apply_grounding(mock_env, "airline", action, '{"reservation_id": "R1"}', state)
    assert len(state.successful_mutations) == 0
    assert state.attempted_mutating_tools == set()


def test_is_success_style_respond():
    assert is_success_style_respond("Your booking is confirmed.") is True
    assert is_success_style_respond("Order completed successfully.") is True
    assert is_success_style_respond("Here is the information you asked for.") is False
    assert is_success_style_respond("OK") is False  # too short
    assert is_success_style_respond("") is False


# ---- Completion guard recovery message ----
def test_get_completion_guard_recovery_message():
    msg = get_completion_guard_recovery_message()
    assert "state-changing" in msg
    assert "Do not claim completion" in msg


# ---- Completion guard blocked recovery decision ----
def test_decide_recovery_completion_guard_blocked():
    action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Your booking is confirmed."})
    rec_input = RecoveryInput(
        failure_type=FailureCategory.completion_guard_blocked.value,
        action=action,
        step_index=3,
        max_num_steps=30,
    )
    config = RecoveryConfig()
    decision = decide_recovery(rec_input, config)
    assert decision.proposed_strategy == RecoveryStrategy.REPLAN_FROM_STATE.value
    assert decision.message_to_user is not None
    assert "Do not claim completion" in (decision.message_to_user or "")


# ---- Subject ambiguity ----
def test_decide_recovery_subject_ambiguity_via_policy_block():
    """When policy returns subject_ambiguity, decide_recovery returns subject_ambiguity decision."""
    action = Action(name="cancel_reservation", kwargs={"reservation_id": "UNKNOWN_ID"})
    rec_input = RecoveryInput(
        failure_type=FailureCategory.policy_block.value,
        action=action,
        step_index=2,
        max_num_steps=30,
        source_code=CODE_SUBJECT_AMBIGUITY,
        source_message="Resolve the target entity...",
        domain="airline",
    )
    config = RecoveryConfig()
    decision = decide_recovery(rec_input, config)
    assert decision.failure_type == FailureCategory.subject_ambiguity.value
    assert "Resolve the target entity" in (decision.message_to_user or "")


def test_policy_guard_subject_ambiguity_reservation_id_not_grounded():
    """Policy guard blocks cancel_reservation when reservation_id not in grounded reservation_ids."""
    mock_env = MagicMock()
    mock_env.tools_map = {"cancel_reservation": None}
    state = TaskState(domain="airline")
    state.identity.user_id = "u1"
    state.grounded["reservation_ids"] = ["R1", "R2"]
    action = Action(name="cancel_reservation", kwargs={"reservation_id": "R99"})
    result = check_policy(mock_env, action, state)
    assert result.allowed is False
    assert result.code == CODE_SUBJECT_AMBIGUITY


def test_policy_guard_subject_resolved_reservation_id_in_grounded():
    """Policy guard allows cancel_reservation when reservation_id is in grounded reservation_ids."""
    mock_env = MagicMock()
    mock_env.tools_map = {"cancel_reservation": None}
    state = TaskState(domain="airline")
    state.identity.user_id = "u1"
    state.grounded["reservation_ids"] = ["R1", "R2"]
    action = Action(name="cancel_reservation", kwargs={"reservation_id": "R1"})
    result = check_policy(mock_env, action, state)
    assert result.allowed is True


def test_policy_guard_subject_no_grounded_ids_allows():
    """When no grounded reservation_ids yet, policy does not block on subject (let tool handle)."""
    mock_env = MagicMock()
    mock_env.tools_map = {"cancel_reservation": None}
    state = TaskState(domain="airline")
    state.identity.user_id = "u1"
    state.grounded["reservation_ids"] = []
    action = Action(name="cancel_reservation", kwargs={"reservation_id": "R1"})
    result = check_policy(mock_env, action, state)
    assert result.allowed is True


# ---- Run loop integration: blocked mutation + false success (mocked) ----
def test_completion_guard_blocks_false_success_respond():
    """When mutating tool was blocked and agent proposes success-style respond, guard blocks (no env.step)."""
    from tau_bench.orchestration.run_loop import run_orchestrated_loop

    trace_events = []
    step_calls = []
    mock_logger = MagicMock()
    mock_logger.write_trace_event = lambda e: trace_events.append(e)
    mock_logger.log_run_start = MagicMock()
    mock_logger.log_step_stage = MagicMock()
    mock_logger.finish_run = MagicMock()

    mock_env = MagicMock()
    mock_env.wiki = "# Policy"
    mock_env.task = Task(user_id="u1", actions=[], instruction="Book a flight", outputs=[])
    mock_env.tools_map = {"book_reservation": None}
    mock_env.tools_info = [
        {
            "type": "function",
            "function": {
                "name": "book_reservation",
                "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]},
            },
        },
    ]
    mock_env.reset.return_value = MagicMock(observation="Book a flight", info=MagicMock(model_dump=lambda: {}))

    def mock_step(action):
        step_calls.append(action.name)
        if action.name == RESPOND_ACTION_NAME:
            return MagicMock(observation="Thanks", reward=0.0, done=False, info=MagicMock(model_dump=lambda: {}))
        return MagicMock(observation="Error", reward=0.0, done=False, info=MagicMock(model_dump=lambda: {}))

    mock_env.step = mock_step

    # Proposer: first step book_reservation (will be blocked), second step respond("Your booking is confirmed.")
    call_count = [0]

    class ProposerBlockThenSuccess:
        def generate_next_step(self, messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "tc_1", "function": {"name": "book_reservation", "arguments": '{"user_id":"sara_doe_496"}'}}
                        ],
                    },
                    Action(name="book_reservation", kwargs={"user_id": "sara_doe_496"}),
                    0.0,
                )
            return (
                {"role": "assistant", "content": "Your booking is confirmed."},
                Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Your booking is confirmed."}),
                0.0,
            )

    run_orchestrated_loop(
        env=mock_env,
        proposer=ProposerBlockThenSuccess(),
        run_logger=mock_logger,
        task_index=0,
        max_num_steps=5,
        domain="airline",
        use_recovery=True,
    )
    # env.step(respond) must NOT have been called because completion guard blocks
    assert RESPOND_ACTION_NAME not in step_calls
    completion_guard_events = [e for e in trace_events if e.get("event_type") == "completion_guard_blocked"]
    assert len(completion_guard_events) >= 1


def test_read_only_task_respond_allowed():
    """Pure read-only task: respond without any mutating attempt is allowed (env.step(respond) called)."""
    from tau_bench.orchestration.run_loop import run_orchestrated_loop

    step_calls = []
    mock_logger = MagicMock()
    mock_logger.write_trace_event = MagicMock()
    mock_logger.log_run_start = MagicMock()
    mock_logger.log_step_stage = MagicMock()
    mock_logger.finish_run = MagicMock()

    mock_env = MagicMock()
    mock_env.wiki = "# Policy"
    mock_env.task = Task(user_id="u1", actions=[], instruction="What is my balance?", outputs=[])
    mock_env.tools_map = {}
    mock_env.tools_info = []
    mock_env.reset.return_value = MagicMock(observation="What is my balance?", info=MagicMock(model_dump=lambda: {}))

    def mock_step(action):
        step_calls.append(action.name)
        return MagicMock(observation="Bye", reward=0.0, done=False, info=MagicMock(model_dump=lambda: {}))

    mock_env.step = mock_step

    class ProposerRespondOnly:
        def generate_next_step(self, messages):
            return (
                {"role": "assistant", "content": "Here is the information you asked for."},
                Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Here is the information you asked for."}),
                0.0,
            )

    run_orchestrated_loop(
        env=mock_env,
        proposer=ProposerRespondOnly(),
        run_logger=mock_logger,
        task_index=0,
        max_num_steps=2,
        domain="airline",
        use_recovery=True,
    )
    # respond should have been passed to env.step (no mutating attempt, so no guard)
    assert RESPOND_ACTION_NAME in step_calls

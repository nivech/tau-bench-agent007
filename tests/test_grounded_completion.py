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
    is_explicit_completion_outcome_claim,
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


def test_is_explicit_completion_outcome_claim():
    assert is_explicit_completion_outcome_claim("Your booking is confirmed.") is True
    assert is_explicit_completion_outcome_claim("The reservation is confirmed.") is True
    assert is_explicit_completion_outcome_claim("Order is complete.") is True
    assert is_explicit_completion_outcome_claim("Please confirm you want to proceed.") is False
    assert is_explicit_completion_outcome_claim("Reply yes to confirm.") is False
    assert is_explicit_completion_outcome_claim("Yes, I confirm I want to proceed.") is False
    assert is_explicit_completion_outcome_claim("") is False


# ---- Completion guard recovery message ----
def test_get_completion_guard_recovery_message():
    msg = get_completion_guard_recovery_message()
    assert "state-changing" in msg
    assert "Do not confirm completion" in msg or "Do not claim completion" in msg
    assert "Continue from" in msg or "complete the required action" in msg


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
    assert "state-changing" in (decision.message_to_user or "") and ("Do not confirm" in (decision.message_to_user or "") or "complete the required action" in (decision.message_to_user or ""))


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
def test_completion_guard_allows_respond_when_waiting_for_confirmation():
    """When we're waiting for user confirmation (pending_side_effect_action), completion guard does not block
    the respond so env.step(respond) runs and the user reply can be appended as role='user' for Phase B."""
    from tau_bench.orchestration.run_loop import run_orchestrated_loop

    step_calls = []
    mock_logger = MagicMock()
    mock_logger.write_trace_event = MagicMock()
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
                {"role": "assistant", "content": "Please confirm you want to proceed with this booking. Reply yes to confirm."},
                Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Please confirm you want to proceed with this booking. Reply yes to confirm."}),
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
    # When awaiting user input we allow confirmation-seeking respond through so the user can reply
    assert RESPOND_ACTION_NAME in step_calls


def test_completion_guard_blocks_explicit_outcome_claim_even_when_awaiting_confirmation():
    """When awaiting confirmation, explicit outcome claims like 'Your booking is confirmed' are still blocked."""
    from tau_bench.orchestration.run_loop import run_orchestrated_loop

    step_calls = []
    mock_logger = MagicMock()
    mock_logger.write_trace_event = MagicMock()
    mock_logger.log_run_start = MagicMock()
    mock_logger.log_step_stage = MagicMock()
    mock_logger.finish_run = MagicMock()

    mock_env = MagicMock()
    mock_env.wiki = "# Policy"
    mock_env.task = Task(user_id="u1", actions=[], instruction="Book a flight", outputs=[])
    mock_env.tools_map = {"book_reservation": None}
    mock_env.tools_info = [
        {"type": "function", "function": {"name": "book_reservation", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}},
    ]
    mock_env.reset.return_value = MagicMock(observation="Book a flight", info=MagicMock(model_dump=lambda: {}))

    def mock_step(action):
        step_calls.append(action.name)
        return MagicMock(observation="Thanks", reward=0.0, done=False, info=MagicMock(model_dump=lambda: {}))

    mock_env.step = mock_step

    call_count = [0]

    class ProposerOutcomeClaimWhileWaiting:
        def generate_next_step(self, messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    {"role": "assistant", "tool_calls": [{"id": "tc_1", "function": {"name": "book_reservation", "arguments": '{"user_id":"u1"}'}}]},
                    Action(name="book_reservation", kwargs={"user_id": "u1"}),
                    0.0,
                )
            return (
                {"role": "assistant", "content": "Your booking is confirmed."},
                Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Your booking is confirmed."}),
                0.0,
            )

    run_orchestrated_loop(
        env=mock_env,
        proposer=ProposerOutcomeClaimWhileWaiting(),
        run_logger=mock_logger,
        task_index=0,
        max_num_steps=5,
        domain="airline",
        use_recovery=True,
    )
    # Explicit outcome claim is blocked even when awaiting_user_input
    assert RESPOND_ACTION_NAME not in step_calls


def test_clarification_respond_passes_when_awaiting_user_input():
    """When recovery is awaiting user input (e.g. after confirmation prompt), clarification-style respond is not blocked."""
    from tau_bench.orchestration.run_loop import run_orchestrated_loop

    step_calls = []
    mock_logger = MagicMock()
    mock_logger.write_trace_event = MagicMock()
    mock_logger.log_run_start = MagicMock()
    mock_logger.log_step_stage = MagicMock()
    mock_logger.finish_run = MagicMock()

    mock_env = MagicMock()
    mock_env.wiki = "# Policy"
    mock_env.task = Task(user_id="u1", actions=[], instruction="Book a flight", outputs=[])
    mock_env.tools_map = {"book_reservation": None}
    mock_env.tools_info = [
        {"type": "function", "function": {"name": "book_reservation", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}},
    ]
    mock_env.reset.return_value = MagicMock(observation="Book a flight", info=MagicMock(model_dump=lambda: {}))

    def mock_step(action):
        step_calls.append(action.name)
        return MagicMock(observation="May 20", reward=0.0, done=False, info=MagicMock(model_dump=lambda: {}))

    mock_env.step = mock_step

    call_count = [0]

    class ProposerClarificationWhenWaiting:
        def generate_next_step(self, messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    {"role": "assistant", "tool_calls": [{"id": "tc_1", "function": {"name": "book_reservation", "arguments": '{"user_id":"u1"}'}}]},
                    Action(name="book_reservation", kwargs={"user_id": "u1"}),
                    0.0,
                )
            return (
                {"role": "assistant", "content": "What date would you like to travel? Please provide the date."},
                Action(name=RESPOND_ACTION_NAME, kwargs={"content": "What date would you like to travel? Please provide the date."}),
                0.0,
            )

    run_orchestrated_loop(
        env=mock_env,
        proposer=ProposerClarificationWhenWaiting(),
        run_logger=mock_logger,
        task_index=0,
        max_num_steps=5,
        domain="airline",
        use_recovery=True,
    )
    # Clarification respond (not success-style) passes; when awaiting_user_input any non-outcome-claim respond passes
    assert RESPOND_ACTION_NAME in step_calls


def test_confirmation_e2e_user_says_yes_then_booking_allowed():
    """E2E: grounding step (get_user_details) → profile_grounded; first book_reservation blocked for missing_confirmation;
    recovery ASK_USER_CONFIRMATION; respond reaches env, user says 'yes'; Phase B records confirmation;
    orchestrator retries → book_reservation executed."""
    from tau_bench.orchestration.run_loop import run_orchestrated_loop

    step_calls = []
    user_replies = []
    trace_events = []

    mock_logger = MagicMock()
    mock_logger.write_trace_event = lambda e: trace_events.append(e)
    mock_logger.log_run_start = MagicMock()
    mock_logger.log_step_stage = MagicMock()
    mock_logger.finish_run = MagicMock()

    mock_env = MagicMock()
    mock_env.wiki = "# Policy"
    mock_env.task = Task(user_id="u1", actions=[], instruction="Book a flight", outputs=[])
    mock_env.tools_map = {"get_user_details": None, "book_reservation": None}
    mock_env.tools_info = [
        {
            "type": "function",
            "function": {
                "name": "get_user_details",
                "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]},
            },
        },
        {
            "type": "function",
            "function": {"name": "book_reservation", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}},
        },
    ]
    mock_env.reset.return_value = MagicMock(observation="Book a flight", info=MagicMock(model_dump=lambda: {}))

    def mock_step(action):
        step_calls.append(action.name)
        if action.name == RESPOND_ACTION_NAME:
            obs = "yes" if not user_replies else "thanks"
            user_replies.append(obs)
            return MagicMock(observation=obs, reward=0.0, done=False, info=MagicMock(model_dump=lambda: {}))
        if action.name == "get_user_details":
            return MagicMock(
                observation='{"payment_methods": {}, "dob": null, "membership": null, "reservations": [], "orders": []}',
                reward=0.0,
                done=False,
                info=MagicMock(model_dump=lambda: {}),
            )
        return MagicMock(observation='{"reservation_id": "R1"}', reward=0.0, done=False, info=MagicMock(model_dump=lambda: {}))

    mock_env.step = mock_step

    call_count = [0]

    class ProposerGroundThenConfirmThenBook:
        def generate_next_step(self, messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    {"role": "assistant", "tool_calls": [{"id": "tc_0", "function": {"name": "get_user_details", "arguments": '{"user_id":"u1"}'}}]},
                    Action(name="get_user_details", kwargs={"user_id": "u1"}),
                    0.0,
                )
            if call_count[0] == 2:
                return (
                    {"role": "assistant", "tool_calls": [{"id": "tc_1", "function": {"name": "book_reservation", "arguments": '{"user_id":"u1"}'}}]},
                    Action(name="book_reservation", kwargs={"user_id": "u1"}),
                    0.0,
                )
            if call_count[0] == 3:
                return (
                    {"role": "assistant", "content": "Do you want me to proceed with this booking? Reply yes to confirm."},
                    Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Do you want me to proceed with this booking? Reply yes to confirm."}),
                    0.0,
                )
            return (
                {"role": "assistant", "tool_calls": [{"id": "tc_2", "function": {"name": "book_reservation", "arguments": '{"user_id":"u1"}'}}]},
                Action(name="book_reservation", kwargs={"user_id": "u1"}),
                0.0,
            )

    result = run_orchestrated_loop(
        env=mock_env,
        proposer=ProposerGroundThenConfirmThenBook(),
        run_logger=mock_logger,
        task_index=0,
        max_num_steps=10,
        domain="airline",
        use_recovery=True,
    )
    # (0) Grounding step ran first so profile_grounded is set via apply_grounding
    assert "get_user_details" in step_calls, "Expected get_user_details to run first so policy allows book_reservation up to confirmation. step_calls=%s" % step_calls
    # (1) First book_reservation was blocked specifically for missing_confirmation (not missing_profile_grounding)
    blocked_missing_conf = [e for e in trace_events if e.get("event_type") == "blocked" and e.get("code") == "missing_confirmation"]
    assert len(blocked_missing_conf) >= 1, (
        "Expected at least one policy_guard 'blocked' with code missing_confirmation. events=%s"
        % [(e.get("event_type"), e.get("code")) for e in trace_events if e.get("module") == "policy_guard"]
    )
    # (2) Recovery chose ASK_USER_CONFIRMATION (sets pending_side_effect_action / pending_confirmation_key)
    ask_confirm_evts = [e for e in trace_events if e.get("event_type") == "recovery_decision" and e.get("chosen_strategy") == "ASK_USER_CONFIRMATION"]
    assert len(ask_confirm_evts) >= 1, (
        "Expected at least one recovery_decision with chosen_strategy ASK_USER_CONFIRMATION. events=%s"
        % [e.get("chosen_strategy") for e in trace_events if e.get("event_type") == "recovery_decision"]
    )
    # (3) Confirmation prompt reached env and user said "yes"
    assert RESPOND_ACTION_NAME in step_calls
    # (4) Orchestrator deterministically retried the pending action after confirmation (trace event)
    retry_evts = [e for e in trace_events if e.get("event_type") == "retry_after_confirmation"]
    assert len(retry_evts) >= 1, "Expected at least one retry_after_confirmation trace event. events=%s" % [e.get("event_type") for e in trace_events]
    assert any(e.get("action_name") == "book_reservation" for e in retry_evts)
    # (5) book_reservation was actually executed (env.step called)
    assert "book_reservation" in step_calls, (
        "Expected book_reservation in step_calls after user said 'yes'. step_calls=%s" % step_calls
    )


def test_missing_profile_grounding_prerequisite_recovery_then_retry():
    """When book_reservation is blocked for missing_profile_grounding, recovery sets SATISFY_PREREQUISITE;
    after get_user_details runs and satisfies profile_grounded, orchestrator schedules retry; no generic replan loop."""
    from tau_bench.orchestration.run_loop import run_orchestrated_loop
    from tau_bench.orchestration.recovery import RecoveryStrategy

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
    mock_env.tools_map = {"get_user_details": None, "book_reservation": None}
    mock_env.tools_info = [
        {"type": "function", "function": {"name": "get_user_details", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}},
        {"type": "function", "function": {"name": "book_reservation", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}},
    ]
    mock_env.reset.return_value = MagicMock(observation="Book a flight", info=MagicMock(model_dump=lambda: {}))

    def mock_step(action):
        step_calls.append(action.name)
        if action.name == RESPOND_ACTION_NAME:
            return MagicMock(observation="yes", reward=0.0, done=False, info=MagicMock(model_dump=lambda: {}))
        if action.name == "get_user_details":
            return MagicMock(
                observation='{"payment_methods": {"pm1": {}}, "dob": null, "membership": null, "reservations": [], "orders": []}',
                reward=0.0,
                done=False,
                info=MagicMock(model_dump=lambda: {}),
            )
        return MagicMock(observation='{"reservation_id": "R1"}', reward=0.0, done=False, info=MagicMock(model_dump=lambda: {}))

    mock_env.step = mock_step

    call_count = [0]

    class ProposerBookFirstThenGroundThenConfirm:
        """Proposer that tries book first (blocked), then get_user_details, then confirm."""
        def generate_next_step(self, messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    {"role": "assistant", "tool_calls": [{"id": "tc_1", "function": {"name": "book_reservation", "arguments": '{"user_id":"u1"}'}}]},
                    Action(name="book_reservation", kwargs={"user_id": "u1"}),
                    0.0,
                )
            if call_count[0] == 2:
                return (
                    {"role": "assistant", "tool_calls": [{"id": "tc_0", "function": {"name": "get_user_details", "arguments": '{"user_id":"u1"}'}}]},
                    Action(name="get_user_details", kwargs={"user_id": "u1"}),
                    0.0,
                )
            return (
                {"role": "assistant", "content": "Please confirm you want to proceed. Reply yes to confirm."},
                Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Please confirm you want to proceed. Reply yes to confirm."}),
                0.0,
            )

    run_orchestrated_loop(
        env=mock_env,
        proposer=ProposerBookFirstThenGroundThenConfirm(),
        run_logger=mock_logger,
        task_index=0,
        max_num_steps=12,
        domain="airline",
        use_recovery=True,
    )
    # First book is blocked (missing_user_id or missing_profile_grounding depending on state).
    satisy_prereq_evts = [e for e in trace_events if e.get("chosen_strategy") == RecoveryStrategy.SATISFY_PREREQUISITE.value]
    assert len(satisy_prereq_evts) >= 1, "Expected at least one SATISFY_PREREQUISITE recovery decision. events=%s" % [e.get("chosen_strategy") for e in trace_events if e.get("event_type") == "recovery_decision"]
    # get_user_details ran (satisfies user_id and profile_grounded).
    assert "get_user_details" in step_calls, "Expected get_user_details to run. step_calls=%s" % step_calls
    # Either retry_after_prereq or normal flow: book_reservation must eventually execute.
    assert "book_reservation" in step_calls, "Expected book_reservation to execute after prerequisite satisfied. step_calls=%s" % step_calls
    # Prerequisite satisfaction should trigger either scheduled retry or actual retry event.
    retry_prereq_evts = [e for e in trace_events if e.get("event_type") == "retry_after_prereq"]
    prereq_scheduled_evts = [e for e in trace_events if e.get("event_type") == "prereq_satisfied_retry_scheduled"]
    assert len(retry_prereq_evts) >= 1 or len(prereq_scheduled_evts) >= 1, (
        "Expected retry_after_prereq or prereq_satisfied_retry_scheduled when prerequisite is satisfied. events=%s"
        % [e.get("event_type") for e in trace_events]
    )


def test_non_confirming_user_reply_pending_remains_action_stays_blocked():
    """User says 'no' or 'not yet' → confirmation not recorded, pending remains, book_reservation still blocked."""
    from tau_bench.orchestration.run_loop import run_orchestrated_loop

    step_calls = []

    mock_logger = MagicMock()
    mock_logger.write_trace_event = MagicMock()
    mock_logger.log_run_start = MagicMock()
    mock_logger.log_step_stage = MagicMock()
    mock_logger.finish_run = MagicMock()

    mock_env = MagicMock()
    mock_env.wiki = "# Policy"
    mock_env.task = Task(user_id="u1", actions=[], instruction="Book a flight", outputs=[])
    mock_env.tools_map = {"book_reservation": None}
    mock_env.tools_info = [
        {"type": "function", "function": {"name": "book_reservation", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}},
    ]
    mock_env.reset.return_value = MagicMock(observation="Book a flight", info=MagicMock(model_dump=lambda: {}))

    def mock_step(action):
        step_calls.append(action.name)
        if action.name == RESPOND_ACTION_NAME:
            return MagicMock(observation="no", reward=0.0, done=False, info=MagicMock(model_dump=lambda: {}))
        return MagicMock(observation="Error", reward=0.0, done=False, info=MagicMock(model_dump=lambda: {}))

    mock_env.step = mock_step

    call_count = [0]

    class ProposerConfirmThenBookAgain:
        def generate_next_step(self, messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    {"role": "assistant", "tool_calls": [{"id": "tc_1", "function": {"name": "book_reservation", "arguments": '{"user_id":"u1"}'}}]},
                    Action(name="book_reservation", kwargs={"user_id": "u1"}),
                    0.0,
                )
            if call_count[0] == 2:
                return (
                    {"role": "assistant", "content": "Please confirm you want to proceed. Reply yes to confirm."},
                    Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Please confirm you want to proceed. Reply yes to confirm."}),
                    0.0,
                )
            return (
                {"role": "assistant", "tool_calls": [{"id": "tc_2", "function": {"name": "book_reservation", "arguments": '{"user_id":"u1"}'}}]},
                Action(name="book_reservation", kwargs={"user_id": "u1"}),
                0.0,
            )

    run_orchestrated_loop(
        env=mock_env,
        proposer=ProposerConfirmThenBookAgain(),
        run_logger=mock_logger,
        task_index=0,
        max_num_steps=6,
        domain="airline",
        use_recovery=True,
    )
    # We called respond once (env returned "no"). book_reservation is never executed (policy blocks it again)
    assert RESPOND_ACTION_NAME in step_calls
    assert "book_reservation" not in step_calls


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


def test_orchestrator_never_injects_synthetic_guidance_as_user():
    """Trust boundary: validation rejection, policy rejection, and completion-guard messages must not use role='user'."""
    from tau_bench.orchestration.run_loop import (
        run_orchestrated_loop,
        ORCHESTRATOR_GUIDANCE_ROLE,
        ORCHESTRATOR_GUIDANCE_PREFIX,
    )

    mock_logger = MagicMock()
    mock_logger.write_trace_event = MagicMock()
    mock_logger.log_run_start = MagicMock()
    mock_logger.log_step_stage = MagicMock()
    mock_logger.finish_run = MagicMock()

    # 1) Validation failure path: proposer suggests invalid tool -> rejection injected
    mock_env = MagicMock()
    mock_env.wiki = "# Policy"
    mock_env.task = Task(user_id="u1", actions=[], instruction="Book flight", outputs=[])
    mock_env.tools_map = {"book_reservation": None}
    mock_env.tools_info = [{"type": "function", "function": {"name": "book_reservation", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}}]
    mock_env.reset.return_value = MagicMock(observation="Book a flight", info=MagicMock(model_dump=lambda: {}))

    class ProposerInvalidTool:
        def generate_next_step(self, messages):
            return (
                {"role": "assistant", "tool_calls": [{"id": "tc1", "function": {"name": "nonexistent_tool", "arguments": "{}"}}]},
                Action(name="nonexistent_tool", kwargs={}),
                0.0,
            )

    result = run_orchestrated_loop(
        env=mock_env,
        proposer=ProposerInvalidTool(),
        run_logger=mock_logger,
        task_index=0,
        max_num_steps=2,
        domain="airline",
        use_recovery=True,
    )
    # After assistant + rejection, last message must not be user (tool branch for validation failure)
    assert len(result.messages) >= 3
    last = result.messages[-1]
    assert last.get("role") != "user", "Validation rejection must not be role='user'"
    # When tool_calls present we get role=tool; else we get runtime guidance
    if last.get("role") == ORCHESTRATOR_GUIDANCE_ROLE:
        assert (last.get("content") or "").startswith(ORCHESTRATOR_GUIDANCE_PREFIX)

    # 2) Policy block path: book_reservation blocked (e.g. missing_confirmation) -> rejection injected
    mock_env2 = MagicMock()
    mock_env2.wiki = "# Policy"
    mock_env2.task = Task(user_id="u1", actions=[], instruction="Book flight", outputs=[])
    mock_env2.tools_map = {"book_reservation": None}
    mock_env2.tools_info = [{"type": "function", "function": {"name": "book_reservation", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}}]
    mock_env2.reset.return_value = MagicMock(observation="Book a flight", info=MagicMock(model_dump=lambda: {}))

    class ProposerBookReservation:
        def generate_next_step(self, messages):
            return (
                {"role": "assistant", "tool_calls": [{"id": "tc1", "function": {"name": "book_reservation", "arguments": '{"user_id":"u1"}'}}]},
                Action(name="book_reservation", kwargs={"user_id": "u1"}),
                0.0,
            )

    result2 = run_orchestrated_loop(
        env=mock_env2,
        proposer=ProposerBookReservation(),
        run_logger=MagicMock(write_trace_event=MagicMock(), log_run_start=MagicMock(), log_step_stage=MagicMock(), finish_run=MagicMock()),
        task_index=0,
        max_num_steps=2,
        domain="airline",
        use_recovery=True,
    )
    last2 = result2.messages[-1]
    assert last2.get("role") != "user", "Policy block rejection must not be role='user'"
    if last2.get("role") == ORCHESTRATOR_GUIDANCE_ROLE:
        assert (last2.get("content") or "").startswith(ORCHESTRATOR_GUIDANCE_PREFIX)

    # 3) Completion guard path: blocked success-style respond -> recovery message injected
    mock_env3 = MagicMock()
    mock_env3.wiki = "# Policy"
    mock_env3.task = Task(user_id="u1", actions=[], instruction="Book a flight", outputs=[])
    mock_env3.tools_map = {"book_reservation": None}
    mock_env3.tools_info = [{"type": "function", "function": {"name": "book_reservation", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}}]
    mock_env3.reset.return_value = MagicMock(observation="Book a flight", info=MagicMock(model_dump=lambda: {}))

    call_count = [0]
    class ProposerBookThenSuccess:
        def generate_next_step(self, messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    {"role": "assistant", "tool_calls": [{"id": "tc1", "function": {"name": "book_reservation", "arguments": '{"user_id":"sara_doe_496"}'}}]},
                    Action(name="book_reservation", kwargs={"user_id": "sara_doe_496"}),
                    0.0,
                )
            return (
                {"role": "assistant", "content": "Your booking is confirmed."},
                Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Your booking is confirmed."}),
                0.0,
            )

    result3 = run_orchestrated_loop(
        env=mock_env3,
        proposer=ProposerBookThenSuccess(),
        run_logger=MagicMock(write_trace_event=MagicMock(), log_run_start=MagicMock(), log_step_stage=MagicMock(), finish_run=MagicMock()),
        task_index=0,
        max_num_steps=5,
        domain="airline",
        use_recovery=True,
    )
    # Completion guard injects runtime guidance when it blocks success-style respond
    last3 = result3.messages[-1]
    assert last3.get("role") == ORCHESTRATOR_GUIDANCE_ROLE, "Completion guard recovery message must not be role='user'"
    assert (last3.get("content") or "").startswith(ORCHESTRATOR_GUIDANCE_PREFIX)


def test_successful_mutation_unlocks_completion():
    """After real successful tool execution (e.g. book_reservation), success-style respond is allowed and run completes."""
    from tau_bench.orchestration.run_loop import run_orchestrated_loop

    step_calls = []
    call_count = [0]

    mock_logger = MagicMock()
    mock_logger.write_trace_event = MagicMock()
    mock_logger.log_run_start = MagicMock()
    mock_logger.log_step_stage = MagicMock()
    mock_logger.finish_run = MagicMock()

    mock_env = MagicMock()
    mock_env.wiki = "# Policy"
    mock_env.task = Task(user_id="u1", actions=[], instruction="Book a flight", outputs=[])
    mock_env.tools_map = {"get_user_details": None, "book_reservation": None}
    mock_env.tools_info = [
        {"type": "function", "function": {"name": "get_user_details", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}},
        {"type": "function", "function": {"name": "book_reservation", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}},
    ]
    mock_env.reset.return_value = MagicMock(observation="Book a flight", info=MagicMock(model_dump=lambda: {}))

    def mock_step(action):
        step_calls.append(action.name)
        if action.name == RESPOND_ACTION_NAME:
            if "book_reservation" in step_calls:
                return MagicMock(observation="Thank you", reward=1.0, done=True, info=MagicMock(model_dump=lambda: {}))
            return MagicMock(observation="yes", reward=0.0, done=False, info=MagicMock(model_dump=lambda: {}))
        if action.name == "get_user_details":
            return MagicMock(
                observation='{"payment_methods": {}, "dob": null, "membership": null, "reservations": [], "orders": []}',
                reward=0.0,
                done=False,
                info=MagicMock(model_dump=lambda: {}),
            )
        if action.name == "book_reservation":
            return MagicMock(observation='{"reservation_id": "R1"}', reward=0.0, done=False, info=MagicMock(model_dump=lambda: {}))
        return MagicMock(observation="ok", reward=0.0, done=False, info=MagicMock(model_dump=lambda: {}))

    mock_env.step = mock_step

    class ProposerFullFlow:
        def generate_next_step(self, messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    {"role": "assistant", "tool_calls": [{"id": "tc0", "function": {"name": "get_user_details", "arguments": '{"user_id":"u1"}'}}]},
                    Action(name="get_user_details", kwargs={"user_id": "u1"}),
                    0.0,
                )
            if call_count[0] == 2:
                return (
                    {"role": "assistant", "tool_calls": [{"id": "tc1", "function": {"name": "book_reservation", "arguments": '{"user_id":"u1"}'}}]},
                    Action(name="book_reservation", kwargs={"user_id": "u1"}),
                    0.0,
                )
            if call_count[0] == 3:
                return (
                    {"role": "assistant", "content": "Do you want to proceed? Reply yes to confirm."},
                    Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Do you want to proceed? Reply yes to confirm."}),
                    0.0,
                )
            if call_count[0] >= 4:
                return (
                    {"role": "assistant", "content": "Your booking is confirmed. Reservation R1."},
                    Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Your booking is confirmed. Reservation R1."}),
                    0.0,
                )
            return (
                {"role": "assistant", "content": "Done."},
                Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Done."}),
                0.0,
            )

    result = run_orchestrated_loop(
        env=mock_env,
        proposer=ProposerFullFlow(),
        run_logger=mock_logger,
        task_index=0,
        max_num_steps=12,
        domain="airline",
        use_recovery=True,
    )
    assert "get_user_details" in step_calls
    assert "book_reservation" in step_calls
    assert step_calls.count(RESPOND_ACTION_NAME) >= 2, "Respond should be called at least twice (confirm prompt + final success)"
    assert result.reward == 1.0


def test_confirmation_detection_only_considers_genuine_user_messages():
    """When last message is synthetic (tool or system), confirmation must not be applied; only role='user' counts."""
    from tau_bench.orchestration.run_loop import run_orchestrated_loop

    mock_env = MagicMock()
    mock_env.wiki = "# Policy"
    mock_env.task = Task(user_id="u1", actions=[], instruction="Book flight", outputs=[])
    mock_env.tools_map = {"book_reservation": None}
    mock_env.tools_info = [{"type": "function", "function": {"name": "book_reservation", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}}]
    mock_env.reset.return_value = MagicMock(observation="Book a flight", info=MagicMock(model_dump=lambda: {}))

    class ProposerBookOnly:
        def generate_next_step(self, messages):
            return (
                {"role": "assistant", "tool_calls": [{"id": "tc1", "function": {"name": "book_reservation", "arguments": '{"user_id":"u1"}'}}]},
                Action(name="book_reservation", kwargs={"user_id": "u1"}),
                0.0,
            )

    result = run_orchestrated_loop(
        env=mock_env,
        proposer=ProposerBookOnly(),
        run_logger=MagicMock(write_trace_event=MagicMock(), log_run_start=MagicMock(), log_step_stage=MagicMock(), finish_run=MagicMock()),
        task_index=0,
        max_num_steps=2,
        domain="airline",
        use_recovery=True,
    )
    # Policy block with tool_calls injects role="tool" (rejection as tool content); never "user".
    # Do not assert role == ORCHESTRATOR_GUIDANCE_ROLE: policy block path uses "tool", not "system".
    last_role = result.messages[-1].get("role")
    assert last_role != "user", "Synthetic orchestrator output must not be injected as role='user'"
    assert last_role in ("tool", "system"), "Synthetic rejection/guidance must be tool or system"

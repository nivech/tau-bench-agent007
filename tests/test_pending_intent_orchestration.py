from unittest.mock import MagicMock

import pytest

from tau_bench.types import Action, Task, RESPOND_ACTION_NAME
from tau_bench.orchestration.pending_intent import PendingIntent, PendingIntentStatus, TrustLevel
from tau_bench.orchestration.task_state import TaskState
from tau_bench.orchestration.action_args_builder import (
    ActionArgumentBuilder,
    ActionBuildError,
)
from tau_bench.orchestration.confirmation import (
    summarize_mutating_action,
    fingerprint_action_summary,
)
from tau_bench.orchestration.run_loop import run_orchestrated_loop


def _make_env_with_tools(tools):
    mock_env = MagicMock()
    mock_env.wiki = "# Policy"
    mock_env.task = Task(user_id="u1", actions=[], instruction="Task", outputs=[])
    mock_env.tools_map = {name: None for name in tools}
    mock_env.tools_info = [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]},
            },
        }
        for name in tools
    ]
    return mock_env


def test_builder_overrides_hallucinated_user_id_with_grounded_state():
    """Builder must ignore hallucinated user_id and use grounded TaskState.identity/grounded."""
    state = TaskState(domain="airline")
    state.identity.user_id = "grounded_user"
    pending = PendingIntent(
        action_name="book_reservation",
        status=PendingIntentStatus.NEEDS_CONFIRMATION,
        unresolved_requirements=["booking_confirmed"],
        metadata={
            "proposed_kwargs": {
                "value": {"user_id": "hallucinated_user"},
                "trust_level": TrustLevel.PROPOSER_DRAFT_UNTRUSTED.value,
            }
        },
    )
    builder = ActionArgumentBuilder()
    built = builder.build_action_args("book_reservation", state, pending)
    assert built.kwargs["user_id"] == "grounded_user"


def test_builder_rejects_untrusted_user_id_when_not_grounded():
    """When no grounded user_id exists, a proposer-only user_id must not be executed."""
    state = TaskState(domain="airline")
    pending = PendingIntent(
        action_name="book_reservation",
        status=PendingIntentStatus.NEEDS_CONFIRMATION,
        unresolved_requirements=["booking_confirmed"],
        metadata={
            "proposed_kwargs": {
                "value": {"user_id": "hallucinated_user"},
                "trust_level": TrustLevel.PROPOSER_DRAFT_UNTRUSTED.value,
            }
        },
    )
    builder = ActionArgumentBuilder()
    with pytest.raises(ActionBuildError):
        builder.build_action_args("book_reservation", state, pending)


def test_confirmation_fingerprint_mismatch_requires_reconfirmation():
    """If grounded state changes between confirmation and execution, fingerprint mismatch forces reconfirmation."""
    state = TaskState(domain="airline")
    state.identity.user_id = "u1"
    pending = PendingIntent(
        action_name="book_reservation",
        status=PendingIntentStatus.NEEDS_CONFIRMATION,
        unresolved_requirements=["booking_confirmed"],
        user_confirmed=True,
        metadata={
            "proposed_kwargs": {
                "value": {"user_id": "u1"},
                "trust_level": TrustLevel.PROPOSER_DRAFT_UNTRUSTED.value,
            }
        },
    )
    builder = ActionArgumentBuilder()
    first = builder.build_action_args("book_reservation", state, pending)
    summary1 = summarize_mutating_action("book_reservation", first.kwargs, state)
    fp1 = fingerprint_action_summary(summary1)
    pending.confirmation_fingerprint = fp1

    # Simulate state change that alters summary (e.g., new reservation_ids grounded)
    state.grounded["reservation_ids"] = ["R1"]
    second = builder.build_action_args("book_reservation", state, pending)
    summary2 = summarize_mutating_action("book_reservation", second.kwargs, state)
    fp2 = fingerprint_action_summary(summary2)

    assert fp1 != fp2


def test_read_only_action_bypasses_builder():
    """Read-only tools are not routed through ActionArgumentBuilder; existing flow remains."""
    trace_events = []
    mock_logger = MagicMock()
    mock_logger.write_trace_event = lambda e: trace_events.append(e)
    mock_logger.log_run_start = MagicMock()
    mock_logger.log_step_stage = MagicMock()
    mock_logger.finish_run = MagicMock()

    mock_env = _make_env_with_tools(["get_user_details"])
    mock_env.reset.return_value = MagicMock(observation="Get my profile", info=MagicMock(model_dump=lambda: {}))
    mock_env.step.return_value = MagicMock(observation="{}", reward=0.0, done=True, info=MagicMock(model_dump=lambda: {}))

    class ProposerReadOnly:
        def generate_next_step(self, messages):
            return (
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tc0",
                            "type": "function",
                            "function": {"name": "get_user_details", "arguments": '{"user_id":"u1"}'},
                        }
                    ],
                },
                Action(name="get_user_details", kwargs={"user_id": "u1"}),
                0.0,
            )

    result = run_orchestrated_loop(
        env=mock_env,
        proposer=ProposerReadOnly(),
        run_logger=mock_logger,
        task_index=0,
        max_num_steps=3,
        domain="airline",
        use_recovery=True,
    )
    # Sanity: run completed and tool was executed.
    assert result is not None
    assert any(e.get("module") == "executor" for e in trace_events)


# Copyright Sierra
# Generic tool-outcome classification for grounded completion guard.
# Uses policy guard's mutating-tool sets; no hardcoded tool names in completion logic.

from dataclasses import dataclass
from typing import Set

from tau_bench.orchestration.policy_guard import (
    AIRLINE_GUARDED_ACTIONS,
    RETAIL_GUARDED_ACTIONS,
)


@dataclass(frozen=True)
class ToolOutcome:
    """Classification of a tool execution result (from env.step observation)."""
    is_mutating: bool
    execution_succeeded: bool
    state_change_confirmed: bool
    policy_blocked: bool = False  # set when observation is synthetic rejection, not from env


def get_mutating_tools(domain: str) -> Set[str]:
    """Return the set of mutating tool names for the domain (or union if unknown)."""
    if domain == "airline":
        return set(AIRLINE_GUARDED_ACTIONS)
    if domain == "retail":
        return set(RETAIL_GUARDED_ACTIONS)
    return set(AIRLINE_GUARDED_ACTIONS) | set(RETAIL_GUARDED_ACTIONS)


def is_mutating_tool(domain: str, tool_name: str) -> bool:
    """True if tool_name is a state-changing tool in this domain."""
    return tool_name in get_mutating_tools(domain)


def classify_observation(
    domain: str,
    tool_name: str,
    observation: str,
) -> ToolOutcome:
    """
    Classify a tool result. Only call with observations from env.step (real execution).
    execution_succeeded = not Error; state_change_confirmed = same for mutating tools in MVP.
    """
    mutating = is_mutating_tool(domain, tool_name)
    obs = (observation or "").strip()
    success = not obs.startswith("Error:")
    # For mutating tools, treat non-error as state change confirmed; extend later with JSON parsing if needed
    state_confirmed = mutating and success
    return ToolOutcome(
        is_mutating=mutating,
        execution_succeeded=success,
        state_change_confirmed=state_confirmed,
        policy_blocked=False,
    )


# Lightweight heuristic: does respond content look like a final success/confirmation?
_SUCCESS_PHRASES = (
    "confirmed",
    "confirmation",
    "completed",
    "completion",
    "done",
    "successfully",
    "success",
    "finished",
    "booked",
    "reservation is confirmed",
    "order is confirmed",
    "your booking",
    "your order",
)


def is_success_style_respond(content: str) -> bool:
    """
    True if the respond content looks like a final success/confirmation message.
    Used only to trigger the completion gate when requires_grounded_completion is true.
    """
    if not content or not isinstance(content, str):
        return False
    text = content.strip().lower()
    if len(text) < 10:
        return False
    return any(phrase in text for phrase in _SUCCESS_PHRASES)

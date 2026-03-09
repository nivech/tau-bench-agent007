import hashlib
import json
from typing import Any, Dict

from tau_bench.orchestration.task_state import TaskState
from tau_bench.orchestration.tool_outcomes import get_mutating_tools


def is_mutating_action(domain: str, action_name: str) -> bool:
    """Return True if the action is mutating for the given domain."""
    return action_name in get_mutating_tools(domain)


def confirmation_required_for(domain: str, action_name: str) -> bool:
    """
    Return True if the action requires explicit user confirmation.

    Current behavior:
    - Airline: book_reservation
    - Retail: all mutating actions (policy/benchmarks already require confirmation for DB writes).
    """
    if domain == "airline":
        return action_name == "book_reservation"
    # For non-airline domains, err on the side of requiring confirmation for mutating tools.
    return action_name in get_mutating_tools(domain)


def summarize_mutating_action(
    action_name: str,
    built_args: Dict[str, Any],
    task_state: TaskState,
) -> str:
    """
    Build a canonical, deterministic summary string for a mutating action from grounded state.

    The summary is used both for user-facing confirmation text (after additional natural
    language wrapping) and as the input to fingerprinting. It must be stable under
    non-semantic changes (e.g. dict ordering).
    """
    domain = task_state.domain
    payload = {
        "domain": domain,
        "action": action_name,
        "args": built_args,
        "grounded_user_id": task_state.grounded.get("user_id") or task_state.identity.user_id,
        "grounded_reservation_ids": task_state.grounded.get("reservation_ids") or [],
        "grounded_order_ids": task_state.grounded.get("order_ids") or [],
    }
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"mutating_action_summary:{normalized}"


def fingerprint_action_summary(summary: str) -> str:
    """Compute a stable fingerprint for a canonical action summary."""
    return hashlib.sha256(summary.encode("utf-8")).hexdigest()


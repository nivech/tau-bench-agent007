import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from tau_bench.orchestration.pending_intent import PendingIntent, TrustLevel
from tau_bench.orchestration.tool_outcomes import get_mutating_tools
from tau_bench.orchestration.task_state import TaskState


class ActionBuildError(Exception):
    """Raised when executable kwargs for a mutating action cannot be safely built from TaskState."""


@dataclass
class BuiltActionArgs:
    """Executable kwargs and optional debug metadata."""

    kwargs: Dict[str, Any]


class ActionArgumentBuilder:
    """
    Generic builder that constructs executable kwargs for mutating actions from TaskState.

    Design:
    - Trusts only grounded TaskState and confirmed user choices for critical identifiers.
    - Treats any kwargs originating from the proposer as PROPOSER_DRAFT_UNTRUSTED hints.
    - Never invents placeholder IDs; fails closed when grounded data is missing.
    - Keeps orchestration-core generic; only relies on common identifier keys like user_id,
      reservation_id, and order_id that already exist in TaskState.grounded schema.
    """

    def build_action_args(
        self,
        action_name: str,
        task_state: TaskState,
        pending_intent: Optional[PendingIntent],
    ) -> BuiltActionArgs:
        domain = task_state.domain
        mutating_tools = get_mutating_tools(domain)
        is_mutating = action_name in mutating_tools

        # For non-mutating tools we do not change the current behavior; the proposer
        # supplies kwargs and validator/policy enforce correctness.
        if not is_mutating:
            # For now, builder is only used for mutating actions.
            raise ActionBuildError(f"ActionArgumentBuilder is only used for mutating tools, got {action_name}")

        # Start from untrusted proposed kwargs as hints (if present).
        hints: Dict[str, Any] = {}
        if pending_intent is not None:
            proposed = pending_intent.metadata.get("proposed_kwargs")
            if isinstance(proposed, dict):
                value = proposed.get("value")
                trust = proposed.get("trust_level")
                if isinstance(value, dict) and trust == TrustLevel.PROPOSER_DRAFT_UNTRUSTED.value:
                    hints = dict(value)

        kwargs: Dict[str, Any] = {}

        # 1) user_id: always take from grounded TaskState / identity when available.
        grounded_user_id = task_state.grounded.get("user_id") or task_state.identity.user_id
        if grounded_user_id:
            kwargs["user_id"] = grounded_user_id
        elif "user_id" in hints:
            # Proposer suggested a user_id but we have no grounded user_id; treat as unsafe.
            raise ActionBuildError("Missing grounded user_id for mutating action")

        # 2) reservation_id / order_id for subject-based tools.
        if "reservation_id" in hints or action_name.endswith("_reservation") or "reservation" in action_name:
            reservation_ids = task_state.grounded.get("reservation_ids") or []
            if "reservation_id" in hints and hints["reservation_id"] in reservation_ids:
                kwargs["reservation_id"] = hints["reservation_id"]
            elif len(reservation_ids) == 1:
                kwargs["reservation_id"] = reservation_ids[0]
            elif "reservation_id" in hints and reservation_ids and hints["reservation_id"] not in reservation_ids:
                raise ActionBuildError("Proposed reservation_id is not grounded")
            # If no grounded reservation_ids and none in hints, leave unset; validator/policy may still block.

        if "order_id" in hints or "order" in action_name:
            order_ids = task_state.grounded.get("order_ids") or []
            if "order_id" in hints and hints["order_id"] in order_ids:
                kwargs["order_id"] = hints["order_id"]
            elif len(order_ids) == 1:
                kwargs["order_id"] = order_ids[0]
            elif "order_id" in hints and order_ids and hints["order_id"] not in order_ids:
                raise ActionBuildError("Proposed order_id is not grounded")

        # 3) For all other fields, we conservatively copy hints through. These may represent
        # user-specified preferences (dates, locations, seat choices) that are not re-grounded
        # by tools. The critical invariant is that identifiers and authentication come from
        # grounded TaskState, not from untrusted drafts.
        for key, value in hints.items():
            if key in ("user_id", "reservation_id", "order_id"):
                # Already handled via grounded pathways above.
                continue
            kwargs.setdefault(key, value)

        # Fail closed when we cannot construct any kwargs at all.
        if not kwargs:
            raise ActionBuildError(
                f"Could not build grounded kwargs for mutating action {action_name}: "
                f"hints={json.dumps(hints)} grounded={json.dumps(task_state.grounded)}"
            )

        return BuiltActionArgs(kwargs=kwargs)


_DEFAULT_BUILDER = ActionArgumentBuilder()


def get_default_builder() -> ActionArgumentBuilder:
    """Return the process-wide default ActionArgumentBuilder."""
    return _DEFAULT_BUILDER


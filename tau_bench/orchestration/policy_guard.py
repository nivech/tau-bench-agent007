# Copyright Sierra
# Policy Guard v1: blocks policy-invalid or premature mutating actions using TaskState.
# Placed between validator and executor: validator = action validity; policy guard = timing/precondition legality.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tau_bench.envs.base import Env
from tau_bench.orchestration.task_state import TaskState
from tau_bench.types import Action, RESPOND_ACTION_NAME


# Block codes (machine-usable for recovery)
CODE_ALLOWED = "allowed"
CODE_MISSING_USER_ID = "missing_user_id"
CODE_NOT_AUTHENTICATED = "not_authenticated"
CODE_MISSING_PROFILE_GROUNDING = "missing_profile_grounding"
CODE_MISSING_CONFIRMATION = "missing_confirmation"
CODE_MISSING_RESERVATION_CONTEXT = "missing_reservation_context"
CODE_MISSING_ORDER_CONTEXT = "missing_order_context"
CODE_SUBJECT_AMBIGUITY = "subject_ambiguity"


# Airline mutating tools that require user_id (and for book: profile_grounded + confirmation)
AIRLINE_GUARDED_ACTIONS = frozenset({
    "book_reservation",
    "cancel_reservation",
    "update_reservation_flights",
    "update_reservation_baggages",
    "update_reservation_passengers",
})

# Retail mutating tools that require authenticated
RETAIL_GUARDED_ACTIONS = frozenset({
    "cancel_pending_order",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
})

# Confirmation key required for airline booking
BOOKING_CONFIRMATION_KEY = "booking_confirmed"

# Subject-entity params: for mutating tools, entity id must be in grounded known set when we have one.
# Tools that take the subject-entity param (subset of guarded actions)
_AIRLINE_SUBJECT_TOOLS = frozenset({"cancel_reservation", "update_reservation_flights", "update_reservation_baggages", "update_reservation_passengers"})
_RETAIL_SUBJECT_TOOLS = frozenset({
    "cancel_pending_order", "modify_pending_order_address", "modify_pending_order_items",
    "modify_pending_order_payment", "return_delivered_order_items", "exchange_delivered_order_items",
})


@dataclass(frozen=True)
class PolicyGuardResult:
    """Structured policy guard output for logging and recovery."""
    allowed: bool
    code: str
    message: str
    missing_prerequisites: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "code": self.code,
            "message": self.message,
            "missing_prerequisites": list(self.missing_prerequisites),
        }


def _allow(msg: str = "ok") -> PolicyGuardResult:
    return PolicyGuardResult(allowed=True, code=CODE_ALLOWED, message=msg, missing_prerequisites=[])


def _block(code: str, message: str, missing: List[str]) -> PolicyGuardResult:
    return PolicyGuardResult(allowed=False, code=code, message=message, missing_prerequisites=missing)


def _check_subject_resolution(domain: str, action: Action, task_state: TaskState) -> Optional[PolicyGuardResult]:
    """
    For mutating tools that take a subject-entity param (e.g. reservation_id, order_id),
    require that the value is in grounded known set when we have one. Generic for airline + retail.
    """
    if domain == "airline" and action.name in _AIRLINE_SUBJECT_TOOLS:
        param = "reservation_id"
        known_ids = task_state.grounded.get("reservation_ids") or []
        if not known_ids:
            return None  # no grounded context yet, allow and let tool/validator handle
        val = (action.kwargs or {}).get(param)
        if val is not None and isinstance(val, str) and val.strip() and val not in known_ids:
            return _block(
                CODE_SUBJECT_AMBIGUITY,
                "Resolve the target entity (account owner, saved entity, or newly introduced entity) before proceeding.",
                [param],
            )
    if domain == "retail" and action.name in _RETAIL_SUBJECT_TOOLS:
        param = "order_id"
        known_ids = task_state.grounded.get("order_ids") or []
        if not known_ids:
            return None
        val = (action.kwargs or {}).get(param)
        if val is not None and isinstance(val, str) and val.strip() and val not in known_ids:
            return _block(
                CODE_SUBJECT_AMBIGUITY,
                "Resolve the target entity (account owner, saved entity, or newly introduced entity) before proceeding.",
                [param],
            )
    return None


def check_policy(env: Env, action: Action, task_state: TaskState) -> PolicyGuardResult:
    """
    Check whether the action is allowed given current policy-relevant state.
    Validator already ensures tool existence and schema; this checks preconditions/timing.
    """
    # Respond: no policy check
    if action.name == RESPOND_ACTION_NAME:
        return _allow("respond allowed")

    domain = task_state.domain

    # Airline guarded actions
    if domain == "airline" and action.name in AIRLINE_GUARDED_ACTIONS:
        if not task_state.identity.user_id or not task_state.identity.user_id.strip():
            return _block(
                CODE_MISSING_USER_ID,
                "user_id must be established (e.g. via get_user_details) before airline mutating actions",
                ["user_id"],
            )
        if action.name == "book_reservation":
            if not task_state.identity.profile_grounded:
                return _block(
                    CODE_MISSING_PROFILE_GROUNDING,
                    "profile must be grounded (get_user_details) before booking to use payment methods",
                    ["profile_grounded"],
                )
            if BOOKING_CONFIRMATION_KEY not in task_state.confirmations:
                return _block(
                    CODE_MISSING_CONFIRMATION,
                    "explicit user confirmation required before booking",
                    ["booking_confirmed"],
                )
        subject_result = _check_subject_resolution(domain, action, task_state)
        if subject_result is not None:
            return subject_result
        return _allow()

    # Retail guarded actions
    if domain == "retail" and action.name in RETAIL_GUARDED_ACTIONS:
        if not task_state.identity.authenticated:
            return _block(
                CODE_NOT_AUTHENTICATED,
                "user must be authenticated (e.g. find_user_id_by_email or find_user_id_by_name_zip) before mutating actions",
                ["authenticated"],
            )
        subject_result = _check_subject_resolution(domain, action, task_state)
        if subject_result is not None:
            return subject_result
        return _allow()

    # Not a guarded action (read-only / lookup / unknown): allow; validator handles unknown tools
    return _allow()

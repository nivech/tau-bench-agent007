from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from tau_bench.types import Action


class TrustLevel(str, Enum):
    """
    Trust level for pieces of state used to build executable mutation arguments.

    The builder must treat only GROUNDED_FROM_TOOL and CONFIRMED_BY_USER as safe
    sources for composing final kwargs. INFERRED_SOFT and PROPOSER_DRAFT_UNTRUSTED
    are hints only and must not be treated as authoritative without promotion.
    """

    GROUNDED_FROM_TOOL = "grounded_from_tool"
    CONFIRMED_BY_USER = "confirmed_by_user"
    INFERRED_SOFT = "inferred_soft"
    PROPOSER_DRAFT_UNTRUSTED = "proposer_draft_untrusted"


class PendingIntentStatus(str, Enum):
    """Lifecycle state for a pending mutating intent."""

    NEEDS_PREREQUISITES = "needs_prerequisites"
    NEEDS_CONFIRMATION = "needs_confirmation"
    READY_TO_EXECUTE = "ready_to_execute"


@dataclass
class PendingIntent:
    """
    Generic representation of a pending mutating action.

    This lives at the orchestration layer and is the durable representation
    of the goal. It MUST NOT be treated as an executable Action payload.
    """

    action_name: str
    status: PendingIntentStatus
    unresolved_requirements: List[str] = field(default_factory=list)
    user_confirmed: bool = False
    confirmation_fingerprint: Optional[str] = None
    # Untrusted hints: proposer suggestions, soft user preferences, notes.
    # The ActionArgumentBuilder may consult these but must not treat them
    # as final executable kwargs without promotion.
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Optional bookkeeping for observability and retry policies.
    created_step: Optional[int] = None
    last_updated_step: Optional[int] = None
    retry_count: int = 0


def pending_intent_from_policy_block(
    action: Action,
    missing_prereqs: List[str],
    *,
    created_step: Optional[int] = None,
    existing: Optional[PendingIntent] = None,
    metadata_hint: Optional[Dict[str, Any]] = None,
) -> PendingIntent:
    """
    Create or update a PendingIntent from a policy block on a mutating action.

    - action.kwargs are treated as untrusted draft hints only.
    - missing_prereqs come from PolicyGuardResult.missing_prerequisites.
    """

    status = PendingIntentStatus.NEEDS_PREREQUISITES
    if "booking_confirmed" in missing_prereqs:
        # Generic confirmation requirement; specific key is handled by policy/recovery.
        status = PendingIntentStatus.NEEDS_CONFIRMATION

    base = existing
    if base is None:
        base = PendingIntent(
            action_name=action.name,
            status=status,
            unresolved_requirements=list(missing_prereqs),
            created_step=created_step,
            last_updated_step=created_step,
            metadata={},
        )
    else:
        base.status = status
        # Merge unresolved requirements; keep unique set but preserve deterministic order.
        merged = list(dict.fromkeys(list(base.unresolved_requirements) + list(missing_prereqs)))
        base.unresolved_requirements = merged
        base.last_updated_step = created_step

    # Attach untrusted hints about the original proposal under a dedicated key.
    # Builders may decide to consult these but must treat them as
    # TrustLevel.PROPOSER_DRAFT_UNTRUSTED.
    if "proposed_kwargs" not in base.metadata:
        base.metadata["proposed_kwargs"] = {
            "value": action.kwargs or {},
            "trust_level": TrustLevel.PROPOSER_DRAFT_UNTRUSTED.value,
        }
    if metadata_hint:
        base.metadata.setdefault("hints", {}).update(metadata_hint)

    return base


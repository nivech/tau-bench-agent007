# Copyright Sierra
# Recovery module for Phase 3 orchestration: typed failure categories, recovery decisions,
# and rule-based recovery logic. Integrates with run_loop on validator/policy/tool failure.

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from tau_bench.types import Action

# Import policy guard constants for side-effecting tools and confirmation key
from tau_bench.orchestration.policy_guard import (
    AIRLINE_GUARDED_ACTIONS,
    CODE_MISSING_CONFIRMATION,
    CODE_MISSING_ORDER_CONTEXT,
    CODE_MISSING_PROFILE_GROUNDING,
    CODE_MISSING_RESERVATION_CONTEXT,
    CODE_MISSING_USER_ID,
    CODE_NOT_AUTHENTICATED,
    CODE_SUBJECT_AMBIGUITY,
    RETAIL_GUARDED_ACTIONS,
)


class FailureCategory(str, Enum):
    """Typed failure categories for recovery input."""
    validation_error = "validation_error"
    policy_block = "policy_block"
    missing_required_user_confirmation = "missing_required_user_confirmation"
    policy_rewrite_needed = "policy_rewrite_needed"
    tool_execution_error = "tool_execution_error"
    no_progress = "no_progress"
    repeated_same_action = "repeated_same_action"
    missing_required_grounding = "missing_required_grounding"
    unresolved_slot_or_constraint = "unresolved_slot_or_constraint"
    memory_inconsistency_or_missing_fact = "memory_inconsistency_or_missing_fact"
    repeated_failure_after_repair = "repeated_failure_after_repair"
    budget_risk_or_turn_limit_risk = "budget_risk_or_turn_limit_risk"
    completion_guard_blocked = "completion_guard_blocked"
    subject_ambiguity = "subject_ambiguity"
    proposer_error = "proposer_error"


class RecoveryStrategy(str, Enum):
    """Explicit recovery strategies the orchestrator can apply."""
    RETRY_SAME_ACTION = "RETRY_SAME_ACTION"
    RETRY_REPAIRED_ACTION = "RETRY_REPAIRED_ACTION"
    ASK_USER_CONFIRMATION = "ASK_USER_CONFIRMATION"
    SATISFY_PREREQUISITE = "SATISFY_PREREQUISITE"
    ASK_CLARIFYING_QUESTION = "ASK_CLARIFYING_QUESTION"
    REPLAN_FROM_STATE = "REPLAN_FROM_STATE"
    SWITCH_TOOL_OR_ACTION_TYPE = "SWITCH_TOOL_OR_ACTION_TYPE"
    SUMMARIZE_AND_CONFIRM_PENDING_SIDE_EFFECT = "SUMMARIZE_AND_CONFIRM_PENDING_SIDE_EFFECT"
    SAFE_TERMINATE = "SAFE_TERMINATE"
    ESCALATION_CANDIDATE = "ESCALATION_CANDIDATE"
    DEFER_UNTIL_MEMORY_UPDATE = "DEFER_UNTIL_MEMORY_UPDATE"


@dataclass
class RecoveryState:
    """Mutable recovery state carried alongside TaskState in the run loop."""
    retry_counts: Dict[str, int] = field(default_factory=dict)
    failure_type_counts: Dict[str, int] = field(default_factory=dict)
    pending_side_effect_action: Optional[Action] = None
    pending_confirmation_key: Optional[str] = None
    pending_since_step: int = 0
    pending_action_id: Optional[str] = None
    recovery_count_this_run: int = 0
    # Last blocked/failed retry_key for repeated_same_action detection
    last_blocked_retry_key: Optional[str] = None
    # Last N last_action values for no-progress detection (Phase C)
    recent_last_actions: List[str] = field(default_factory=list)
    # True when recovery is in a flow that requires user interaction (confirmation, clarification,
    # missing slot, ambiguity resolution, etc.). Used so completion guard allows respond through
    # to reach env and get a user turn; extend when adding ASK_CLARIFYING_QUESTION, slot prompts, etc.
    awaiting_user_input: bool = False
    # When set, the orchestrator replays this action on the same step (instead of calling the proposer).
    # Current semantics: a blocked side-effect action becomes eligible again after user confirmation;
    # we store it here so the run loop can re-execute it deterministically.
    #
    # Future generalization: other recovery types may need different retry semantics (e.g. clarification
    # may require patching arguments; entity disambiguation may require rewriting action inputs; tool
    # failure may require an alternate action, not the same one). Consider a more general abstraction
    # later, e.g. resume_action / recovery_resume_action / post_recovery_next_action, with optional
    # "retry with update" when arguments could be stale, state changed, or user input should modify
    # the action before retry. For confirmation-only flows, replaying the exact stored action is correct.
    retry_action_after_confirmation: Optional[Action] = None
    # Prerequisite-targeted recovery: when a mutating action is blocked for missing prerequisite(s),
    # we store the blocked intent so planner can steer toward satisfying the prerequisite, then resume.
    blocked_goal_action: Optional[Action] = None
    missing_prerequisites: List[str] = field(default_factory=list)
    resume_intent_after_prereq: bool = False
    # When all prerequisites are satisfied, run_loop can set this to replay the blocked action (similar to confirmation).
    retry_action_after_prereq: Optional[Action] = None


def _default_side_effecting_tools() -> Set[str]:
    return set(AIRLINE_GUARDED_ACTIONS) | set(RETAIL_GUARDED_ACTIONS)


# Policy block codes that indicate a missing prerequisite (not confirmation). Recovery records blocked intent
# and steers next step toward satisfying the prerequisite before retrying the mutating action.
PREREQUISITE_BLOCK_CODES = frozenset({
    CODE_MISSING_USER_ID,
    CODE_MISSING_PROFILE_GROUNDING,
    CODE_NOT_AUTHENTICATED,
    CODE_MISSING_RESERVATION_CONTEXT,
    CODE_MISSING_ORDER_CONTEXT,
})


def _prereq_code_to_required_state(code: str, missing: List[str]) -> Dict[str, Any]:
    """Map policy block code to a minimal required-state description for planner/run_loop."""
    if code == CODE_MISSING_USER_ID:
        return {"user_id": "established"}
    if code == CODE_MISSING_PROFILE_GROUNDING:
        return {"profile_grounded": True}
    if code == CODE_NOT_AUTHENTICATED:
        return {"authenticated": True}
    if code == CODE_MISSING_RESERVATION_CONTEXT:
        return {"reservation_context": "grounded"}
    if code == CODE_MISSING_ORDER_CONTEXT:
        return {"order_context": "grounded"}
    return {"missing_prerequisites": missing}


@dataclass(frozen=True)
class RecoveryConfig:
    """Configuration for recovery budgets and side-effecting tools."""
    max_recovery_per_run: int = 10
    max_retries_per_action: int = 2
    side_effecting_tools: Set[str] = field(default_factory=_default_side_effecting_tools)


@dataclass
class RecoveryInput:
    """Typed input to decide_recovery."""
    failure_type: str  # FailureCategory value
    action: Action
    step_index: int
    max_num_steps: int
    # Source result: ValidatorResult or PolicyGuardResult or tool observation summary
    source_code: Optional[str] = None
    source_message: Optional[str] = None
    missing_prerequisites: Optional[List[str]] = None
    tool_observation_summary: Optional[str] = None
    # State refs (recovery reads only; orchestrator applies updates)
    domain: Optional[str] = None
    recovery_state: Optional[RecoveryState] = None
    last_action: Optional[str] = None
    # Future: planner_hint, memory_status
    planner_hint: Optional[str] = None
    memory_status: Optional[str] = None


@dataclass(frozen=True)
class RecoveryDecision:
    """Typed recovery output for orchestrator."""
    failure_type: str
    diagnosis: str
    confidence: float
    recoverable: bool
    proposed_strategy: str  # RecoveryStrategy value
    message_to_user: Optional[str] = None
    repaired_action: Optional[Action] = None
    replanning_hint: Optional[str] = None
    state_updates: Dict[str, Any] = field(default_factory=dict)
    retry_allowed: bool = False
    retry_key: Optional[str] = None
    retry_budget_cost: int = 1
    terminal_reason: Optional[str] = None
    trace_metadata: Dict[str, Any] = field(default_factory=dict)


def action_retry_key(action: Action) -> str:
    """Deterministic fingerprint for action to count retries."""
    kwargs = action.kwargs or {}
    return f"{action.name}|{json.dumps(sorted(kwargs.items()), sort_keys=True)}"


NO_PROGRESS_WINDOW = 3


def is_no_progress(recovery_state: RecoveryState, window: int = NO_PROGRESS_WINDOW) -> bool:
    """True if the last `window` steps had the same last_action (and it is not None/empty)."""
    recent = recovery_state.recent_last_actions
    if len(recent) < window:
        return False
    last_n = recent[-window:]
    if not last_n[0] or last_n[0].strip() == "":
        return False
    return all(x == last_n[0] for x in last_n)


def default_recovery_config(domain: str = "airline") -> RecoveryConfig:
    """Build RecoveryConfig with side-effecting tools for domain."""
    if domain == "airline":
        tools = set(AIRLINE_GUARDED_ACTIONS)
    elif domain == "retail":
        tools = set(RETAIL_GUARDED_ACTIONS)
    else:
        tools = set(AIRLINE_GUARDED_ACTIONS) | set(RETAIL_GUARDED_ACTIONS)
    return RecoveryConfig(side_effecting_tools=tools)


def decide_recovery(input_: RecoveryInput, config: RecoveryConfig) -> RecoveryDecision:
    """
    Decide recovery strategy from failure input. Phase A: stub returns REPLAN_FROM_STATE.
    Phase B+ will implement ASK_USER_CONFIRMATION, RETRY_REPAIRED_ACTION, etc.
    """
    failure_type = input_.failure_type
    action = input_.action
    step_index = input_.step_index
    recovery_state = input_.recovery_state or RecoveryState()
    retry_key = action_retry_key(action)

    # Trace metadata for logging
    trace_metadata: Dict[str, Any] = {
        "failure_code": input_.source_code,
        "missing_prerequisites": input_.missing_prerequisites or [],
        "recovery_count_this_run": recovery_state.recovery_count_this_run,
        "retry_key": retry_key,
    }

    # Budget / turn limit risk
    if recovery_state.recovery_count_this_run >= config.max_recovery_per_run:
        return RecoveryDecision(
            failure_type=failure_type,
            diagnosis="Recovery budget exhausted",
            confidence=1.0,
            recoverable=False,
            proposed_strategy=RecoveryStrategy.SAFE_TERMINATE.value,
            retry_key=retry_key,
            terminal_reason="max_recovery_per_run",
            trace_metadata={**trace_metadata, "reason": "budget_exhausted"},
        )
    if step_index >= input_.max_num_steps - 1:
        return RecoveryDecision(
            failure_type=failure_type,
            diagnosis="Turn limit near; safe terminate",
            confidence=1.0,
            recoverable=False,
            proposed_strategy=RecoveryStrategy.SAFE_TERMINATE.value,
            retry_key=retry_key,
            terminal_reason="turn_limit_risk",
            trace_metadata=trace_metadata,
        )

    # Phase B: policy_block -> ASK_USER_CONFIRMATION when missing_confirmation + side-effecting; else REPLAN
    if failure_type in (
        FailureCategory.policy_block.value,
        FailureCategory.missing_required_user_confirmation.value,
    ):
        diagnosis = f"Policy blocked: {input_.source_message or input_.source_code or 'unknown'}"
        # Subject ambiguity: resolve target entity before mutating
        if input_.source_code == CODE_SUBJECT_AMBIGUITY:
            return RecoveryDecision(
                failure_type=FailureCategory.subject_ambiguity.value,
                diagnosis="Target entity for the action is ambiguous (account owner, saved entity, or new entity).",
                confidence=0.9,
                recoverable=True,
                proposed_strategy=RecoveryStrategy.REPLAN_FROM_STATE.value,
                message_to_user="Resolve the target entity before taking this action (account owner, saved entity, or newly introduced entity).",
                replanning_hint="Resolve the action target before retrying; do not assume which entity the action applies to.",
                retry_key=retry_key,
                retry_budget_cost=1,
                trace_metadata={**trace_metadata, "subject_ambiguity": True},
            )
        # Same retry_key blocked again while we already have a pending action -> repeated_same_action
        if recovery_state.pending_side_effect_action is not None:
            pending_key = action_retry_key(recovery_state.pending_side_effect_action)
            if retry_key == pending_key:
                return RecoveryDecision(
                    failure_type=FailureCategory.repeated_same_action.value,
                    diagnosis="Same side-effecting action blocked again without confirmation; replan.",
                    confidence=1.0,
                    recoverable=True,
                    proposed_strategy=RecoveryStrategy.REPLAN_FROM_STATE.value,
                    replanning_hint="Ask for explicit user confirmation before retrying this action.",
                    retry_key=retry_key,
                    retry_budget_cost=1,
                    trace_metadata=trace_metadata,
                )
        # missing_confirmation + side-effecting -> ASK_USER_CONFIRMATION (first time for this action)
        if (
            input_.source_code == CODE_MISSING_CONFIRMATION
            and action.name in config.side_effecting_tools
        ):
            confirmation_key = (
                input_.missing_prerequisites[0]
                if input_.missing_prerequisites
                else "booking_confirmed"
            )
            return RecoveryDecision(
                failure_type=failure_type,
                diagnosis=diagnosis,
                confidence=0.9,
                recoverable=True,
                proposed_strategy=RecoveryStrategy.ASK_USER_CONFIRMATION.value,
                message_to_user="Please confirm you want to proceed with this action.",
                state_updates={
                    "set_pending_side_effect_action": action,
                    "pending_confirmation_key": confirmation_key,
                    "pending_since_step": step_index,
                },
                retry_allowed=True,
                retry_key=retry_key,
                retry_budget_cost=1,
                trace_metadata={**trace_metadata, "confirmation_key": confirmation_key},
            )
        # Prerequisite block (missing_user_id, missing_profile_grounding, not_authenticated, etc.)
        # -> SATISFY_PREREQUISITE: record blocked intent so planner steers toward satisfying prerequisite.
        if (
            input_.source_code in PREREQUISITE_BLOCK_CODES
            and action.name in config.side_effecting_tools
        ):
            missing = input_.missing_prerequisites or []
            return RecoveryDecision(
                failure_type=failure_type,
                diagnosis=diagnosis,
                confidence=0.9,
                recoverable=True,
                proposed_strategy=RecoveryStrategy.SATISFY_PREREQUISITE.value,
                message_to_user=(
                    "You must first establish the missing prerequisite before retrying the blocked action. "
                    + (input_.source_message or "")
                ),
                state_updates={
                    "blocked_goal_action": action,
                    "missing_prerequisites": list(missing),
                    "resume_intent_after_prereq": True,
                    "next_required_state": _prereq_code_to_required_state(input_.source_code, missing),
                },
                replanning_hint=(
                    "Next step: satisfy the missing prerequisite(s) (" + ", ".join(missing) + "), "
                    "then retry the blocked mutating action. Do not repeat the same blocked action until the prerequisite is satisfied."
                ),
                retry_key=retry_key,
                retry_budget_cost=1,
                trace_metadata={**trace_metadata, "prerequisite_block": True, "missing_prereqs": missing},
            )
        # Other policy blocks -> REPLAN with recovery guidance
        return RecoveryDecision(
            failure_type=failure_type,
            diagnosis=diagnosis,
            confidence=0.8,
            recoverable=True,
            proposed_strategy=RecoveryStrategy.REPLAN_FROM_STATE.value,
            message_to_user="If a required state-changing action has not yet succeeded, do not claim completion. Recover from the last blocked or failed step. Resolve the target entity before retrying if ambiguous.",
            replanning_hint="Reconsider last error; satisfy missing prerequisite or resolve target entity, then retry.",
            retry_key=retry_key,
            retry_budget_cost=1,
            trace_metadata=trace_metadata,
        )

    # validation_error -> REPLAN (optional: RETRY_REPAIRED_ACTION for trivial repairs later)
    if failure_type == FailureCategory.validation_error.value:
        diagnosis = f"Validation failed: {input_.source_message or input_.source_code or 'unknown'}"
        return RecoveryDecision(
            failure_type=failure_type,
            diagnosis=diagnosis,
            confidence=0.8,
            recoverable=True,
            proposed_strategy=RecoveryStrategy.REPLAN_FROM_STATE.value,
            replanning_hint="Reconsider last error and try a different approach.",
            retry_key=retry_key,
            retry_budget_cost=1,
            trace_metadata=trace_metadata,
        )

    # Phase C: no_progress -> REPLAN
    if failure_type == FailureCategory.no_progress.value:
        diagnosis = "No progress: same action repeated without success."
        return RecoveryDecision(
            failure_type=failure_type,
            diagnosis=diagnosis,
            confidence=0.9,
            recoverable=True,
            proposed_strategy=RecoveryStrategy.REPLAN_FROM_STATE.value,
            message_to_user="Base progress on actual tool outcomes. Confirm completion only after successful execution is observed.",
            replanning_hint="Try a different action or approach; no progress detected.",
            retry_key=retry_key,
            retry_budget_cost=1,
            trace_metadata=trace_metadata,
        )

    # Phase D: tool_execution_error -> REPLAN with grounded guidance
    if failure_type == FailureCategory.tool_execution_error.value:
        diagnosis = f"Tool execution failed: {input_.tool_observation_summary or input_.source_message or 'unknown'}"
        return RecoveryDecision(
            failure_type=failure_type,
            diagnosis=diagnosis,
            confidence=0.8,
            recoverable=True,
            proposed_strategy=RecoveryStrategy.REPLAN_FROM_STATE.value,
            message_to_user="Use actual tool outcomes to assess progress. A blocked or failed state-changing action means the task is still incomplete. Do not claim success until successful execution is observed.",
            replanning_hint="Try different arguments or approach; treat this as incomplete and recover.",
            retry_key=retry_key,
            retry_budget_cost=1,
            trace_metadata={**trace_metadata, "tool_observation_summary": input_.tool_observation_summary},
        )

    # Grounded completion guard: do not claim success without real tool execution
    if failure_type == FailureCategory.completion_guard_blocked.value:
        diagnosis = "Success-style response blocked: no grounded state-changing tool execution observed."
        return RecoveryDecision(
            failure_type=failure_type,
            diagnosis=diagnosis,
            confidence=1.0,
            recoverable=True,
            proposed_strategy=RecoveryStrategy.REPLAN_FROM_STATE.value,
            message_to_user=get_completion_guard_recovery_message(),
            replanning_hint="Do not finalize until the required state-changing action succeeds. Use actual tool outcomes, not conversational claims.",
            retry_key=retry_key,
            retry_budget_cost=1,
            trace_metadata=trace_metadata,
        )

    # Subject ambiguity: resolve target entity before mutating
    if failure_type == FailureCategory.subject_ambiguity.value:
        diagnosis = "Target entity for the action is ambiguous (account owner, saved entity, or new entity)."
        return RecoveryDecision(
            failure_type=failure_type,
            diagnosis=diagnosis,
            confidence=0.9,
            recoverable=True,
            proposed_strategy=RecoveryStrategy.REPLAN_FROM_STATE.value,
            message_to_user="Resolve the target entity before taking this action (account owner, saved entity, or newly introduced entity).",
            replanning_hint="Resolve the action target before retrying; do not assume which entity the action applies to.",
            retry_key=retry_key,
            retry_budget_cost=1,
            trace_metadata=trace_metadata,
        )

    # Proposer exception: model call or message_to_action failed; replan so next step can retry.
    if failure_type == FailureCategory.proposer_error.value:
        diagnosis = f"Proposer failed: {input_.source_message or 'unknown error'}"
        return RecoveryDecision(
            failure_type=failure_type,
            diagnosis=diagnosis,
            confidence=0.8,
            recoverable=True,
            proposed_strategy=RecoveryStrategy.REPLAN_FROM_STATE.value,
            message_to_user="The previous step could not produce a valid action. Continue from the latest state and try again.",
            replanning_hint="Reconsider and produce a valid tool call or response.",
            retry_key=retry_key,
            retry_budget_cost=1,
            trace_metadata=trace_metadata,
        )

    diagnosis = f"Failure type {failure_type}"
    return RecoveryDecision(
        failure_type=failure_type,
        diagnosis=diagnosis,
        confidence=0.8,
        recoverable=True,
        proposed_strategy=RecoveryStrategy.REPLAN_FROM_STATE.value,
        replanning_hint="Reconsider last error and try a different approach.",
        retry_key=retry_key,
        retry_budget_cost=1,
        trace_metadata=trace_metadata,
    )


def get_completion_guard_recovery_message() -> str:
    """Concise message injected when completion guard blocks a success-style respond. Generic for airline + retail."""
    return (
        "No successful state-changing tool execution has been observed yet. "
        "Do not confirm completion. Continue from the latest grounded state and complete the required action first."
    )


def detect_confirmation_satisfied(
    last_user_content: str,
    pending_confirmation_key: str,
) -> bool:
    """
    Heuristic: user message indicates confirmation for the pending key.
    Used for booking_confirmed and other confirmation keys.
    """
    if not last_user_content or not isinstance(last_user_content, str):
        return False
    text = last_user_content.strip().lower()
    # Strong negatives
    if any(x in text for x in ("no", "don't", "do not", "cancel", "never mind")):
        return False
    # Affirmatives
    if any(x in text for x in ("yes", "confirm", "go ahead", "please do", "proceed", "ok", "okay", "sure")):
        return True
    return False

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


class RecoveryStrategy(str, Enum):
    """Explicit recovery strategies the orchestrator can apply."""
    RETRY_SAME_ACTION = "RETRY_SAME_ACTION"
    RETRY_REPAIRED_ACTION = "RETRY_REPAIRED_ACTION"
    ASK_USER_CONFIRMATION = "ASK_USER_CONFIRMATION"
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


def _default_side_effecting_tools() -> Set[str]:
    return set(AIRLINE_GUARDED_ACTIONS) | set(RETAIL_GUARDED_ACTIONS)


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
                message_to_user="Resolve the target entity before mutating state if ambiguous (account owner, saved entity, or newly introduced entity).",
                replanning_hint="Resolve who or what the action applies to before proceeding.",
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
        # Other policy blocks -> REPLAN
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

    # Phase C: no_progress -> REPLAN or SAFE_TERMINATE
    if failure_type == FailureCategory.no_progress.value:
        diagnosis = "No progress: same action repeated without success."
        return RecoveryDecision(
            failure_type=failure_type,
            diagnosis=diagnosis,
            confidence=0.9,
            recoverable=True,
            proposed_strategy=RecoveryStrategy.REPLAN_FROM_STATE.value,
            replanning_hint="Try a different action or approach; no progress detected.",
            retry_key=retry_key,
            retry_budget_cost=1,
            trace_metadata=trace_metadata,
        )

    # Phase D: tool_execution_error -> ASK_CLARIFYING_QUESTION or REPLAN
    if failure_type == FailureCategory.tool_execution_error.value:
        diagnosis = f"Tool execution failed: {input_.tool_observation_summary or input_.source_message or 'unknown'}"
        return RecoveryDecision(
            failure_type=failure_type,
            diagnosis=diagnosis,
            confidence=0.8,
            recoverable=True,
            proposed_strategy=RecoveryStrategy.REPLAN_FROM_STATE.value,
            replanning_hint="Tool returned an error; try different arguments or another approach.",
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
            replanning_hint="Do not claim completion until the relevant action succeeds in the environment.",
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
            message_to_user="Resolve the target entity before mutating state if ambiguous (account owner, saved entity, or newly introduced entity).",
            replanning_hint="Resolve who or what the action applies to before proceeding.",
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
    """Generic message injected when completion guard blocks a success-style respond."""
    return (
        "A successful state-changing tool execution has not yet been observed. "
        "Do not claim completion until the relevant action succeeds in the environment."
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

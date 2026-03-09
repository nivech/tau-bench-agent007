# Copyright Sierra
# Planner / Decomposer for Phase 3 orchestration. Domain-agnostic short-horizon planning
# from TaskState and RecoveryState. Produces structured PlanResult for proposer guidance.

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from tau_bench.orchestration.recovery import RecoveryState, is_no_progress
from tau_bench.orchestration.task_state import TaskState


def _prereq_to_action_hint(prereq: str) -> str:
    """Short hint for planner: what kind of action satisfies this prerequisite (no tool names required)."""
    if prereq == "user_id":
        return "establish user identity (e.g. get user details or lookup)"
    if prereq == "profile_grounded":
        return "retrieve user profile so payment methods and context are grounded"
    if prereq == "authenticated":
        return "authenticate the user (e.g. find user by email or name/zip)"
    if prereq in ("reservation_id", "reservation_context"):
        return "resolve reservation context (e.g. get reservation details)"
    if prereq in ("order_id", "order_context"):
        return "resolve order context (e.g. get order details)"
    return f"satisfy {prereq}"


class StepType(str, Enum):
    """Type of step in the short-horizon plan."""
    tool_call = "tool_call"
    ask_user = "ask_user"
    respond = "respond"
    wait_terminate_safe = "wait_terminate_safe"


@dataclass(frozen=True)
class StepSpec:
    """One step in the next_steps list (1-3 items)."""
    step_type: str  # StepType value
    rationale: str
    prerequisite_satisfied: Optional[str] = None
    expected_info: Optional[str] = None


# Replan triggers and anti-patterns (generic strings for plan output)
DEFAULT_REPLAN_TRIGGERS = [
    "validation_failure",
    "policy_block",
    "missing_required_slot_or_entity",
    "repeated_same_failure",
    "tool_returned_empty_or_incompatible_result",
]

DEFAULT_ANTI_PATTERNS = [
    "repeating the same action after it failed or was blocked",
    "switching goals mid-task",
    "inventing arguments or IDs not from task state or tool results",
    "responding with success before executing required tool",
]


@dataclass
class PlanResult:
    """Structured output of the planner for proposer guidance."""
    current_goal: str
    active_constraints: List[str] = field(default_factory=list)
    relevant_known_facts: List[str] = field(default_factory=list)
    subgoal: str = ""
    next_steps: List[StepSpec] = field(default_factory=list)
    preferred_next_action_type: str = "tool_call"
    success_checkpoint: str = ""
    replan_triggers: List[str] = field(default_factory=list)
    anti_patterns_to_avoid: List[str] = field(default_factory=list)
    planning_notes: Optional[str] = None


def _current_goal(task_state: TaskState) -> str:
    """Canonical task goal from intent."""
    if task_state.intent.current_objective:
        return task_state.intent.current_objective
    if task_state.intent.initial_instruction:
        return task_state.intent.initial_instruction
    return "Complete the user's task per instructions."


def _active_constraints(task_state: TaskState) -> List[str]:
    """Hard constraints from unresolved_slots and required_prerequisites."""
    out: List[str] = []
    for s in task_state.intent.unresolved_slots or []:
        if s and s.strip():
            out.append(f"Resolve slot: {s.strip()}")
    for p in task_state.checklist.required_prerequisites or []:
        if p and p.strip():
            out.append(f"Prerequisite: {p.strip()}")
    return out


def _relevant_known_facts(task_state: TaskState) -> List[str]:
    """Compact grounded facts only; no domain-specific branching beyond keys in TaskState."""
    facts: List[str] = []
    uid = task_state.grounded.get("user_id")
    if uid:
        facts.append(f"user_id={uid}")
    else:
        facts.append("user_id=not established")
    if task_state.identity.profile_grounded:
        facts.append("profile_grounded=True")
    pm = task_state.grounded.get("known_payment_method_ids") or []
    if pm:
        facts.append(f"known_payment_method_ids={pm}")
    if task_state.domain == "airline":
        rids = task_state.grounded.get("reservation_ids") or []
        if rids:
            facts.append(f"reservation_ids={rids}")
    elif task_state.domain == "retail":
        oids = task_state.grounded.get("order_ids") or []
        if oids:
            facts.append(f"order_ids={oids}")
    if task_state.subject_resolution_status == "ambiguous":
        facts.append("subject_resolution_status=ambiguous; resolve target entity before mutating.")
    return facts


def _infer_subgoal(task_state: TaskState, recovery_state: Optional[RecoveryState]) -> str:
    """Infer immediate local objective from state."""
    if task_state.subject_resolution_status == "ambiguous":
        return "Resolve who or what the action applies to (account owner, saved entity, or new entity) before proceeding."
    # Blocked mutating action with missing prerequisites: steer next step toward satisfying them.
    if recovery_state and recovery_state.blocked_goal_action and recovery_state.missing_prerequisites:
        blocked_name = recovery_state.blocked_goal_action.name
        hints = [_prereq_to_action_hint(p) for p in recovery_state.missing_prerequisites]
        prereq_text = "; ".join(hints[:3])
        return (
            f"First satisfy the missing prerequisite(s): {prereq_text}. "
            f"Then retry the blocked action ({blocked_name}). Do not repeat {blocked_name} until the prerequisite is satisfied."
        )
    if recovery_state and is_no_progress(recovery_state):
        return "Make progress: avoid repeating the same action; try a different step or ask the user."
    if task_state.last_error:
        return "Recover from last error: satisfy missing prerequisite or clarify before retrying."
    if len(task_state.attempted_mutating_tools) > 0 and len(task_state.successful_mutations) == 0:
        return "Complete the required state-changing action (do not claim success until a tool succeeds)."
    if task_state.checklist.pending:
        return f"Address pending: {', '.join(task_state.checklist.pending[:3])}."
    if task_state.intent.unresolved_slots:
        return f"Resolve unresolved slots: {', '.join(task_state.intent.unresolved_slots[:3])}."
    goal = _current_goal(task_state)
    if len(goal) > 80:
        return goal[:77] + "..."
    return goal


def _next_steps(
    task_state: TaskState, preferred: str, recovery_state: Optional[RecoveryState] = None
) -> List[StepSpec]:
    """Build 1-3 next steps."""
    steps: List[StepSpec] = []
    if preferred == "ask_user":
        steps.append(StepSpec(
            step_type=StepType.ask_user.value,
            rationale="Clarify target entity or missing information before mutating.",
            prerequisite_satisfied="None",
            expected_info="User clarification.",
        ))
        return steps[:3]
    if preferred == "respond":
        steps.append(StepSpec(
            step_type=StepType.respond.value,
            rationale="Task is complete; respond to user.",
            prerequisite_satisfied="Required tools succeeded.",
            expected_info="User sees final response.",
        ))
        return steps[:3]
    # Blocked on missing prerequisite: first step is to satisfy it, then retry the blocked action.
    if recovery_state and recovery_state.blocked_goal_action and recovery_state.missing_prerequisites:
        prereq_hints = [_prereq_to_action_hint(p) for p in recovery_state.missing_prerequisites[:2]]
        steps.append(StepSpec(
            step_type=StepType.tool_call.value,
            rationale=f"Satisfy missing prerequisite(s): {'; '.join(prereq_hints)}. Do not retry the blocked mutating action until this is done.",
            prerequisite_satisfied="Prerequisite will be satisfied by this tool result.",
            expected_info="Grounded user/profile/context so the blocked action can be retried.",
        ))
        steps.append(StepSpec(
            step_type=StepType.tool_call.value,
            rationale=f"After prerequisite is satisfied, retry the blocked action ({recovery_state.blocked_goal_action.name}).",
            prerequisite_satisfied="Prerequisite satisfied from previous step.",
            expected_info="Mutating action succeeds.",
        ))
        return steps[:3]
    # Prefer tool_call: add one or two tool-oriented steps
    steps.append(StepSpec(
        step_type=StepType.tool_call.value,
        rationale="Execute next tool to make progress toward goal.",
        prerequisite_satisfied="Use only grounded IDs from task state.",
        expected_info="Tool result or error.",
    ))
    if task_state.intent.unresolved_slots or task_state.checklist.required_prerequisites:
        steps.append(StepSpec(
            step_type=StepType.tool_call.value,
            rationale="Obtain or confirm missing slot or prerequisite if needed.",
            prerequisite_satisfied="Grounded facts available.",
            expected_info="Slot value or confirmation.",
        ))
    return steps[:3]


def _preferred_next_action_type(task_state: TaskState, recovery_state: Optional[RecoveryState]) -> str:
    """Decide preferred next action type."""
    if task_state.subject_resolution_status == "ambiguous":
        return StepType.ask_user.value
    # When blocked on missing prerequisite, next step is a tool call to satisfy it (e.g. get_user_details).
    if recovery_state and recovery_state.blocked_goal_action and recovery_state.missing_prerequisites:
        return StepType.tool_call.value
    if recovery_state and recovery_state.last_blocked_retry_key:
        return StepType.ask_user.value  # Prefer clarify or different action over repeat
    if len(task_state.attempted_mutating_tools) > 0 and len(task_state.successful_mutations) == 0:
        return StepType.tool_call.value
    if recovery_state and is_no_progress(recovery_state):
        return StepType.tool_call.value  # Need to do something different (tool or ask)
    if task_state.successful_mutations and not task_state.intent.unresolved_slots and not task_state.checklist.pending:
        return StepType.respond.value
    return StepType.tool_call.value


def _success_checkpoint(
    task_state: TaskState, recovery_state: Optional[RecoveryState] = None
) -> str:
    """What observation would count as progress."""
    if recovery_state and recovery_state.blocked_goal_action and recovery_state.missing_prerequisites:
        return (
            "Missing prerequisite is satisfied (e.g. profile retrieved or user_id established); "
            "then the blocked mutating action can be retried and must execute successfully."
        )
    if len(task_state.attempted_mutating_tools) > 0 and len(task_state.successful_mutations) == 0:
        return "A state-changing tool executes successfully (non-error observation)."
    if task_state.subject_resolution_status == "ambiguous":
        return "Target entity is resolved (user clarification or grounded inference)."
    return "Next tool returns useful result or user provides needed information."


def _replan_triggers(task_state: TaskState, recovery_state: Optional[RecoveryState]) -> List[str]:
    """When to replan."""
    triggers = list(DEFAULT_REPLAN_TRIGGERS)
    if recovery_state and recovery_state.last_blocked_retry_key:
        triggers.append("repeated_same_action_blocked")
    if recovery_state and is_no_progress(recovery_state):
        triggers.append("no_progress_same_action_repeated")
    return triggers


def _anti_patterns(task_state: TaskState, recovery_state: Optional[RecoveryState]) -> List[str]:
    """Anti-patterns to avoid."""
    patterns = list(DEFAULT_ANTI_PATTERNS)
    if recovery_state and recovery_state.blocked_goal_action and recovery_state.missing_prerequisites:
        patterns.append(
            "Do not retry the blocked mutating action until the missing prerequisite is satisfied; "
            "take the prerequisite-satisfying step first (e.g. get user profile for profile_grounded)."
        )
    if recovery_state and recovery_state.last_blocked_retry_key:
        patterns.append("Do not repeat the same blocked or failed action; try a different step or ask user.")
    if recovery_state and is_no_progress(recovery_state):
        patterns.append("Do not repeat the same action again; make progress via different tool or ask_user.")
    return patterns


def plan(
    task_state: TaskState,
    recovery_state: Optional[RecoveryState],
    step_index: int,
    max_num_steps: int,
) -> PlanResult:
    """
    Produce a short-horizon plan from current TaskState and RecoveryState.
    Domain-agnostic; no hardcoded airline/retail flows.
    """
    current_goal = _current_goal(task_state)
    active_constraints = _active_constraints(task_state)
    relevant_known_facts = _relevant_known_facts(task_state)
    subgoal = _infer_subgoal(task_state, recovery_state)
    preferred = _preferred_next_action_type(task_state, recovery_state)
    next_steps = _next_steps(task_state, preferred, recovery_state)
    success_checkpoint = _success_checkpoint(task_state, recovery_state)
    replan_triggers = _replan_triggers(task_state, recovery_state)
    anti_patterns_to_avoid = _anti_patterns(task_state, recovery_state)

    planning_notes: Optional[str] = None
    if step_index >= max_num_steps - 2:
        planning_notes = "Turn budget nearly exhausted; prefer concluding or safe terminate."

    return PlanResult(
        current_goal=current_goal,
        active_constraints=active_constraints,
        relevant_known_facts=relevant_known_facts,
        subgoal=subgoal,
        next_steps=next_steps,
        preferred_next_action_type=preferred,
        success_checkpoint=success_checkpoint,
        replan_triggers=replan_triggers,
        anti_patterns_to_avoid=anti_patterns_to_avoid,
        planning_notes=planning_notes,
    )


def build_planner_guidance_text(plan_result: PlanResult) -> str:
    """
    Build a compact, line-based planner summary for injection into proposer message.
    Kept short to avoid blowing up context length.
    """
    lines: List[str] = ["Plan:"]
    lines.append(f"Goal: {plan_result.current_goal[:200]}")
    lines.append(f"Subgoal: {plan_result.subgoal[:200]}")
    if plan_result.active_constraints:
        lines.append("Constraints: " + "; ".join(plan_result.active_constraints[:5]))
    if plan_result.relevant_known_facts:
        lines.append("Facts: " + "; ".join(plan_result.relevant_known_facts[:8]))
    lines.append(f"Preferred next: {plan_result.preferred_next_action_type}")
    for i, step in enumerate(plan_result.next_steps[:3], 1):
        lines.append(f"Step{i}: [{step.step_type}] {step.rationale[:100]}")
    lines.append(f"Progress means: {plan_result.success_checkpoint[:150]}")
    if plan_result.replan_triggers:
        lines.append("Replan if: " + ", ".join(plan_result.replan_triggers[:4]))
    if plan_result.anti_patterns_to_avoid:
        lines.append("Avoid: " + "; ".join(plan_result.anti_patterns_to_avoid[:3]))
    if plan_result.planning_notes:
        lines.append(f"Note: {plan_result.planning_notes}")
    return "\n".join(lines)

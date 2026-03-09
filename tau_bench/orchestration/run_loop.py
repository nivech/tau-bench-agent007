# Copyright Sierra
# Minimal orchestrator run loop: init/reset → proposer → validator → executor → state update → finish_run.
# Only the orchestrator calls the logger.

import json
from typing import Any, Dict, List, Optional

from tau_bench.envs.base import Env
from tau_bench.orchestration.grounding import apply_grounding, build_grounded_facts_summary
from tau_bench.orchestration.logging import observation_summary
from tau_bench.orchestration.planner import build_planner_guidance_text, plan
from tau_bench.orchestration.task_state import TaskState, create_initial_task_state


def _is_prerequisite_satisfied(prereq: str, task_state: TaskState) -> bool:
    """True if the given prerequisite name is satisfied by current task_state (for prerequisite recovery sync)."""
    if prereq == "user_id":
        return bool(task_state.identity.user_id and str(task_state.identity.user_id).strip())
    if prereq == "profile_grounded":
        return task_state.identity.profile_grounded
    if prereq == "authenticated":
        return task_state.identity.authenticated
    if prereq in ("reservation_id", "reservation_context"):
        return bool((task_state.grounded.get("reservation_ids") or []) or task_state.grounded.get("reservation_details"))
    if prereq in ("order_id", "order_context"):
        return bool((task_state.grounded.get("order_ids") or []) or task_state.grounded.get("order_details"))
    return False
from tau_bench.orchestration.tool_outcomes import (
    classify_observation,
    is_explicit_completion_outcome_claim,
    is_mutating_tool,
    is_success_style_respond,
)
from tau_bench.orchestration.validator import ValidatorResult, validate_action
from tau_bench.orchestration.policy_guard import CODE_SUBJECT_AMBIGUITY, PolicyGuardResult, check_policy
from tau_bench.orchestration.recovery import (
    FailureCategory,
    RecoveryConfig,
    RecoveryInput,
    RecoveryState,
    action_retry_key,
    default_recovery_config,
    decide_recovery,
    detect_confirmation_satisfied,
    get_completion_guard_recovery_message,
    is_no_progress,
)
from tau_bench.types import Action, SolveResult, RESPOND_ACTION_NAME

# Trust boundary: only genuine user/env-originated content uses role="user".
# Orchestrator/recovery/planner/validator/policy-guard guidance uses this role and prefix.
ORCHESTRATOR_GUIDANCE_ROLE = "system"
ORCHESTRATOR_GUIDANCE_PREFIX = "[Runtime guidance] "


def _runtime_guidance_message(content: str) -> Dict[str, Any]:
    """Build a message for synthetic orchestrator guidance. Must not use role='user'."""
    return {"role": ORCHESTRATOR_GUIDANCE_ROLE, "content": ORCHESTRATOR_GUIDANCE_PREFIX + content}


def _append_rejection_to_messages(
    messages: List[Dict[str, Any]],
    next_message: Dict[str, Any],
    action: Action,
    rejection: str,
) -> None:
    """Append rejection to messages (validator or policy block path). Tool-calling format: assistant + tool message with rejection; else next_message + system guidance."""
    if action.name != RESPOND_ACTION_NAME and "tool_calls" in next_message and next_message.get("tool_calls"):
        next_message = {**next_message, "tool_calls": next_message["tool_calls"][:1]}
        messages.extend([
            next_message,
            {
                "role": "tool",
                "tool_call_id": next_message["tool_calls"][0]["id"],
                "name": next_message["tool_calls"][0]["function"]["name"],
                "content": rejection,
            },
        ])
    else:
        messages.extend([next_message, _runtime_guidance_message(rejection)])


# Logger protocol: has log_run_start, log_step_stage, write_trace_event, finish_run
RunLogger = Any

# Cap on consecutive proposer failures before terminating the run.
MAX_CONSECUTIVE_PROPOSER_FAILURES = 3


def run_orchestrated_loop(
    env: Env,
    proposer: Any,  # has generate_next_step(messages) -> (next_message, action, cost)
    run_logger: RunLogger,
    task_index: Optional[int],
    max_num_steps: int,
    domain: Optional[str] = None,
    use_recovery: bool = True,
) -> SolveResult:
    """Run one task: reset → start log → loop (propose → validate → execute → state update) → finish_run.
    TaskState is created at entry and updated each step for policy guard, planner, recovery, etc."""
    total_cost = 0.0
    steps = 0
    reward = 0.0
    num_validation_failures = 0
    info: Dict[str, Any] = {}
    messages: List[Dict[str, Any]] = []
    recovery_state: Optional[RecoveryState] = None
    try:
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        messages = [
            {"role": "system", "content": env.wiki},
            {"role": "user", "content": obs},
        ]
        task_state: TaskState = create_initial_task_state(
            domain=domain or "airline",
            task=env.task,
            initial_observation=obs,
        )
        recovery_state = RecoveryState()
        recovery_config = default_recovery_config(domain or "airline")
        run_logger.log_run_start()
        first_event = {
            "step_index": 0,
            "module": "orchestrator",
            "event_type": "run_start",
        }
        run_logger.write_trace_event(first_event)

        last_action: Optional[str] = None
        last_observation_summary: str = observation_summary(obs)
        done = False
        consecutive_proposer_failures = 0

        for step_index in range(1, max_num_steps + 1):
            # Lightweight state snapshot at beginning of step (no full message history)
            run_logger.write_trace_event({
                "step_index": step_index,
                "module": "orchestrator",
                "event_type": "state_snapshot",
                "last_action": last_action,
                "last_observation_summary": last_observation_summary,
                "messages_len": len(messages),
                "total_cost": total_cost,
                "done": done,
            })
            # Phase B: if pending side-effect and the most recent user message indicates confirmation, clear pending and set retry
            # Use the last message with role=="user" (scan from end) so we detect confirmation even if message order varies
            if use_recovery and recovery_state.pending_side_effect_action is not None and recovery_state.pending_confirmation_key:
                last_user_msg = None
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        last_user_msg = messages[i]
                        break
                if last_user_msg:
                    last_content = last_user_msg.get("content")
                    if isinstance(last_content, str) and detect_confirmation_satisfied(
                        last_content, recovery_state.pending_confirmation_key
                    ):
                        task_state.add_confirmation(recovery_state.pending_confirmation_key)
                        # Deterministic retry: re-execute the blocked action this step instead of relying on the proposer
                        recovery_state.retry_action_after_confirmation = recovery_state.pending_side_effect_action
                        recovery_state.pending_side_effect_action = None
                        recovery_state.pending_confirmation_key = None
                        recovery_state.pending_since_step = 0
                        recovery_state.awaiting_user_input = False
            # Phase C: no-progress check at start of step
            if use_recovery and is_no_progress(recovery_state):
                rec_input = RecoveryInput(
                    failure_type="no_progress",
                    action=Action(name=RESPOND_ACTION_NAME, kwargs={"content": ""}),
                    step_index=step_index,
                    max_num_steps=max_num_steps,
                    recovery_state=recovery_state,
                    last_action=last_action,
                )
                rec_decision = decide_recovery(rec_input, recovery_config)
                recovery_state.recovery_count_this_run += rec_decision.retry_budget_cost
                run_logger.write_trace_event({
                    "step_index": step_index,
                    "module": "recovery",
                    "event_type": "recovery_decision",
                    "failure_trigger": rec_decision.failure_type,
                    "diagnosis": rec_decision.diagnosis,
                    "chosen_strategy": rec_decision.proposed_strategy,
                    **rec_decision.trace_metadata,
                    "recovery_count_this_run": recovery_state.recovery_count_this_run,
                })
                if rec_decision.terminal_reason:
                    run_logger.finish_run(
                        exit_reason="recovery_terminated",
                        steps=steps,
                        total_cost=total_cost,
                        reward=reward,
                        done=False,
                        counters={"num_validation_failures": num_validation_failures, "num_recovery_invocations": recovery_state.recovery_count_this_run},
                    )
                    return SolveResult(reward=reward, info=info, messages=messages, total_cost=total_cost)
            # Inject grounded facts summary and planner guidance into the last message with string content.
            # Only that message's content is prepended; tool-calling format (roles, tool_calls) is preserved.
            summary = build_grounded_facts_summary(task_state)
            plan_result = plan(task_state, recovery_state, step_index, max_num_steps)
            plan_text = build_planner_guidance_text(plan_result)
            run_logger.write_trace_event({
                "step_index": step_index,
                "module": "planner",
                "event_type": "planner_invoked",
                "subgoal": plan_result.subgoal,
                "preferred_next_action_type": plan_result.preferred_next_action_type,
                "success_checkpoint": plan_result.success_checkpoint,
                "replan_triggers": plan_result.replan_triggers,
                "planning_notes": plan_result.planning_notes,
            })
            for i in range(len(messages) - 1, -1, -1):
                if "content" in messages[i] and isinstance(messages[i].get("content"), str):
                    messages[i]["content"] = f"[{summary}]\n[{plan_text}]\n\n{messages[i]['content']}"
                    break
            # Deterministic retry after confirmation: replay the stored action instead of calling the proposer.
            # For confirmation-only flows, the stored action is still valid at retry time (no state change
            # or user input that should modify it). For future recovery types (clarification, disambiguation,
            # tool failure), consider validating or updating the action before retry (e.g. stale arguments,
            # schema refresh, or "retry with update" from recovered user input).
            if use_recovery and recovery_state.retry_action_after_confirmation is not None:
                action = recovery_state.retry_action_after_confirmation
                retry_id = f"retry-{step_index}-{action.name}"
                next_message = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": retry_id,
                            "type": "function",
                            "function": {
                                "name": action.name,
                                "arguments": json.dumps(action.kwargs) if action.kwargs else "{}",
                            },
                        }
                    ],
                }
                cost = 0.0
                recovery_state.retry_action_after_confirmation = None
                run_logger.write_trace_event({
                    "step_index": step_index,
                    "module": "orchestrator",
                    "event_type": "retry_after_confirmation",
                    "action_name": action.name,
                })
            elif use_recovery and recovery_state.retry_action_after_prereq is not None:
                action = recovery_state.retry_action_after_prereq
                retry_id = f"retry-prereq-{step_index}-{action.name}"
                next_message = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": retry_id,
                            "type": "function",
                            "function": {
                                "name": action.name,
                                "arguments": json.dumps(action.kwargs) if action.kwargs else "{}",
                            },
                        }
                    ],
                }
                cost = 0.0
                recovery_state.retry_action_after_prereq = None
                run_logger.write_trace_event({
                    "step_index": step_index,
                    "module": "orchestrator",
                    "event_type": "retry_after_prereq",
                    "action_name": action.name,
                })
            else:
                try:
                    next_message, action, cost = proposer.generate_next_step(messages)
                    consecutive_proposer_failures = 0
                except Exception as e:  # noqa: BLE001
                    run_logger.write_trace_event({
                        "step_index": step_index,
                        "module": "proposer",
                        "event_type": "proposer_exception",
                        "error": str(e),
                    })
                    rejection = f"Proposer failed: {e!s}"
                    next_message = {"role": "assistant", "content": ""}
                    action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": ""})
                    cost = 0.0
                    if use_recovery:
                        rec_input = RecoveryInput(
                            failure_type=FailureCategory.proposer_error.value,
                            action=action,
                            step_index=step_index,
                            max_num_steps=max_num_steps,
                            source_message=str(e),
                            domain=task_state.domain,
                            recovery_state=recovery_state,
                            last_action=last_action,
                        )
                        rec_decision = decide_recovery(rec_input, recovery_config)
                        recovery_state.recovery_count_this_run += rec_decision.retry_budget_cost
                        run_logger.write_trace_event({
                            "step_index": step_index,
                            "module": "recovery",
                            "event_type": "recovery_decision",
                            "failure_trigger": rec_decision.failure_type,
                            "diagnosis": rec_decision.diagnosis,
                            "chosen_strategy": rec_decision.proposed_strategy,
                            **rec_decision.trace_metadata,
                            "recovery_count_this_run": recovery_state.recovery_count_this_run,
                        })
                    _append_rejection_to_messages(messages, next_message, action, rejection)
                    last_action = "proposer_error"
                    last_observation_summary = observation_summary(rejection)
                    steps = step_index
                    consecutive_proposer_failures += 1
                    if use_recovery:
                        recovery_state.recent_last_actions = (recovery_state.recent_last_actions + [last_action])[-3:]
                    if consecutive_proposer_failures >= MAX_CONSECUTIVE_PROPOSER_FAILURES:
                        run_logger.finish_run(
                            exit_reason="proposer_repeated_failure",
                            steps=steps,
                            total_cost=total_cost,
                            reward=reward,
                            done=False,
                            counters={
                                "num_validation_failures": num_validation_failures,
                                "num_recovery_invocations": recovery_state.recovery_count_this_run,
                            },
                        )
                        return SolveResult(reward=reward, info=info, messages=messages, total_cost=total_cost)
                    continue
            total_cost += cost
            # Proposer stage (log + trace)
            run_logger.log_step_stage(
                step_index,
                "proposer",
                {"action_name": action.name, "cost": cost},
            )
            run_logger.write_trace_event({
                "step_index": step_index,
                "module": "proposer",
                "event_type": "proposed",
                "action_name": action.name,
                "cost": cost,
            })
            # Validator stage (structured result; log + trace)
            v_result: ValidatorResult = validate_action(env, action, step_index)
            run_logger.log_step_stage(
                step_index,
                "validator",
                {"allowed": v_result.allowed, "code": v_result.code, "message": v_result.message, "action_name": action.name},
            )
            run_logger.write_trace_event({
                "step_index": step_index,
                "module": "validator",
                "event_type": "validated",
                "allowed": v_result.allowed,
                "code": v_result.code,
                "message": v_result.message,
                "action_name": action.name,
            })
            if not v_result.allowed:
                num_validation_failures += 1
                rejection = f"Validation failed: {v_result.message}"
                if use_recovery:
                    rec_input = RecoveryInput(
                        failure_type=FailureCategory.validation_error.value,
                        action=action,
                        step_index=step_index,
                        max_num_steps=max_num_steps,
                        source_code=v_result.code,
                        source_message=v_result.message,
                        domain=task_state.domain,
                        recovery_state=recovery_state,
                        last_action=last_action,
                    )
                    rec_decision = decide_recovery(rec_input, recovery_config)
                    recovery_state.recovery_count_this_run += rec_decision.retry_budget_cost
                    run_logger.write_trace_event({
                        "step_index": step_index,
                        "module": "recovery",
                        "event_type": "recovery_decision",
                        "failure_trigger": rec_decision.failure_type,
                        "failure_code": rec_decision.trace_metadata.get("failure_code"),
                        "diagnosis": rec_decision.diagnosis,
                        "chosen_strategy": rec_decision.proposed_strategy,
                        "retry_key": rec_decision.retry_key,
                        "retry_allowed": rec_decision.retry_allowed,
                        "retry_budget_cost": rec_decision.retry_budget_cost,
                        "terminal_reason": rec_decision.terminal_reason,
                        **rec_decision.trace_metadata,
                        "recovery_count_this_run": recovery_state.recovery_count_this_run,
                    })
                _append_rejection_to_messages(messages, next_message, action, rejection)
                last_action = action.name
                last_observation_summary = observation_summary(rejection)
                steps = step_index
                if use_recovery:
                    recovery_state.recent_last_actions = (recovery_state.recent_last_actions + [last_action])[-3:]
                continue
            # Policy guard stage (after validator, before executor)
            p_result: PolicyGuardResult = check_policy(env, action, task_state)
            run_logger.log_step_stage(
                step_index,
                "policy_guard",
                {"allowed": p_result.allowed, "code": p_result.code, "message": p_result.message, "action_name": action.name},
            )
            run_logger.write_trace_event({
                "step_index": step_index,
                "module": "policy_guard",
                "event_type": "blocked" if not p_result.allowed else "checked",
                "allowed": p_result.allowed,
                "code": p_result.code,
                "message": p_result.message,
                "action_name": action.name,
            })
            if not p_result.allowed:
                rejection = f"Policy blocked: {p_result.message}"
                task_state.set_last_error(f"Policy blocked ({p_result.code}): {p_result.message}")
                if p_result.code == CODE_SUBJECT_AMBIGUITY:
                    task_state.subject_resolution_status = "ambiguous"
                # Record that a mutating tool was attempted (so completion guard knows task expects mutation)
                if is_mutating_tool(task_state.domain, action.name):
                    task_state.record_mutating_attempt(action.name)
                if use_recovery:
                    rec_input = RecoveryInput(
                        failure_type=FailureCategory.policy_block.value,
                        action=action,
                        step_index=step_index,
                        max_num_steps=max_num_steps,
                        source_code=p_result.code,
                        source_message=p_result.message,
                        missing_prerequisites=p_result.missing_prerequisites,
                        domain=task_state.domain,
                        recovery_state=recovery_state,
                        last_action=last_action,
                    )
                    rec_decision = decide_recovery(rec_input, recovery_config)
                    recovery_state.recovery_count_this_run += rec_decision.retry_budget_cost
                    # Phase B: apply ASK_USER_CONFIRMATION state updates
                    if rec_decision.proposed_strategy == "ASK_USER_CONFIRMATION":
                        su = rec_decision.state_updates
                        if "set_pending_side_effect_action" in su:
                            recovery_state.pending_side_effect_action = su["set_pending_side_effect_action"]
                            recovery_state.pending_confirmation_key = su.get("pending_confirmation_key") or "booking_confirmed"
                            recovery_state.pending_since_step = step_index
                            recovery_state.awaiting_user_input = True
                        if rec_decision.message_to_user:
                            rejection = rejection + "\n\n" + rec_decision.message_to_user
                    # Prerequisite-targeted recovery: carry blocked intent so planner can steer next step
                    if rec_decision.proposed_strategy == "SATISFY_PREREQUISITE":
                        su = rec_decision.state_updates
                        if "blocked_goal_action" in su:
                            recovery_state.blocked_goal_action = su["blocked_goal_action"]
                            recovery_state.missing_prerequisites = list(su.get("missing_prerequisites") or [])
                            recovery_state.resume_intent_after_prereq = bool(su.get("resume_intent_after_prereq", True))
                        if rec_decision.replanning_hint:
                            rejection = rejection + "\n\n" + rec_decision.replanning_hint
                    run_logger.write_trace_event({
                        "step_index": step_index,
                        "module": "recovery",
                        "event_type": "recovery_decision",
                        "failure_trigger": rec_decision.failure_type,
                        "failure_code": rec_decision.trace_metadata.get("failure_code"),
                        "diagnosis": rec_decision.diagnosis,
                        "chosen_strategy": rec_decision.proposed_strategy,
                        "retry_key": rec_decision.retry_key,
                        "retry_allowed": rec_decision.retry_allowed,
                        "retry_budget_cost": rec_decision.retry_budget_cost,
                        "terminal_reason": rec_decision.terminal_reason,
                        **rec_decision.trace_metadata,
                        "recovery_count_this_run": recovery_state.recovery_count_this_run,
                    })
                _append_rejection_to_messages(messages, next_message, action, rejection)
                last_action = action.name
                last_observation_summary = observation_summary(rejection)
                steps = step_index
                if use_recovery:
                    recovery_state.last_blocked_retry_key = action_retry_key(action)
                    recovery_state.recent_last_actions = (recovery_state.recent_last_actions + [action.name])[-3:]
                continue
            # Completion guard: reward 1.0 requires real env.step(tool) success, not assistant wording.
            # Block success-style respond when no grounded mutation success. Allow through when recovery
            # is awaiting user input (confirmation, clarification)—except block explicit outcome claims
            # (e.g. "Your booking is confirmed") even during that flow.
            if action.name == RESPOND_ACTION_NAME and use_recovery:
                content = (action.kwargs or {}).get("content") or ""
                if is_success_style_respond(content):
                    pending = recovery_state.pending_side_effect_action if use_recovery else None
                    requires_grounded = task_state.requires_grounded_completion(pending)
                    awaiting_user = recovery_state.awaiting_user_input
                    explicit_outcome_claim = is_explicit_completion_outcome_claim(content)
                    block = (
                        requires_grounded
                        and len(task_state.successful_mutations) == 0
                        and (not awaiting_user or explicit_outcome_claim)
                    )
                    if block:
                        recovery_message = get_completion_guard_recovery_message()
                        messages.extend([next_message, _runtime_guidance_message(recovery_message)])
                        recovery_state.recovery_count_this_run += 1
                        run_logger.write_trace_event({
                            "step_index": step_index,
                            "module": "completion_guard",
                            "event_type": "completion_guard_blocked",
                            "recovery_count_this_run": recovery_state.recovery_count_this_run,
                        })
                        last_action = RESPOND_ACTION_NAME
                        last_observation_summary = observation_summary(recovery_message)
                        steps = step_index
                        continue
            # Record mutating attempt when we are about to execute a mutating tool (policy already allowed)
            if action.name != RESPOND_ACTION_NAME and is_mutating_tool(task_state.domain, action.name):
                task_state.record_mutating_attempt(action.name)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            obs_summary = observation_summary(env_response.observation)
            run_logger.log_step_stage(
                step_index,
                "executor",
                {"reward": reward, "done": env_response.done, "observation_summary": obs_summary},
            )
            trace_evt = {
                "step_index": step_index,
                "module": "executor",
                "event_type": "step",
                "action_name": action.name,
                "reward": reward,
                "done": env_response.done,
                "total_cost": total_cost,
                "observation_summary": obs_summary,
            }
            run_logger.write_trace_event(trace_evt)

            task_state.update_after_step(action.name, env_response.observation)
            # Record successful mutation when a mutating tool ran and returned non-error (real env.step result only)
            if action.name != RESPOND_ACTION_NAME:
                outcome = classify_observation(
                    task_state.domain,
                    action.name,
                    env_response.observation,
                )
                if outcome.is_mutating and outcome.execution_succeeded:
                    task_state.record_successful_mutation(action.name)
            # Phase D: tool execution error recovery
            if use_recovery and env_response.observation.strip().startswith("Error:"):
                rec_input = RecoveryInput(
                    failure_type="tool_execution_error",
                    action=action,
                    step_index=step_index,
                    max_num_steps=max_num_steps,
                    source_message=env_response.observation[:500],
                    tool_observation_summary=obs_summary,
                    domain=task_state.domain,
                    recovery_state=recovery_state,
                    last_action=last_action,
                )
                rec_decision = decide_recovery(rec_input, recovery_config)
                recovery_state.recovery_count_this_run += rec_decision.retry_budget_cost
                run_logger.write_trace_event({
                    "step_index": step_index,
                    "module": "recovery",
                    "event_type": "recovery_decision",
                    "failure_trigger": rec_decision.failure_type,
                    "diagnosis": rec_decision.diagnosis,
                    "chosen_strategy": rec_decision.proposed_strategy,
                    "retry_key": rec_decision.retry_key,
                    **rec_decision.trace_metadata,
                    "recovery_count_this_run": recovery_state.recovery_count_this_run,
                })
            # Grounding only for env tool steps, not for terminal respond actions.
            if action.name != RESPOND_ACTION_NAME:
                apply_grounding(
                    env,
                    task_state.domain,
                    action,
                    env_response.observation,
                    task_state,
                )
                # Prerequisite satisfaction: if we had a blocked mutating goal, drop now-satisfied prereqs;
                # when all are satisfied, schedule deterministic retry of the blocked action next step.
                if use_recovery and recovery_state.blocked_goal_action is not None and recovery_state.missing_prerequisites:
                    still_missing = [
                        p for p in recovery_state.missing_prerequisites
                        if not _is_prerequisite_satisfied(p, task_state)
                    ]
                    recovery_state.missing_prerequisites = still_missing
                    if not still_missing:
                        next_retry_action = recovery_state.blocked_goal_action
                        recovery_state.retry_action_after_prereq = next_retry_action
                        recovery_state.blocked_goal_action = None
                        recovery_state.resume_intent_after_prereq = False
                        run_logger.write_trace_event({
                            "step_index": step_index,
                            "module": "orchestrator",
                            "event_type": "prereq_satisfied_retry_scheduled",
                            "action_name": next_retry_action.name if next_retry_action else None,
                        })

            last_action = action.name
            last_observation_summary = obs_summary
            done = env_response.done

            # Phase B: after successful execution, clear pending if this was the pending side-effect action
            if use_recovery and recovery_state.pending_side_effect_action is not None:
                if action_retry_key(action) == action_retry_key(recovery_state.pending_side_effect_action):
                    recovery_state.pending_side_effect_action = None
                    recovery_state.pending_confirmation_key = None
                    recovery_state.awaiting_user_input = False
            if use_recovery:
                recovery_state.recent_last_actions = (recovery_state.recent_last_actions + [action.name])[-3:]

            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            steps = step_index
            if env_response.done:
                run_logger.finish_run(
                    exit_reason="success",
                    steps=steps,
                    total_cost=total_cost,
                    reward=reward,
                    done=True,
                    counters={
                        "num_validation_failures": num_validation_failures,
                        "num_recovery_invocations": recovery_state.recovery_count_this_run,
                    },
                )
                return SolveResult(
                    reward=reward,
                    info=info,
                    messages=messages,
                    total_cost=total_cost,
                )

        run_logger.finish_run(
            exit_reason="budget_exhausted",
            steps=steps,
            total_cost=total_cost,
            reward=reward,
            done=False,
            counters={
                "num_validation_failures": num_validation_failures,
                "num_recovery_invocations": recovery_state.recovery_count_this_run,
            },
        )
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )
    except Exception as e:
        run_logger.finish_run(
            exit_reason="error",
            steps=steps,
            total_cost=total_cost,
            reward=reward,
            done=False,
            counters={
                "error": 1,
                "num_validation_failures": num_validation_failures,
                "num_recovery_invocations": recovery_state.recovery_count_this_run if recovery_state else 0,
            },
        )
        raise

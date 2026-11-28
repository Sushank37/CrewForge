"""Specialist worker execution for DAG tasks and targeted peer-message follow-ups."""

from __future__ import annotations

from dataclasses import dataclass, field

from crewforge.config import sanitize_error_message
from crewforge.core.executor import ModelExecutor
from crewforge.core.ledger import ClaimLedger, ClaimRevisionError
from crewforge.core.messaging import MessageBus
from crewforge.core.requirements import deliverable_lines
from crewforge.prompts.worker import (
    build_followup_user_prompt,
    build_worker_system_prompt,
    build_worker_user_prompt,
)
from crewforge.schemas import (
    AgentMessage,
    AgentSpec,
    Claim,
    Evidence,
    ExecutionEvent,
    ExecutionPlan,
    FollowUpOutput,
    SpecialistOutput,
    TaskSpec,
    ToolExecutionRecord,
)
from crewforge.storage.database import CrossRunReferenceError
from crewforge.tools.arithmetic import build_provenance_catalog


@dataclass
class WorkerExecutionOutcome:
    """Structured outcome returned by a specialist task or follow-up execution."""

    task_id: str
    agent_id: str
    findings: str
    claims: list[Claim] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    tool_executions: list[ToolExecutionRecord] = field(default_factory=list)
    sent_messages: list[AgentMessage] = field(default_factory=list)
    consumed_messages: list[AgentMessage] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)


class SpecialistWorker:
    """Executes assigned tasks and peer-communication checkpoints for a specialist agent."""

    def __init__(
        self,
        *,
        executor: ModelExecutor,
        ledger: ClaimLedger,
        message_bus: MessageBus,
    ) -> None:
        self._executor = executor
        self._ledger = ledger
        self._bus = message_bus

    @staticmethod
    def _build_agent_labels(plan: ExecutionPlan) -> tuple[dict[str, str], list[str]]:
        labels: dict[str, str] = {"lead": "Lead"}
        peer_directory: list[str] = ["lead"]
        for agent in plan.agents:
            ref = agent.agent_ref or agent.agent_id
            labels[agent.agent_id] = f"{ref} ({agent.role})"
            peer_directory.append(f"{ref} [{agent.role}]")
        return labels, peer_directory

    async def execute_task(
        self,
        *,
        run_id: str,
        original_prompt: str,
        plan: ExecutionPlan,
        agent: AgentSpec,
        task: TaskSpec,
        completed_tasks_by_id: dict[str, TaskSpec],
        run_deadline_monotonic: float | None = None,
        compact_recovery: bool = False,
    ) -> WorkerExecutionOutcome:
        """Execute a specialist's assigned task with dependency context and inbox delivery."""
        dependency_findings: list[tuple[str, str]] = []
        dep_task_ids: set[str] = set()
        for dep_id in task.depends_on:
            dep_task = completed_tasks_by_id.get(dep_id)
            if dep_task is not None and dep_task.findings:
                label = f"{dep_task.task_ref or dep_task.task_id}: {dep_task.title}"
                dependency_findings.append((label, dep_task.findings))
                dep_task_ids.add(dep_id)

        dep_claims_summary: list[str] = []
        if dep_task_ids:
            all_claims = await self._ledger.get_claims(run_id, latest_only=True)
            for c in all_claims:
                if c.task_id in dep_task_ids:
                    dep_claims_summary.append(
                        f"{c.claim_id} (v{c.version}, {c.category.value}): {c.statement}"
                    )

        dependency_text_blob = "\n".join(text for _, text in dependency_findings)
        inbound_messages = (
            await self._bus.read_inbox(
                run_id=run_id,
                recipient_id=agent.agent_id,
                mark_delivered=True,
            )
            if self._executor._settings.enable_peer_messaging
            else []
        )
        agent_labels, peer_directory = self._build_agent_labels(plan)

        all_claims_run = await self._ledger.get_claims(run_id, latest_only=False)
        verified_evd = [
            e for e in await self._ledger._db.get_evidence_for_run(run_id) if e.verified_in_input
        ]
        prior_tools = await self._ledger._db.get_tool_executions_for_run(run_id)
        prov_catalog = build_provenance_catalog(
            original_prompt=original_prompt,
            verified_evidence=verified_evd,
            all_claims=all_claims_run,
            tool_records=prior_tools,
            assumptions=plan.assumptions,
            run_id=run_id,
        )

        system_prompt = build_worker_system_prompt(agent)
        user_prompt = build_worker_user_prompt(
            original_prompt=original_prompt,
            task=task,
            acceptance_criteria=plan.acceptance_criteria,
            dependency_findings=dependency_findings,
            dependency_claims_summary=dep_claims_summary,
            shared_assumptions=plan.assumptions,
            inbound_messages=inbound_messages,
            agent_labels=agent_labels,
            peer_directory=peer_directory,
            provenance_lines=prov_catalog.prompt_summary_lines(),
            user_requirements=plan.user_requirements,
            assigned_deliverables=deliverable_lines([
                d for d in plan.deliverables
                if d.task_ref == task.task_ref
            ]),
        )
        if compact_recovery:
            user_prompt += (
                "\n\nTRUNCATION RECOVERY: Your previous structured response hit "
                "max_output_tokens. Return a much shorter, strictly valid JSON response: "
                "concise findings (<= 80 words), short verbatim evidence excerpts (<= 200 chars), "
                "and no repetition, while preserving all required atomic claims and calculations. "
                "If any requirement cannot fit, record it in `missing_information` or as an "
                "`is_unresolved=true` claim."
            )

        response = await self._executor.execute_structured(
            run_id=run_id,
            actor=agent.agent_id,
            operation="worker",
            prompt=user_prompt,
            response_model=SpecialistOutput,
            system_prompt=system_prompt,
            run_deadline_monotonic=run_deadline_monotonic,
        )
        if inbound_messages:
            await self._bus.mark_consumed(
                run_id=run_id,
                recipient_id=agent.agent_id,
                messages=inbound_messages,
            )

        output = response.parsed
        (
            persisted_evidence,
            persisted_claims,
            tool_records,
        ) = await self._ledger.record_specialist_outputs(
            run_id=run_id,
            agent_id=agent.agent_id,
            task_id=task.task_id,
            original_prompt=original_prompt,
            dependency_findings_text=dependency_text_blob,
            evidence_proposals=output.evidence_proposals,
            claim_proposals=output.claims,
            plan_assumptions=plan.assumptions,
        )
        sent_messages = (
            await self._bus.dispatch_instructions(
                run_id=run_id,
                sender_id=agent.agent_id,
                instructions=output.outgoing_messages,
                agents=plan.agents,
                created_claim_ids=(
                    [c.claim_id for c in persisted_claims]
                    if len(persisted_claims) == len(output.claims) else []
                ),
            )
            if self._executor._settings.enable_peer_messaging
            else []
        )

        return WorkerExecutionOutcome(
            task_id=task.task_id,
            agent_id=agent.agent_id,
            findings=output.findings,
            claims=persisted_claims,
            evidence=persisted_evidence,
            tool_executions=tool_records,
            sent_messages=sent_messages,
            consumed_messages=inbound_messages,
            assumptions=output.assumptions,
            uncertainties=output.uncertainties,
            missing_information=output.missing_information,
        )

    async def execute_followup(
        self,
        *,
        run_id: str,
        original_prompt: str,
        plan: ExecutionPlan,
        agent: AgentSpec,
        prior_findings: str,
        prior_claims: list[Claim],
        allow_outgoing_messages: bool = True,
        run_deadline_monotonic: float | None = None,
    ) -> WorkerExecutionOutcome | None:
        """Run a bounded peer-communication follow-up with explicit ID/version targeting."""
        inbound_messages = await self._bus.read_inbox(
            run_id=run_id,
            recipient_id=agent.agent_id,
            mark_delivered=True,
        )
        if not inbound_messages:
            return None

        agent_labels, peer_directory = self._build_agent_labels(plan)
        claims_summary = [
            f"claim_id={c.claim_id}, expected_version={c.version}: {c.statement}"
            for c in prior_claims
        ]
        system_prompt = build_worker_system_prompt(agent)
        user_prompt = build_followup_user_prompt(
            original_prompt=original_prompt,
            prior_findings=prior_findings,
            prior_claims_summary=claims_summary,
            inbound_messages=inbound_messages,
            agent_labels=agent_labels,
            peer_directory=peer_directory,
        )

        cited = {cid for m in inbound_messages for cid in m.related_claim_ids}
        trace = await self._ledger._db.get_run_trace(run_id)
        if trace is not None:
            catalog = build_provenance_catalog(
                original_prompt=original_prompt, verified_evidence=trace.evidence,
                all_claims=trace.claims, tool_records=trace.tool_executions, run_id=run_id,
            )
            latest = {c.claim_id: c for c in sorted(trace.claims, key=lambda c: c.version)}
            peer_claims = [
                f"- {c.claim_id}@v{c.version} ({c.status.value}): {c.statement}"
                for cid in sorted(cited) if (c := latest.get(cid))
            ]
            user_prompt += "\n\nCited peer claims (check against primary inputs):\n" + "\n".join(
                peer_claims
            )
            user_prompt += "\n\nPrimary sources and reusable results:\n" + "\n".join(
                catalog.prompt_summary_lines()
            )

        response = await self._executor.execute_structured(
            run_id=run_id,
            actor=agent.agent_id,
            operation="worker_followup",
            prompt=user_prompt,
            response_model=FollowUpOutput,
            system_prompt=system_prompt,
            run_deadline_monotonic=run_deadline_monotonic,
        )
        await self._bus.mark_consumed(
            run_id=run_id,
            recipient_id=agent.agent_id,
            messages=inbound_messages,
        )

        followup = response.parsed
        inbound_blob = "\n".join(m.content for m in inbound_messages)
        task_id = prior_claims[0].task_id if prior_claims else next(
            t.task_id for t in plan.tasks if t.assigned_agent_id == agent.agent_id
        )

        new_evidence, new_claims, tool_records = await self._ledger.record_specialist_outputs(
            run_id=run_id,
            agent_id=agent.agent_id,
            task_id=task_id,
            original_prompt=original_prompt,
            dependency_findings_text=f"{prior_findings}\n{inbound_blob}",
            evidence_proposals=followup.evidence_proposals,
            claim_proposals=followup.new_claims,
            plan_assumptions=plan.assumptions,
        )

        updated_claims: list[Claim] = list(new_claims)
        for rev in followup.revised_claims:
            evd_ids = [
                new_evidence[e_idx].evidence_id
                for e_idx in rev.evidence_indices
                if 0 <= e_idx < len(new_evidence)
            ]
            try:
                revised = await self._ledger.apply_targeted_revision(
                    run_id=run_id,
                    target_claim_id=rev.target_claim_id,
                    expected_version=rev.expected_version,
                    new_statement=rev.new_statement,
                    author_agent_id=agent.agent_id,
                    original_prompt=original_prompt,
                    dependency_findings_text=f"{prior_findings}\n{inbound_blob}",
                    category=rev.category,
                    additional_evidence_ids=evd_ids,
                    premise_claim_ids=rev.premise_claim_ids or None,
                    computation_request=rev.computation,
                    plan_assumptions=plan.assumptions,
                    assumptions=rev.assumptions or None,
                    revision_notes=rev.revision_notes,
                    is_unresolved=rev.is_unresolved,
                    confidence=rev.confidence,
                    require_same_author=True,
                )
                updated_claims.append(revised)
            except (ClaimRevisionError, CrossRunReferenceError) as exc:
                await self._ledger._db.append_event(
                    ExecutionEvent(
                        run_id=run_id,
                        event_type="revision_rejected",
                        actor=agent.agent_id,
                        message=sanitize_error_message(str(exc)),
                    )
                )

        sent_messages: list[AgentMessage] = []
        if allow_outgoing_messages and followup.outgoing_messages:
            sent_messages = await self._bus.dispatch_instructions(
                run_id=run_id,
                sender_id=agent.agent_id,
                instructions=followup.outgoing_messages,
                agents=plan.agents,
                created_claim_ids=(
                    [c.claim_id for c in updated_claims]
                    if len(new_claims) == len(followup.new_claims) else []
                ),
            )

        return WorkerExecutionOutcome(
            task_id=task_id,
            agent_id=agent.agent_id,
            findings=followup.updated_findings,
            claims=updated_claims,
            evidence=new_evidence,
            tool_executions=tool_records,
            sent_messages=sent_messages,
            consumed_messages=inbound_messages,
            uncertainties=followup.uncertainties,
        )

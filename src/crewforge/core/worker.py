"""Specialist worker execution for DAG tasks and bounded peer-message follow-ups."""

from __future__ import annotations

from dataclasses import dataclass, field

from crewforge.core.executor import ModelExecutor
from crewforge.core.ledger import ClaimLedger
from crewforge.core.messaging import MessageBus
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
    ExecutionPlan,
    FollowUpOutput,
    SpecialistOutput,
    TaskSpec,
)


@dataclass
class WorkerExecutionOutcome:
    """Structured outcome returned by a specialist task or follow-up execution."""

    task_id: str
    agent_id: str
    findings: str
    claims: list[Claim] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
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
    ) -> WorkerExecutionOutcome:
        """Execute a specialist's assigned task with dependency context and inbox delivery."""
        dependency_findings: list[tuple[str, str]] = []
        for dep_id in task.depends_on:
            dep_task = completed_tasks_by_id.get(dep_id)
            if dep_task is not None and dep_task.findings:
                label = f"{dep_task.task_ref or dep_task.task_id}: {dep_task.title}"
                dependency_findings.append((label, dep_task.findings))

        dependency_text_blob = "\n".join(text for _, text in dependency_findings)
        inbound_messages = await self._bus.read_inbox(
            run_id=run_id,
            recipient_id=agent.agent_id,
            mark_delivered=True,
        )
        agent_labels, peer_directory = self._build_agent_labels(plan)

        system_prompt = build_worker_system_prompt(agent)
        user_prompt = build_worker_user_prompt(
            original_prompt=original_prompt,
            task=task,
            acceptance_criteria=plan.acceptance_criteria,
            dependency_findings=dependency_findings,
            shared_assumptions=plan.assumptions,
            inbound_messages=inbound_messages,
            agent_labels=agent_labels,
            peer_directory=peer_directory,
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
        persisted_evidence, persisted_claims = await self._ledger.record_specialist_outputs(
            run_id=run_id,
            agent_id=agent.agent_id,
            task_id=task.task_id,
            original_prompt=original_prompt,
            dependency_findings_text=dependency_text_blob,
            evidence_proposals=output.evidence_proposals,
            claim_proposals=output.claims,
        )
        sent_messages = await self._bus.dispatch_instructions(
            run_id=run_id,
            sender_id=agent.agent_id,
            instructions=output.outgoing_messages,
            agents=plan.agents,
            created_claim_ids=[c.claim_id for c in persisted_claims],
        )

        return WorkerExecutionOutcome(
            task_id=task.task_id,
            agent_id=agent.agent_id,
            findings=output.findings,
            claims=persisted_claims,
            evidence=persisted_evidence,
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
        """Run a bounded peer-communication follow-up if the agent has pending messages."""
        inbound_messages = await self._bus.read_inbox(
            run_id=run_id,
            recipient_id=agent.agent_id,
            mark_delivered=True,
        )
        if not inbound_messages:
            return None

        agent_labels, peer_directory = self._build_agent_labels(plan)
        claims_summary = [f"{c.claim_id} (v{c.version}): {c.statement}" for c in prior_claims]
        system_prompt = build_worker_system_prompt(agent)
        user_prompt = build_followup_user_prompt(
            original_prompt=original_prompt,
            prior_findings=prior_findings,
            prior_claims_summary=claims_summary,
            inbound_messages=inbound_messages,
            agent_labels=agent_labels,
            peer_directory=peer_directory,
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
        task_id = prior_claims[0].task_id if prior_claims else None

        # Record any new evidence proposals and create revised/new claims cleanly.
        new_evidence: list[Evidence] = []
        if followup.evidence_proposals:
            new_evidence, _ = await self._ledger.record_specialist_outputs(
                run_id=run_id,
                agent_id=agent.agent_id,
                task_id=task_id or plan.tasks[0].task_id,
                original_prompt=original_prompt,
                dependency_findings_text=f"{prior_findings}\n{inbound_blob}",
                evidence_proposals=followup.evidence_proposals,
                claim_proposals=[],
            )

        updated_claims: list[Claim] = []
        if followup.revised_claims:
            for idx, prop in enumerate(followup.revised_claims):
                evd_ids = [
                    new_evidence[e_idx].evidence_id
                    for e_idx in prop.evidence_indices
                    if 0 <= e_idx < len(new_evidence)
                ]
                if idx < len(prior_claims):
                    revised = await self._ledger.revise_claim(
                        prior_claim=prior_claims[idx],
                        new_statement=prop.statement,
                        author_agent_id=agent.agent_id,
                        task_id=task_id,
                        additional_evidence_ids=evd_ids,
                        assumptions=prop.assumptions or None,
                        revision_notes="Updated after peer message delivery.",
                        is_unresolved=prop.is_unresolved,
                        confidence=prop.confidence,
                    )
                    updated_claims.append(revised)
                else:
                    _, created = await self._ledger.record_specialist_outputs(
                        run_id=run_id,
                        agent_id=agent.agent_id,
                        task_id=task_id or plan.tasks[0].task_id,
                        original_prompt=original_prompt,
                        dependency_findings_text=f"{prior_findings}\n{inbound_blob}",
                        evidence_proposals=[],
                        claim_proposals=[prop],
                    )
                    updated_claims.extend(created)

        sent_messages: list[AgentMessage] = []
        if allow_outgoing_messages and followup.outgoing_messages:
            sent_messages = await self._bus.dispatch_instructions(
                run_id=run_id,
                sender_id=agent.agent_id,
                instructions=followup.outgoing_messages,
                agents=plan.agents,
                created_claim_ids=[c.claim_id for c in updated_claims],
            )

        return WorkerExecutionOutcome(
            task_id=task_id or "",
            agent_id=agent.agent_id,
            findings=followup.updated_findings,
            claims=updated_claims,
            evidence=new_evidence,
            sent_messages=sent_messages,
            consumed_messages=inbound_messages,
            uncertainties=followup.uncertainties,
        )

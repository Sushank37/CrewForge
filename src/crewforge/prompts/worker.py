"""Prompt builders for specialist task execution and peer follow-ups."""

from __future__ import annotations

from crewforge.schemas import AgentMessage, AgentSpec, TaskSpec


def build_worker_system_prompt(agent: AgentSpec) -> str:
    """Build concise system instructions for a dynamically instantiated specialist."""
    caps = ", ".join(agent.capabilities) if agent.capabilities else "reasoning"
    return (
        f"You are specialist '{agent.role}' ({agent.agent_ref or agent.agent_id}).\n"
        f"Role instructions: {agent.instructions}\n"
        f"Allowed capabilities: [{caps}].\n\n"
        "Output rules:\n"
        "- Produce concise, accurate `findings` directly addressing your assigned task.\n"
        "- Express key points as atomic `claims` with explicit `category` (`given_fact`, "
        "`external_factual_assertion`, `computation`, `logical_inference`, `assumption`, "
        "`recommendation`, `uncertainty`) and `evidence_indices`.\n"
        "- For `computation` claims, attach `computation` (`expression`, `variables`, "
        "`input_provenance_refs`, `proposed_result`, `units`) with plain numeric values "
        "(no currency symbols like ₹ or $) so the arithmetic tool verifies exact numbers.\n"
        "- Only classify a claim as `external_factual_assertion` if it asserts outside facts "
        "not given in the prompt; use `logical_inference` or `recommendation` for analysis.\n"
        "- For derived `logical_inference` claims, link `premise_claim_ids` to upstream claims.\n"
        "- Only cite `user_prompt` or `dependency_finding` excerpts that exist verbatim in input; "
        "mark outside references as `external_unverified` and calculations as `calculation`.\n"
        "- Do not fabricate facts or emit hidden chain-of-thought; list `assumptions`, "
        "`uncertainties`, and `missing_information` explicitly.\n"
        "- Emit `outgoing_messages` only when a peer specialist or 'lead' genuinely needs "
        "clarification, warning, or handoff context."
    )


def _format_inbound_messages(
    inbound_messages: list[AgentMessage],
    agent_labels: dict[str, str],
) -> str:
    if not inbound_messages:
        return "None."
    lines: list[str] = []
    for msg in inbound_messages:
        sender_label = agent_labels.get(msg.sender_id, msg.sender_id)
        claims_note = (
            f" [claims: {', '.join(msg.related_claim_ids)}]" if msg.related_claim_ids else ""
        )
        lines.append(f"- [{msg.message_type}] from {sender_label}{claims_note}: {msg.content}")
    return "\n".join(lines)


def build_worker_user_prompt(
    *,
    original_prompt: str,
    task: TaskSpec,
    acceptance_criteria: list[str],
    dependency_findings: list[tuple[str, str]],
    dependency_claims_summary: list[str],
    shared_assumptions: list[str],
    inbound_messages: list[AgentMessage],
    agent_labels: dict[str, str],
    peer_directory: list[str],
) -> str:
    """Assemble bounded task context for a specialist without dumping full transcripts."""
    criteria = acceptance_criteria or task.acceptance_criteria
    criteria_block = "\n".join(f"- {c}" for c in criteria) if criteria else "- None specified."
    deliverables_block = (
        "\n".join(f"- {d}" for d in task.expected_outputs)
        if task.expected_outputs
        else "- Concise findings."
    )
    assumptions_block = (
        "\n".join(f"- {a}" for a in shared_assumptions) if shared_assumptions else "- None."
    )
    deps_block = (
        "\n\n".join(f"[{title}]\n{text}" for title, text in dependency_findings)
        if dependency_findings
        else "None."
    )
    dep_claims_block = (
        "\n".join(f"- {line}" for line in dependency_claims_summary)
        if dependency_claims_summary
        else "None."
    )
    peers_block = ", ".join(peer_directory) if peer_directory else "lead"
    messages_block = _format_inbound_messages(inbound_messages, agent_labels)

    return (
        f"Original user request:\n{original_prompt}\n\n"
        f"Assigned task ({task.task_ref or task.task_id}): {task.title}\n"
        f"Task description:\n{task.description}\n\n"
        f"Expected deliverables:\n{deliverables_block}\n\n"
        f"Relevant acceptance criteria:\n{criteria_block}\n\n"
        f"Shared plan assumptions:\n{assumptions_block}\n\n"
        f"Completed dependency findings:\n{deps_block}\n\n"
        "Upstream claim IDs (reference in premise_claim_ids when deriving):\n"
        f"{dep_claims_block}\n\n"
        f"Inbound peer messages:\n{messages_block}\n\n"
        "Available message recipients (use agent_ref or 'lead' or null for broadcast): "
        f"[{peers_block}]"
    )


def build_followup_user_prompt(
    *,
    original_prompt: str,
    prior_findings: str,
    prior_claims_summary: list[str],
    inbound_messages: list[AgentMessage],
    agent_labels: dict[str, str],
    peer_directory: list[str],
) -> str:
    """Assemble bounded peer-communication follow-up prompt with explicit claim IDs/versions."""
    claims_block = (
        "\n".join(f"- {item}" for item in prior_claims_summary)
        if prior_claims_summary
        else "- None recorded."
    )
    messages_block = _format_inbound_messages(inbound_messages, agent_labels)
    peers_block = ", ".join(peer_directory) if peer_directory else "lead"

    return (
        f"Original user request:\n{original_prompt}\n\n"
        f"Your prior findings:\n{prior_findings}\n\n"
        f"Your existing claims (target exact claim_id + expected_version when revising):\n"
        f"{claims_block}\n\n"
        f"New inbound peer messages requiring your consideration:\n{messages_block}\n\n"
        f"Available message recipients: [{peers_block}]\n\n"
        "Incorporate or answer the inbound peer messages concisely, update findings, "
        "and use `revised_claims` ONLY with exact `target_claim_id` and `expected_version`."
    )

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
        "Compact Output Rules:\n"
        "- Produce concise `findings` (<= 200 words, no prompt repetition) addressing your task.\n"
        "- Emit atomic `claims` sized dynamically to task complexity within your output-token "
        "budget (never combine unrelated results into one claim, and never omit requested "
        "calculations for brevity). Each independently requested numerical result must have its "
        "own atomic claim and deterministic `computation` record.\n"
        "- Each claim needs `statement`, `category` (`given_fact`, `external_factual_assertion`, "
        "`computation`, `logical_inference`, `assumption`, `recommendation`, `uncertainty`), "
        "`evidence_indices`, and `requirement_refs` (`req_1`).\n"
        "- For `computation` claims, attach ONE single-formula `computation` (`expression` using "
        "Python operators `+`, `-`, `*`, `/`, `**` — never LaTeX `^` or `sum_{...}` — and allowed "
        "functions `sum`, `min`, `max`, `avg`, `abs`, `round`, `comb(n,k)`, `perm(n,k)`, "
        "`factorial(n)`, `binomial_pmf(n,k,p)`, `binomial_cdf(n,k,p)` (P(X <= k)), "
        "`binomial_sf(n,k,p)` (inclusive P(X >= k)), "
        "or `sum(expr for i in range(a,b))`; short `variable_bindings` with `name`, plain-number "
        "`value`, `provenance_ref` set to the exact `up_N`, `doc_N`, `clm_...`, `tlx_...`, "
        "`const:...`, or `assumption:...` line containing that value, plus `proposed_result` and "
        "`units`).\n"
        "- Evidence `source_type` rules (prefer existing verified refs or short contiguous "
        "quotations <= 300 chars; never stitch non-contiguous fragments):\n"
        "  * `user_prompt`: verbatim substring of Original user request (`source_ref`: `up_N`).\n"
        "  * `external_retrieved`: verbatim quote from attached/retrieved doc "
        "(`doc_N` or filename).\n"
        "  * `dependency_finding`: verbatim substring of Completed dependency findings.\n"
        "  * `calculation`: deterministic arithmetic output.\n"
        "  * `external_unverified`: outside fact not in input/docs.\n"
        "- Use `logical_inference` (with `premise_claim_ids`), `recommendation`, or `uncertainty` "
        "rather than `external_factual_assertion` for analysis or missing-data statements.\n"
        "- Keep `assumptions`, `uncertainties`, and `missing_information` minimal "
        "(<= 3 items each).\n"
        "- Emit `outgoing_messages` only when a peer or 'lead' genuinely needs clarification."
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
    provenance_lines: list[str] | None = None,
    user_requirements: list[str] | None = None,
) -> str:
    """Assemble bounded task context for a specialist without dumping full transcripts."""
    reqs_block = (
        "\n".join(f"- [req_{idx}] {r}" for idx, r in enumerate(user_requirements or [], start=1))
        if user_requirements
        else "- None specified."
    )
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
    prov_block = "\n".join(provenance_lines) if provenance_lines else "None."
    peers_block = ", ".join(peer_directory) if peer_directory else "lead"
    messages_block = _format_inbound_messages(inbound_messages, agent_labels)

    return (
        f"Original user request:\n{original_prompt}\n\n"
        f"Target user requirements to address (tag in requirement_refs):\n{reqs_block}\n\n"
        f"Trusted provenance catalog (use aliases in input_provenance_refs):\n{prov_block}\n\n"
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

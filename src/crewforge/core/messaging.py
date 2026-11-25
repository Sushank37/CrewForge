"""Run-scoped peer-to-peer MessageBus with persisted delivery and consumption tracking."""

from __future__ import annotations

from crewforge.schemas import AgentMessage, AgentSpec, ExecutionEvent, OutgoingMessageInstruction
from crewforge.storage.database import Database


class MessageBus:
    """Delivers and tracks validated peer-to-peer and broadcast messages within a run."""

    def __init__(
        self,
        database: Database,
        *,
        max_message_chars: int = 1000,
        max_messages_per_turn: int = 3,
    ) -> None:
        self._db = database
        self._max_message_chars = max_message_chars
        self._max_messages_per_turn = max_messages_per_turn

    @staticmethod
    def build_recipient_lookup(agents: list[AgentSpec]) -> tuple[dict[str, str], list[str]]:
        """Map local handles (agent_ref, agent_id, role) to trusted agent_ids."""
        lookup: dict[str, str] = {"lead": "lead"}
        all_agent_ids: list[str] = []
        for agent in agents:
            all_agent_ids.append(agent.agent_id)
            lookup[agent.agent_id] = agent.agent_id
            if agent.agent_ref:
                lookup[agent.agent_ref] = agent.agent_id
                lookup[agent.agent_ref.lower()] = agent.agent_id
            lookup[agent.role.lower()] = agent.agent_id
        return lookup, all_agent_ids

    async def send_message(
        self,
        *,
        run_id: str,
        sender_id: str,
        recipient_id: str | None,
        all_agent_ids: list[str],
        message_type: str,
        content: str,
        related_claim_ids: list[str] | None = None,
    ) -> AgentMessage:
        """Persist and enqueue a direct or broadcast message inside the run."""
        bounded_content = content.strip()
        if len(bounded_content) > self._max_message_chars:
            bounded_content = bounded_content[: self._max_message_chars - 3] + "..."

        if recipient_id is None:
            delivery_targets = [aid for aid in all_agent_ids if aid != sender_id]
        else:
            delivery_targets = [recipient_id]

        msg = AgentMessage(
            run_id=run_id,
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=bounded_content,
            related_claim_ids=list(related_claim_ids or []),
            delivery_status="pending",
        )
        await self._db.insert_message(msg, delivery_targets)
        await self._db.append_event(
            ExecutionEvent(
                run_id=run_id,
                event_type="message_sent",
                actor=sender_id,
                message=(
                    f"Sent {message_type} message to "
                    f"{recipient_id if recipient_id is not None else 'broadcast'}."
                ),
                payload={
                    "message_id": msg.message_id,
                    "recipient_id": recipient_id,
                    "related_claim_ids": msg.related_claim_ids,
                },
            )
        )
        return msg

    async def dispatch_instructions(
        self,
        *,
        run_id: str,
        sender_id: str,
        instructions: list[OutgoingMessageInstruction],
        agents: list[AgentSpec],
        created_claim_ids: list[str],
    ) -> list[AgentMessage]:
        """Validate and send structured outgoing message instructions emitted by a specialist."""
        if not instructions:
            return []

        lookup, all_agent_ids = self.build_recipient_lookup(agents)
        sent: list[AgentMessage] = []

        for item in instructions[: self._max_messages_per_turn]:
            raw_recipient = item.recipient_ref.strip() if item.recipient_ref else ""
            if not raw_recipient or raw_recipient.lower() in {"all", "broadcast", "*"}:
                resolved_recipient: str | None = None
            else:
                resolved_recipient = lookup.get(raw_recipient) or lookup.get(raw_recipient.lower())
                if resolved_recipient is None or resolved_recipient == sender_id:
                    await self._db.append_event(
                        ExecutionEvent(
                            run_id=run_id,
                            event_type="message_rejected",
                            actor=sender_id,
                            message=(
                                f"Ignored invalid or self-addressed recipient '{raw_recipient}'."
                            ),
                        )
                    )
                    continue

            related_claims = [
                created_claim_ids[idx]
                for idx in item.related_claim_indices
                if 0 <= idx < len(created_claim_ids)
            ]
            msg = await self.send_message(
                run_id=run_id,
                sender_id=sender_id,
                recipient_id=resolved_recipient,
                all_agent_ids=all_agent_ids,
                message_type=item.message_type,
                content=item.content,
                related_claim_ids=related_claims,
            )
            sent.append(msg)

        return sent

    async def read_inbox(
        self,
        *,
        run_id: str,
        recipient_id: str,
        mark_delivered: bool = True,
    ) -> list[AgentMessage]:
        """Read pending/delivered inbox messages for an agent in the same run."""
        messages = await self._db.get_pending_messages_for_agent(
            run_id=run_id,
            recipient_id=recipient_id,
            mark_delivered=mark_delivered,
        )
        if messages and mark_delivered:
            await self._db.append_event(
                ExecutionEvent(
                    run_id=run_id,
                    event_type="messages_delivered",
                    actor=recipient_id,
                    message=f"Delivered {len(messages)} inbound message(s) to {recipient_id}.",
                    payload={"message_ids": [m.message_id for m in messages]},
                )
            )
        return messages

    async def mark_consumed(
        self,
        *,
        run_id: str,
        recipient_id: str,
        messages: list[AgentMessage],
    ) -> None:
        """Mark delivered messages as consumed once incorporated into model context."""
        if not messages:
            return
        msg_ids = [m.message_id for m in messages]
        await self._db.mark_messages_consumed(
            run_id=run_id,
            recipient_id=recipient_id,
            message_ids=msg_ids,
        )
        await self._db.append_event(
            ExecutionEvent(
                run_id=run_id,
                event_type="messages_consumed",
                actor=recipient_id,
                message=f"Consumed {len(msg_ids)} inbound message(s) in model context.",
                payload={"message_ids": msg_ids},
            )
        )

    async def agents_with_pending_messages(self, run_id: str) -> list[str]:
        """List specialist agent IDs with unconsumed messages in this run."""
        return await self._db.list_agents_with_pending_messages(run_id)

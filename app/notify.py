"""Outbound side-effects interface (Telegram in prod, recorder in tests/simulation)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class Notifier(Protocol):
    async def send_customer(self, customer_id: int, text: str) -> bool: ...
    async def incident_changed(self, incident_id: int) -> None: ...
    async def escalation_created(self, escalation_id: int) -> None: ...
    async def dev_ticket(self, bug_ticket_id: int) -> None: ...


@dataclass
class RecordingNotifier:
    """Collects everything instead of sending it."""

    customer_messages: list[tuple[int, str]] = field(default_factory=list)
    incidents_changed: list[int] = field(default_factory=list)
    escalations: list[int] = field(default_factory=list)
    dev_tickets: list[int] = field(default_factory=list)

    async def send_customer(self, customer_id: int, text: str) -> bool:
        self.customer_messages.append((customer_id, text))
        return True

    async def incident_changed(self, incident_id: int) -> None:
        self.incidents_changed.append(incident_id)

    async def escalation_created(self, escalation_id: int) -> None:
        self.escalations.append(escalation_id)

    async def dev_ticket(self, bug_ticket_id: int) -> None:
        self.dev_tickets.append(bug_ticket_id)

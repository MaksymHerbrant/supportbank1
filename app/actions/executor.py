"""Executes operator-approved plans only. Idempotent and fully audited."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.agent import policy
from app.agent.policy import ActionContext, Level, PolicyViolation
from app.agent.responder import _amount
from app.agent.tools import BankTools
from app.db.repo import Database, local_time, loads, now_ts

log = logging.getLogger(__name__)


@dataclass
class ExecutionReport:
    plan_id: int
    status: str
    already_processed: bool = False
    done: int = 0
    failed: int = 0
    refunded_total: float = 0.0
    notified: int = 0
    sample_message: str | None = None
    errors: list[str] = field(default_factory=list)


def incident_problem(db: Database, incident_id: int) -> str:
    """Short human label of the incident's problem (used in customer messages)."""
    inc = db.q1("SELECT category, signature_json FROM incidents WHERE id=?", (incident_id,))
    if inc is None:
        return "сервісом"
    sig = loads(inc["signature_json"], {})
    if inc["category"] == "double_charge":
        name = db.scalar("SELECT name FROM merchants WHERE id=?", (sig.get("merchant_id"),)) or "мерчанта"
        return f"подвійними списаннями в {name}"
    if inc["category"] == "app_payment_error":
        return f"оплатами у застосунку версії {sig.get('app_version')}"
    if inc["category"] == "atm_capture":
        return f"банкоматом {sig.get('atm_id')}"
    if inc["category"] == "phishing":
        return f"шахрайськими SMS з посиланням {sig.get('scam_domain')}"
    return "сервісом"


def proactive_message(category: str, problem: str, name: str, actions: dict[str, list]) -> str:
    """Factual message to an affected customer, built after the actions really happened."""
    first = name.split()[0] if name else ""
    hello = f"{first}, вітаємо! " if first else "Вітаємо! "
    if category == "double_charge":
        total = sum(actions.get("refund_transaction", []))
        return (f"{hello}Ми виявили проблему з {problem}, яка торкнулася і вас. Дубль списання {_amount(total)} грн "
                f"уже повернуто, кошти надійдуть протягом 1–3 робочих днів. Вам нічого робити не потрібно.")
    if category == "phishing":
        return (f"{hello}Ми зафіксували шахрайську операцію після переходу за фішинговим посиланням. Вашу картку "
                f"заблоковано, по операції відкрито диспут. Ніколи не вводьте дані картки за посиланнями з SMS.")
    if category == "atm_capture":
        return (f"{hello}Банкомат затримав вашу картку через технічну несправність. Ми заблокували її для безпеки "
                f"та замовили перевипуск — нова картка буде готова протягом 5 робочих днів.")
    if category == "app_payment_error":
        return (f"{hello}Бачимо, що у вас не проходили оплати через {problem}. Ми вже виправляємо помилку. "
                f"Поки що можна оплатити через Apple Pay / Google Pay або на сайті demobank.ua.")
    return f"{hello}Ми виявили проблему з {problem} і вже її вирішили."


class Executor:
    def __init__(self, db: Database, tools: BankTools) -> None:
        self.db = db
        self.tools = tools

    async def approve_plan(self, plan_id: int, approved_by: str) -> ExecutionReport:
        """Execute a pending plan exactly once. A second click returns the existing result."""
        cur = self.db.ex(
            "UPDATE plans SET status='executing', decided_by=?, decided_at=? WHERE id=? AND status='pending'",
            (approved_by, now_ts(), plan_id),
        )
        if cur.rowcount == 0:
            status = self.db.scalar("SELECT status FROM plans WHERE id=?", (plan_id,)) or "missing"
            return ExecutionReport(plan_id, status, already_processed=True)
        incident_id = int(self.db.scalar("SELECT incident_id FROM plans WHERE id=?", (plan_id,)))
        category = self.db.scalar("SELECT category FROM incidents WHERE id=?", (incident_id,))
        problem = incident_problem(self.db, incident_id)
        ctx = ActionContext(level=Level.INVESTIGATE, approved_by=approved_by, incident_id=incident_id)
        report = ExecutionReport(plan_id, "executed")
        items = [dict(r) for r in self.db.q(
            "SELECT * FROM plan_items WHERE plan_id=? AND status='pending' ORDER BY customer_id, "
            "CASE action WHEN 'notify_customer' THEN 1 ELSE 0 END, id", (plan_id,))]
        done_by_customer: dict[int, dict[str, list]] = {}
        for item in items:
            params = loads(item["params_json"], {})
            evidence = loads(item["evidence_json"], [])
            reason = f"incident #{incident_id}: {item['reason']}"
            try:
                if item["action"] == "notify_customer":
                    acts = done_by_customer.get(item["customer_id"], {})
                    name = self.db.scalar("SELECT full_name FROM customers WHERE id=?", (item["customer_id"],)) or ""
                    text = proactive_message(category, problem, name, acts)
                    refund_done = bool(acts.get("refund_transaction"))
                    if not policy.is_reply_safe(text, refund_executed=refund_done or category != "double_charge"):
                        raise PolicyViolation("notification would promise an unexecuted refund")
                    await self.tools.notify_customer(ctx, item["customer_id"], text, reason,
                                                     idempotency_key=f"plan_item:{item['id']}", evidence=evidence)
                    report.notified += 1
                    report.sample_message = report.sample_message or text
                else:
                    result = self._run(item["action"], ctx, params, reason, evidence, item["id"])
                    if item["action"] == "refund_transaction" and result is not None:
                        report.refunded_total += result
                        done_by_customer.setdefault(item["customer_id"], {}).setdefault(
                            "refund_transaction", []).append(result)
                    else:
                        done_by_customer.setdefault(item["customer_id"], {}).setdefault(item["action"], []).append(1)
                self.db.ex("UPDATE plan_items SET status='done' WHERE id=?", (item["id"],))
                report.done += 1
            except Exception as exc:  # noqa: BLE001 - one failed item must not stop the plan
                log.exception("plan item %s failed", item["id"])
                self.db.ex("UPDATE plan_items SET status='failed' WHERE id=?", (item["id"],))
                report.failed += 1
                report.errors.append(f"item {item['id']} {item['action']}: {exc}")
        self.db.ex("UPDATE plans SET status='executed' WHERE id=?", (plan_id,))
        report.refunded_total = round(report.refunded_total, 2)
        return report

    def _run(self, action: str, ctx: ActionContext, params: dict, reason: str, evidence: list[str],
             item_id: int) -> float | None:
        if action == "refund_transaction":
            tx = self.db.q1("SELECT amount, status FROM transactions WHERE id=?", (params["tx_id"],))
            res = self.tools.refund_transaction(ctx, params["tx_id"], reason, evidence, plan_item_id=item_id)
            return float(tx["amount"]) if res.created else (float(tx["amount"]) if tx["status"] == "refunded" else None)
        if action == "block_card":
            self.tools.block_card(ctx, params["card_id"], reason, evidence)
        elif action == "order_card_reissue":
            self.tools.order_card_reissue(ctx, params["card_id"], reason, evidence)
        elif action == "open_dispute":
            self.tools.open_dispute(ctx, params["tx_id"], reason, evidence)
        else:
            raise PolicyViolation(f"unknown plan action {action}")
        return None

    def reject_plan(self, plan_id: int, rejected_by: str) -> bool:
        cur = self.db.ex("UPDATE plans SET status='rejected', decided_by=?, decided_at=? WHERE id=? AND status='pending'",
                         (rejected_by, now_ts(), plan_id))
        if cur.rowcount:
            self.db.ex("UPDATE plan_items SET status='rejected' WHERE plan_id=? AND status='pending'", (plan_id,))
        return bool(cur.rowcount)

    async def decide_escalation(self, esc_id: int, approved: bool, operator: str) -> str:
        """Approve/reject an escalation's proposed action (e.g. a single refund)."""
        esc = self.db.q1("SELECT * FROM escalations WHERE id=?", (esc_id,))
        if esc is None:
            return "Ескалацію не знайдено."
        new_status = "approved" if approved else "rejected"
        cur = self.db.ex("UPDATE escalations SET status=?, decided_by=? WHERE id=? AND status='open'",
                         (new_status, operator, esc_id))
        if cur.rowcount == 0:
            return f"Вже оброблено (статус: {esc['status']})."
        if not approved:
            return "Відхилено оператором."
        proposed = loads(esc["proposed_json"], None)
        if not proposed:
            return "Взято в роботу оператором."
        ctx = ActionContext(level=Level.INVESTIGATE, approved_by=operator, requester_id=esc["customer_id"])
        if proposed.get("action") == "refund_transaction":
            res = self.tools.refund_transaction(ctx, int(proposed["tx_id"]), f"escalation #{esc_id}",
                                                proposed.get("evidence"))
            amount = self.db.scalar("SELECT amount FROM transactions WHERE id=?", (proposed["tx_id"],))
            if res.created:
                text = (f"Оператор підтвердив повернення: {_amount(amount)} грн повернуто о "
                        f"{local_time(now_ts())}, кошти надійдуть протягом 1–3 робочих днів.")
                await self.tools.notify_customer(ctx, esc["customer_id"], text, f"escalation #{esc_id} approved",
                                                 idempotency_key=f"esc_notify:{esc_id}")
                return f"Повернено {_amount(amount)} грн, клієнта повідомлено."
            return "Повернення вже було виконано раніше (ідемпотентність)."
        if proposed.get("action") == "unblock_card":
            self.tools.unblock_card(ctx, int(proposed["card_id"]), f"escalation #{esc_id}")
            return "Картку розблоковано."
        return "Підтверджено."

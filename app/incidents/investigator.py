"""Root-cause investigation: deterministic evidence + an LLM tool-using loop for the hypothesis."""
from __future__ import annotations

import json
import logging
import re
from collections import Counter

from pydantic import BaseModel, Field

from app.agent import policy
from app.agent.tools import INVESTIGATOR_TOOLS, BankTools
from app.agent.triage import Signature
from app.db.repo import Database, local_time
from app.incidents.impact import Impact
from app.llm import LLMClient, Message, ToolSpec, _schema_for

log = logging.getLogger(__name__)
MAX_STEPS = 4


class Hypothesis(BaseModel):
    hypothesis: str = Field(..., max_length=500)
    evidence: list[str] = Field(default_factory=list, max_length=10)
    suspected_component: str = Field("unknown", max_length=80)


SYSTEM_PROMPT = """You are a bank incident investigator. Use the read-only tools to inspect transactions,
app events and ATM events, then call submit_hypothesis with: a concise root-cause hypothesis (Ukrainian),
concrete evidence lines that cite real ids you saw (e.g. "tx #123", "app_event #45"), and the suspected component.
Do not speculate beyond the data."""


def baseline(db: Database, sig: Signature, impact: Impact) -> Hypothesis:
    """Rule-based hypothesis from the impact evidence; always available."""
    ev: list[str] = []
    for m in impact.affected[:5]:
        ev.extend(m.evidence[:1])
    n = len(impact.affected)
    if sig.category == "double_charge":
        merchant = db.scalar("SELECT name FROM merchants WHERE id=?", (sig.merchant_id,)) or sig.merchant_id
        gaps = [int(g) for m in impact.affected for e in m.evidence for g in re.findall(r"інтервал (\d+) с", e)]
        rng = f"{min(gaps)}–{max(gaps)} с" if gaps else "<2 хв"
        text = (f"Повторна авторизація платежів у {merchant}: у {n} клієнтів однакова сума списана двічі з "
                f"інтервалом {rng}. Ймовірно, термінал/платіжний шлюз мерчанта повторює запит без ідемпотентного "
                f"ключа після таймауту.")
        comp = f"acquiring/{sig.merchant_id}"
    elif sig.category == "app_payment_error":
        details = Counter(e.split("«")[1].split("»")[0] for m in impact.affected for e in m.evidence if "«" in e)
        top = details.most_common(1)[0][0] if details else "помилка оплати"
        text = (f"Регресія у мобільному застосунку v{sig.app_version}: {n} клієнтів отримують "
                f"{sig.error_code or 'помилку'} при оплаті («{top}»). На попередній версії помилок немає.")
        comp = f"mobile-app/{sig.app_version}/payments"
    elif sig.category == "atm_capture":
        text = f"Несправність картридера банкомата {sig.atm_id}: {n} карток захоплено з помилкою таймауту."
        comp = f"atm/{sig.atm_id}/card-reader"
    elif sig.category == "phishing":
        text = (f"Фішингова SMS-кампанія з доменом {sig.scam_domain}: {n} клієнтів після переходу мають "
                f"шахрайські списання на мерчант-одержувач шахраїв.")
        comp = "fraud/phishing"
    else:
        text, comp = "Недостатньо даних для гіпотези.", "unknown"
    return Hypothesis(hypothesis=text, evidence=ev, suspected_component=comp)


class Investigator:
    def __init__(self, db: Database, llm: LLMClient, tools: BankTools) -> None:
        self.db = db
        self.llm = llm
        self.tools = tools

    def _run_tool(self, name: str, args: dict) -> str:
        if name not in policy.READ_TOOLS:
            return json.dumps({"error": f"tool {name} not allowed"})
        allowed = {"merchant_id", "customer_id", "since_hours", "limit", "app_version", "event_type", "atm_id"}
        clean = {k: v for k, v in args.items() if k in allowed}
        clean["limit"] = min(int(clean.get("limit", 30) or 30), 50)
        try:
            data = getattr(self.tools, name)(**clean)
        except (TypeError, ValueError) as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps(data, ensure_ascii=False, default=str)[:6000]

    async def investigate(self, sig: Signature, impact: Impact) -> Hypothesis:
        base = baseline(self.db, sig, impact)
        if not self.llm.available:
            return base
        submit = ToolSpec("submit_hypothesis", "Submit the final root-cause hypothesis.", _schema_for(Hypothesis))
        messages = [
            Message("system", SYSTEM_PROMPT),
            Message("user", f"Incident signature: {sig.model_dump_json()}\nConfirmed affected: {len(impact.affected)}, "
                            f"at_risk: {len(impact.at_risk)}\nInitial evidence:\n- " + "\n- ".join(base.evidence)),
        ]
        seen_ids: set[str] = set(re.findall(r"#(\d+)", " ".join(base.evidence)))
        for _ in range(MAX_STEPS):
            resp = await self.llm.chat(messages, tools=INVESTIGATOR_TOOLS + [submit])
            if resp is None:
                break
            final = next((c for c in resp.tool_calls if c.name == "submit_hypothesis"), None)
            if final:
                try:
                    hyp = Hypothesis.model_validate(final.arguments)
                except ValueError:
                    break
                # Keep only evidence that cites ids the tools actually returned (anti-hallucination).
                grounded = [e for e in hyp.evidence if set(re.findall(r"#(\d+)", e)) & seen_ids]
                return Hypothesis(hypothesis=hyp.hypothesis, suspected_component=hyp.suspected_component,
                                  evidence=(base.evidence + grounded)[:10])
            if not resp.tool_calls:
                break
            messages.append(Message("assistant", resp.text or "(tool calls)"))
            for call in resp.tool_calls[:3]:
                output = self._run_tool(call.name, call.arguments)
                seen_ids |= set(re.findall(r'"id": (\d+)', output))
                messages.append(Message("tool", f"{call.name} -> {output}", name=call.name))
        return base


def describe_timeline(db: Database, incident_id: int) -> str:
    row = db.q1("SELECT created_at FROM incidents WHERE id=?", (incident_id,))
    return local_time(row["created_at"]) if row else "—"

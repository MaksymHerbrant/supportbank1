"""Customer-facing replies built only from verified facts.

The LLM may phrase the reply, but only from the facts dict; every reply is
checked by ``policy`` and replaced with a deterministic template if it promises
something that has not been approved/executed.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from app.agent import policy
from app.db.repo import dumps, money
from app.llm import LLMClient

log = logging.getLogger(__name__)


@dataclass
class ReplyFacts:
    branch: str
    language: str = "uk"
    problem: str | None = None          # e.g. "подвійними списаннями на АЗС Fuel+"
    amount: float | None = None
    refund_executed: bool = False
    refund_time: str | None = None
    actions_done: list[str] = field(default_factory=list)
    details: str | None = None
    customer_name: str | None = None


def _amount(a: float | None) -> str:
    return money(a)


TEMPLATES_UK = {
    "linked_done": "Ми знаємо про проблему з {problem}. Ваш випадок підтверджено: кошти {amount} грн повернуто о "
                   "{refund_time}, вони надійдуть на картку протягом 1–3 робочих днів. Дякуємо за терпіння!",
    "linked_done_noamount": "Ми знаємо про проблему з {problem}. Для вашого випадку вже виконано: {actions}. "
                            "Дякуємо за терпіння!",
    "linked_pending": "Ми знаємо про проблему з {problem} і вже її вирішуємо. Ваш випадок підтверджено, "
                      "ми повідомимо вас, щойно кошти буде повернуто.",
    "linked_pending_noamount": "Ми знаємо про проблему з {problem} і вже її вирішуємо. Ваш випадок підтверджено — "
                               "ми напишемо вам, щойно все буде виконано.",
    "linked_unconfirmed": "Ми знаємо про проблему з {problem}, але у вашому випадку поки не бачимо її ознак ({details}). "
                          "Уточніть, будь ласка, дату і суму операції або надішліть скріншот.",
    "linked_unconfirmed_escalated": "Ми знаємо про проблему з {problem}, але у вашому випадку не бачимо її ознак "
                                    "({details}). Я передав ваше звернення оператору — він перевірить вручну.",
    "linked_resolved": "Проблему з {problem} вже виправлено. Статус вашого випадку: {details}.",
    "own_duplicate": "Бачу у вашій виписці подвійне списання {amount} грн ({details}). Я відкрив диспут — "
                     "повернення дубля погоджує оператор, ми повідомимо вас, щойно кошти буде повернуто.",
    "own_no_duplicate": "Я перевірив ваші операції ({details}) і не бачу подвійного списання. Уточніть, будь ласка, "
                        "дату і суму або надішліть скріншот з застосунку.",
    "screenshot_mismatch": "Дякую за скріншот. У наших даних немає подвійного списання ({details}), тому я передав "
                           "звернення оператору для ручної перевірки.",
    "refund_unfounded": "Я перевірив ваші останні операції і не бачу підстав для повернення коштів ({details}). "
                        "Звернення передано оператору — він зв'яжеться з вами.",
    "card_blocked_reissue": "Вашу картку {details} заблоковано, щоб ніхто не міг нею скористатися, і замовлено "
                            "перевипуск. Нова картка буде готова протягом 5 робочих днів.",
    "social_engineering": "Я заблокував вашу картку {details} з міркувань безпеки. Банк ніколи не запитує коди з SMS "
                          "чи CVV. Оператор уже отримав ваше звернення і зв'яжеться з вами найближчим часом.",
    "phishing_victim": "Це фішинг. Я заблокував вашу картку {details}, щоб зупинити подальші списання. Оператор "
                       "перевірить підозрілі операції та відкриє диспут.",
    "phishing_warning": "Дякую, що повідомили! Це шахрайське повідомлення ({details}) — не переходьте за посиланням "
                        "і не вводьте дані картки. Якщо ви вже ввели дані, напишіть нам — ми одразу заблокуємо картку.",
    "app_error": "Бачу помилку оплати {details}. Ми передали її розробникам. Поки що можна оплатити через "
                 "Apple Pay / Google Pay або на сайті demobank.ua.",
    "app_no_error": "Я не бачу помилок оплати у вашому застосунку за останню добу. Опишіть, будь ласка, що саме "
                    "відбувається, або надішліть запис екрана.",
    "escalated": "Дякую за звернення. Я передав його оператору — він зв'яжеться з вами найближчим часом.",
    "faq": "{details}",
    "resolved_final": "Проблему з {problem} вирішено. {details}",
}
TEMPLATES_EN = {
    "linked_done": "We are aware of the issue with {problem}. Your case is confirmed: {amount} UAH was refunded at "
                   "{refund_time} and will arrive within 1–3 business days.",
    "linked_done_noamount": "We are aware of the issue with {problem}. Already done for you: {actions}.",
    "linked_pending": "We are aware of the issue with {problem} and are already fixing it. Your case is confirmed; "
                      "we will notify you as soon as the money is refunded.",
    "linked_pending_noamount": "We are aware of the issue with {problem} and are fixing it. Your case is confirmed — "
                               "we will message you once everything is done.",
    "linked_unconfirmed": "We are aware of the issue with {problem}, but we don't see signs of it in your case "
                          "({details}). Please share the date and amount or a screenshot.",
    "linked_unconfirmed_escalated": "We are aware of the issue with {problem}, but we don't see signs of it in your "
                                    "case ({details}). I've passed your request to an operator for a manual check.",
    "linked_resolved": "The issue with {problem} has been fixed. Your case status: {details}.",
    "own_duplicate": "I can see a double charge of {amount} UAH ({details}). I opened a dispute; an operator approves "
                     "the refund and we will notify you as soon as it is refunded.",
    "own_no_duplicate": "I checked your transactions ({details}) and see no double charge. Please share the date and "
                        "amount or a screenshot.",
    "screenshot_mismatch": "Thanks for the screenshot. Our records show no double charge ({details}), so an operator "
                           "will review it manually.",
    "refund_unfounded": "I checked your recent transactions and found no grounds for a refund ({details}). An operator "
                        "will contact you.",
    "card_blocked_reissue": "Your card {details} is blocked so nobody can use it, and a replacement is ordered "
                            "(ready within 5 business days).",
    "social_engineering": "I blocked your card {details} for your safety. The bank never asks for SMS codes or CVV. "
                          "An operator will contact you shortly.",
    "phishing_victim": "This is phishing. I blocked your card {details} to stop further charges. An operator will "
                       "review the transactions and open a dispute.",
    "phishing_warning": "Thanks for reporting! This message is fraudulent ({details}) — don't open the link or enter "
                        "card details. If you already did, tell us and we'll block the card immediately.",
    "app_error": "I can see the payment error {details}. It's been sent to our developers. Meanwhile you can pay via "
                 "Apple Pay / Google Pay or on demobank.ua.",
    "app_no_error": "I don't see payment errors in your app over the last day. Please describe what happens or send "
                    "a screen recording.",
    "escalated": "Thank you. I've passed your request to an operator who will contact you shortly.",
    "faq": "{details}",
    "resolved_final": "The issue with {problem} is resolved. {details}",
}


def render_template(facts: ReplyFacts) -> str:
    templates = TEMPLATES_EN if facts.language == "en" else TEMPLATES_UK
    tpl = templates.get(facts.branch, templates["escalated"])
    return tpl.format(
        problem=facts.problem or "", amount=_amount(facts.amount), refund_time=facts.refund_time or "",
        details=facts.details or "", actions=", ".join(facts.actions_done) or "—",
    ).strip()


SYSTEM_PROMPT = """You write short, warm customer-support replies for DemoBank in {language}.
Use ONLY the facts given in JSON. Do not add amounts, times, or promises not present in the facts.
If refund_executed is false, NEVER say money was or will be refunded; at most say we will notify the customer.
Keep the meaning of the draft, max 3 sentences, no greeting by name unless given."""


class Responder:
    def __init__(self, llm: LLMClient, use_llm: bool = True) -> None:
        self.llm = llm
        self.use_llm = use_llm

    async def compose(self, facts: ReplyFacts) -> str:
        draft = render_template(facts)
        if facts.branch == "faq" or not (self.use_llm and self.llm.available):
            return policy.mask_pan(draft) or draft
        text = await self.llm.text(
            SYSTEM_PROMPT.format(language="Ukrainian" if facts.language == "uk" else "English"),
            f"Facts: {dumps(asdict(facts))}\nDraft reply: {draft}",
        )
        if not text:
            return draft
        problems = policy.reply_violations(text, refund_executed=facts.refund_executed)
        if problems:
            log.warning("LLM reply rejected by policy: %s", problems)
            return draft
        return policy.mask_pan(text) or draft

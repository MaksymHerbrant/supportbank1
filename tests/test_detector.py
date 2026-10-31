"""Incident detection: thresholds and paraphrase grouping."""
from __future__ import annotations

from tests.conftest import make_container, run_double_charge, ConceptEmbedder

PARAPHRASES = [
    (92, "на заправці двічі зняли гроші"),
    (93, "Бензин оплатив, а гроші списали два рази"),
    (94, "подвійне списання коштів на АЗС сьогодні"),
]


async def test_incident_created_at_threshold(container) -> None:  # noqa: ANN001
    incident_id = await run_double_charge(container)
    inc = container.db.q1("SELECT * FROM incidents WHERE id=?", (incident_id,))
    assert inc["category"] == "double_charge"
    assert '"merchant_id":"fuelplus"' in inc["signature_json"].replace(" ", "")
    assert container.db.scalar("SELECT COUNT(*) FROM tickets WHERE incident_id=?", (incident_id,)) == 3


async def test_no_incident_below_threshold(container) -> None:  # noqa: ANN001
    for cid, text in [(1, "Мене двічі списали на АЗС Fuel+"), (2, "Fuel+ зняли гроші два рази")]:
        reply = await container.pipeline.handle(cid, text)
        assert reply.new_incident_id is None
    assert container.db.scalar("SELECT COUNT(*) FROM incidents") == 0


async def test_same_customer_twice_does_not_count_twice(container) -> None:  # noqa: ANN001
    for text in ["Двічі списали на Fuel+", "Fuel+ знову двічі", "Fuel+ дубль списання"]:
        await container.pipeline.handle(1, text)
    assert container.db.scalar("SELECT COUNT(*) FROM incidents") == 0


async def test_paraphrased_complaints_group_via_embeddings(db, notifier) -> None:  # noqa: ANN001
    c = make_container(db, notifier, embedder=ConceptEmbedder())
    replies = [await c.pipeline.handle(cid, text) for cid, text in PARAPHRASES]
    # No merchant in the texts and no duplicates in these customers' data -> no exact signature.
    assert all(r.triage.signature.key() is None for r in replies)
    assert replies[-1].new_incident_id is not None


async def test_paraphrases_without_embeddings_do_not_group(db, notifier) -> None:  # noqa: ANN001
    c = make_container(db, notifier)  # NullEmbedder -> exact signatures only
    replies = [await c.pipeline.handle(cid, text) for cid, text in PARAPHRASES]
    assert all(r.new_incident_id is None for r in replies)


async def test_different_merchants_are_not_grouped(db, notifier) -> None:  # noqa: ANN001
    c = make_container(db, notifier, embedder=ConceptEmbedder())
    await c.pipeline.handle(92, "двічі списали на АЗС Fuel+")
    await c.pipeline.handle(93, "двічі списали в Кава Зерно")
    reply = await c.pipeline.handle(94, "двічі списали в Маркет Колосок")
    assert reply.new_incident_id is None

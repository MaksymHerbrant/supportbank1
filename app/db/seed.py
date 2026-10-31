"""Generate fictional DemoBank data with the demo scenarios baked in.

Run: ``python -m app.db.seed`` (recreates the database).

Scenario map (customer ids are stable because the RNG is seeded):
  A  Fuel+ double charge: customers 1-15 have duplicate transactions today
     (1-4 complain, 5 complains late); 16-40 have Fuel+ recurring payments in
     the next days (at_risk); 41 paid Fuel+ once today (no duplicate).
  B  Phishing: 42-47 opened the fake link; 42-45 have fraud transactions.
  C  ATM-0417 captured cards of 48-52; 53-58 only withdrew there.
  D  App 5.4.0: 59-66 have failed payments, 67-86 run 5.4.0 without errors.
  E  Risky: 87 VIP demand, 88 fake screenshot, 89 social engineering,
     90 prompt injection; 91 FAQ persona.
"""
from __future__ import annotations

import logging
import random
from datetime import timedelta

from app.config import get_settings
from app.db.repo import Database, ts, utcnow

log = logging.getLogger(__name__)

FIRST_NAMES = [
    "Олена", "Іван", "Марія", "Андрій", "Наталія", "Олег", "Софія", "Дмитро", "Ірина", "Тарас",
    "Юлія", "Максим", "Катерина", "Богдан", "Оксана", "Сергій", "Анна", "Василь", "Дарина", "Роман",
]
LAST_NAMES = [
    "Коваленко", "Шевчук", "Бондар", "Ткаченко", "Мельник", "Кравчук", "Олійник", "Литвин",
    "Савченко", "Руденко", "Мороз", "Поліщук", "Гончар", "Левченко", "Марченко", "Лисенко",
]

MERCHANTS = [
    # id, name, aliases, category, is_scam, scam_domain
    ("fuelplus", "АЗС Fuel+", "fuel+,fuel plus,фуел,фьюел,азс fuel,fuelplus", "fuel", 0, None),
    ("kolosok", "Маркет Колосок", "колосок,kolosok", "grocery", 0, None),
    ("zerno", "Кава Зерно", "зерно,кава зерно,zerno", "cafe", 0, None),
    ("shvydko", "Таксі Швидко", "швидко,таксі,shvydko", "transport", 0, None),
    ("zdorovia", "Аптека Здоровʼя", "аптека,здоровʼя,zdorovia", "pharmacy", 0, None),
    ("zoria", "Кіно Зоря", "кіно,зоря,zoria", "entertainment", 0, None),
    ("khvylia", "Стрімінг Хвиля", "хвиля,khvylia,стрімінг", "subscription", 0, None),
    ("scam_secure", "SecurePay-Verify", "securepay,demobank-secure", "p2p", 1, "demobank-secure.top"),
]
REGULAR_MERCHANTS = ["kolosok", "zerno", "shvydko", "zdorovia", "zoria", "khvylia"]

ATMS = [
    ("ATM-0417", "Київ, вул. Хрещатик-Демо, 17"),
    ("ATM-0233", "Київ, просп. Перемоги-Демо, 33"),
    ("ATM-0101", "Львів, пл. Ринок-Демо, 1"),
]

DEMO_TAGS = {
    1: "fuel_complainer_1", 2: "fuel_complainer_2", 3: "fuel_complainer_3", 4: "fuel_complainer_4",
    5: "fuel_late_affected", 16: "fuel_promoted", 41: "fuel_late_clean",
    42: "phish_reporter_1", 43: "phish_reporter_2", 46: "phish_reporter_3",
    48: "atm_reporter_1", 49: "atm_reporter_2", 50: "atm_reporter_3",
    59: "app_reporter_1", 60: "app_reporter_2", 61: "app_reporter_3",
    87: "vip_demand", 88: "fake_screenshot", 89: "social_eng", 90: "injection", 91: "faq_user",
}
NUM_CUSTOMERS = 100
APP_BUG_VERSION = "5.4.0"
APP_OK_VERSION = "5.3.2"


def seed(db: Database, rng_seed: int = 42) -> None:
    """Recreate schema and fill it with deterministic demo data."""
    rng = random.Random(rng_seed)
    now = utcnow()
    db.init_schema()
    db.ex("BEGIN")
    try:
        _seed_all(db, rng, now)
        db.ex("COMMIT")
    except Exception:
        db.ex("ROLLBACK")
        raise


def _seed_all(db: Database, rng: random.Random, now) -> None:  # noqa: ANN001
    for m in MERCHANTS:
        db.insert("merchants", dict(zip(("id", "name", "aliases", "category", "is_scam", "scam_domain"), m)))
    for atm_id, address in ATMS:
        db.insert("atms", {"id": atm_id, "address": address})

    card_of: dict[int, int] = {}
    for cid in range(1, NUM_CUSTOMERS + 1):
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        db.insert("customers", {
            "id": cid,
            "full_name": name,
            "phone": f"+38050{rng.randint(1000000, 9999999)}",
            "language": "uk",
            "segment": "vip" if cid == 87 else "standard",
            "demo_tag": DEMO_TAGS.get(cid),
            "created_at": ts(now - timedelta(days=rng.randint(100, 2000))),
        })
        card_of[cid] = db.insert("cards", {"customer_id": cid, "last4": f"{rng.randint(0, 9999):04d}"})
        if rng.random() < 0.3:
            db.insert("cards", {"customer_id": cid, "last4": f"{rng.randint(0, 9999):04d}", "product": "credit"})

    def tx(cid: int, merchant: str, amount: float, at, **extra) -> int:  # noqa: ANN001
        return db.insert("transactions", {
            "customer_id": cid, "card_id": card_of[cid], "merchant_id": merchant,
            "amount": round(amount, 2), "created_at": ts(at), **extra,
        })

    # Background activity: older than 3 hours, never at Fuel+ today.
    for cid in range(1, NUM_CUSTOMERS + 1):
        for _ in range(rng.randint(6, 10)):
            at = now - timedelta(hours=rng.uniform(3, 24 * 14))
            tx(cid, rng.choice(REGULAR_MERCHANTS), rng.uniform(40, 1500), at)
        if rng.random() < 0.5:
            tx(cid, "fuelplus", rng.uniform(300, 1800), now - timedelta(days=rng.uniform(2, 14)))

    # A) Fuel+ double charges today for customers 1-15.
    for cid in range(1, 16):
        amount = 450.0 if cid == 5 else float(rng.choice([380, 520, 640, 710, 900, 1150, 1240]))
        first = now - timedelta(minutes=rng.uniform(25, 170))
        tx(cid, "fuelplus", amount, first)
        tx(cid, "fuelplus", amount, first + timedelta(seconds=rng.randint(25, 110)))
    # Single (correct) Fuel+ payments today: no duplicates.
    for cid in [41] + rng.sample(range(42, 100), 18):
        tx(cid, "fuelplus", rng.uniform(300, 1500), now - timedelta(minutes=rng.uniform(20, 170)))
    # Recurring Fuel+ payments in the next days -> at_risk (16-40); one affected also has one.
    for cid in list(range(16, 41)) + [3]:
        db.insert("recurring_payments", {
            "customer_id": cid, "card_id": card_of[cid], "merchant_id": "fuelplus",
            "amount": float(rng.choice([500, 750, 1000])),
            "next_date": (now + timedelta(days=rng.randint(1, 5))).strftime("%Y-%m-%d"),
        })
    db.insert("recurring_payments", {  # far in the future -> not at risk
        "customer_id": 95, "card_id": card_of[95], "merchant_id": "fuelplus",
        "amount": 500.0, "next_date": (now + timedelta(days=40)).strftime("%Y-%m-%d"),
    })

    # B) Phishing: 42-47 opened the link; 42-45 lost money to the scam merchant.
    for cid in range(42, 48):
        opened = now - timedelta(minutes=rng.uniform(60, 200))
        db.insert("app_events", {
            "customer_id": cid, "app_version": APP_OK_VERSION, "event_type": "phishing_link_open",
            "details": "referrer=sms domain=demobank-secure.top", "created_at": ts(opened),
        })
        if cid <= 45:
            tx(cid, "scam_secure", float(rng.choice([2500, 4999, 7200, 9000])),
               opened + timedelta(minutes=rng.uniform(3, 15)), is_fraud=1)

    # C) ATM-0417 captured cards of 48-52 today; 53-58 only withdrew there.
    for cid in range(48, 53):
        db.insert("atm_events", {
            "atm_id": "ATM-0417", "card_id": card_of[cid], "customer_id": cid,
            "event_type": "card_captured", "details": "card reader timeout E17, card retained",
            "created_at": ts(now - timedelta(minutes=rng.uniform(15, 180))),
        })
    for cid in range(53, 59):
        db.insert("atm_events", {
            "atm_id": "ATM-0417", "card_id": card_of[cid], "customer_id": cid,
            "event_type": "withdrawal", "details": "ok",
            "created_at": ts(now - timedelta(hours=rng.uniform(4, 72))),
        })
    for cid in rng.sample(range(60, 100), 10):
        db.insert("atm_events", {
            "atm_id": rng.choice(["ATM-0233", "ATM-0101"]), "card_id": card_of[cid], "customer_id": cid,
            "event_type": "withdrawal", "details": "ok",
            "created_at": ts(now - timedelta(hours=rng.uniform(1, 72))),
        })

    # D) App 5.4.0: failed payments for 59-66; 67-86 on 5.4.0 without errors.
    for cid in range(1, NUM_CUSTOMERS + 1):
        if 59 <= cid <= 86:
            version = APP_BUG_VERSION
        else:
            version = APP_OK_VERSION
        db.insert("app_events", {
            "customer_id": cid, "app_version": version, "event_type": "app_open",
            "created_at": ts(now - timedelta(minutes=rng.uniform(30, 300))),
        })
    for cid in range(59, 67):
        for _ in range(rng.randint(1, 3)):
            db.insert("app_events", {
                "customer_id": cid, "app_version": APP_BUG_VERSION, "event_type": "payment_failed",
                "error_code": "E-PAY-502", "details": "3DS callback returned null (PaymentSDK 7.1)",
                "created_at": ts(now - timedelta(minutes=rng.uniform(10, 200))),
            })

    # E) Fake screenshot persona has a single 1200 UAH coffee payment (no duplicate).
    tx(88, "zerno", 1200.0, now - timedelta(hours=5))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    db = Database(settings.db_path)
    seed(db)
    count = db.scalar("SELECT COUNT(*) FROM customers")
    txs = db.scalar("SELECT COUNT(*) FROM transactions")
    log.info("Seeded %s customers, %s transactions into %s", count, txs, settings.db_path)
    db.close()


if __name__ == "__main__":
    main()

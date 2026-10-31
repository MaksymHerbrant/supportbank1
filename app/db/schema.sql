-- DemoBank schema. All data is fictional.
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS decisions;
DROP TABLE IF EXISTS credit_charges;
DROP TABLE IF EXISTS credit_accounts;
DROP TABLE IF EXISTS devices;
DROP TABLE IF EXISTS disputes;
DROP TABLE IF EXISTS refunds;
DROP TABLE IF EXISTS vectors;
DROP TABLE IF EXISTS bug_tickets;
DROP TABLE IF EXISTS escalations;
DROP TABLE IF EXISTS actions;
DROP TABLE IF EXISTS plan_items;
DROP TABLE IF EXISTS plans;
DROP TABLE IF EXISTS incident_members;
DROP TABLE IF EXISTS tickets;
DROP TABLE IF EXISTS incidents;
DROP TABLE IF EXISTS recurring_payments;
DROP TABLE IF EXISTS app_events;
DROP TABLE IF EXISTS atm_events;
DROP TABLE IF EXISTS atms;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS merchants;
DROP TABLE IF EXISTS cards;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    full_name   TEXT NOT NULL,
    phone       TEXT,
    language    TEXT NOT NULL DEFAULT 'uk',
    segment     TEXT NOT NULL DEFAULT 'standard',
    tg_chat_id  INTEGER,
    demo_tag    TEXT UNIQUE,
    kyc_complete INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL
);

-- Only last4 is ever stored: full card numbers never enter the system.
CREATE TABLE cards (
    id              INTEGER PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(id),
    last4           TEXT NOT NULL CHECK (length(last4) = 4),
    product         TEXT NOT NULL DEFAULT 'debit',
    status          TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'blocked')),
    reissue_ordered INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE merchants (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    aliases     TEXT NOT NULL DEFAULT '',
    category    TEXT NOT NULL,
    is_scam     INTEGER NOT NULL DEFAULT 0,
    scam_domain TEXT
);

CREATE TABLE transactions (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    card_id     INTEGER NOT NULL REFERENCES cards(id),
    merchant_id TEXT NOT NULL REFERENCES merchants(id),
    amount      REAL NOT NULL,
    currency    TEXT NOT NULL DEFAULT 'UAH',
    status      TEXT NOT NULL DEFAULT 'settled'
                CHECK (status IN ('settled', 'failed', 'refunded', 'disputed')),
    is_fraud    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);
CREATE INDEX ix_tx_merchant ON transactions(merchant_id, created_at);
CREATE INDEX ix_tx_customer ON transactions(customer_id, created_at);

CREATE TABLE atms (
    id      TEXT PRIMARY KEY,
    address TEXT NOT NULL,
    status  TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok', 'out_of_service'))
);

CREATE TABLE atm_events (
    id          INTEGER PRIMARY KEY,
    atm_id      TEXT NOT NULL REFERENCES atms(id),
    card_id     INTEGER REFERENCES cards(id),
    customer_id INTEGER REFERENCES customers(id),
    event_type  TEXT NOT NULL CHECK (event_type IN ('withdrawal', 'card_captured', 'error')),
    details     TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE app_events (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    app_version TEXT NOT NULL,
    event_type  TEXT NOT NULL
                CHECK (event_type IN ('app_open', 'payment_ok', 'payment_failed', 'phishing_link_open')),
    error_code  TEXT,
    details     TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE recurring_payments (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    card_id     INTEGER NOT NULL REFERENCES cards(id),
    merchant_id TEXT NOT NULL REFERENCES merchants(id),
    amount      REAL NOT NULL,
    next_date   TEXT NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE incidents (
    id              INTEGER PRIMARY KEY,
    category        TEXT NOT NULL,
    signature_json  TEXT NOT NULL,
    summary         TEXT NOT NULL,
    hypothesis      TEXT,
    evidence_json   TEXT NOT NULL DEFAULT '[]',
    severity        TEXT NOT NULL DEFAULT 'medium',
    status          TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'resolved')),
    operator_msg_id INTEGER,
    created_at      TEXT NOT NULL,
    resolved_at     TEXT,
    resolved_by     TEXT
);

CREATE TABLE tickets (
    id              INTEGER PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(id),
    text            TEXT NOT NULL,
    media_json      TEXT,
    intent          TEXT,
    category        TEXT,
    level           INTEGER,
    confidence      REAL,
    risk_flags_json TEXT NOT NULL DEFAULT '[]',
    signature_json  TEXT NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'new'
                    CHECK (status IN ('new', 'auto_resolved', 'linked', 'escalated', 'waiting_customer', 'resolved')),
    incident_id     INTEGER REFERENCES incidents(id),
    link_score      REAL,
    link_method     TEXT,
    link_branch     TEXT,
    reply           TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX ix_tickets_created ON tickets(created_at);

CREATE TABLE incident_members (
    incident_id   INTEGER NOT NULL REFERENCES incidents(id),
    customer_id   INTEGER NOT NULL REFERENCES customers(id),
    role          TEXT NOT NULL CHECK (role IN ('affected', 'at_risk')),
    reason        TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    amount        REAL,
    ref_id        TEXT,
    card_id       INTEGER,
    first_seen_at TEXT NOT NULL,
    promoted_at   TEXT,
    PRIMARY KEY (incident_id, customer_id)
);

CREATE TABLE plans (
    id          INTEGER PRIMARY KEY,
    incident_id INTEGER NOT NULL REFERENCES incidents(id),
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'executing', 'executed', 'rejected')),
    created_at  TEXT NOT NULL,
    decided_by  TEXT,
    decided_at  TEXT
);

CREATE TABLE plan_items (
    id            INTEGER PRIMARY KEY,
    plan_id       INTEGER NOT NULL REFERENCES plans(id),
    incident_id   INTEGER NOT NULL REFERENCES incidents(id),
    customer_id   INTEGER REFERENCES customers(id),
    action        TEXT NOT NULL,
    ref           TEXT NOT NULL DEFAULT '',
    params_json   TEXT NOT NULL DEFAULT '{}',
    amount        REAL,
    reason        TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    status        TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'done', 'failed', 'rejected')),
    UNIQUE (incident_id, customer_id, action, ref)
);

-- Audit log: every action the system performs, with who approved it and why.
CREATE TABLE actions (
    id              INTEGER PRIMARY KEY,
    action_type     TEXT NOT NULL,
    customer_id     INTEGER,
    incident_id     INTEGER,
    ticket_id       INTEGER,
    plan_item_id    INTEGER,
    params_json     TEXT NOT NULL DEFAULT '{}',
    amount          REAL,
    reason          TEXT NOT NULL,
    evidence_json   TEXT NOT NULL DEFAULT '[]',
    actor           TEXT NOT NULL CHECK (actor IN ('agent', 'operator', 'system')),
    approved_by     TEXT,
    status          TEXT NOT NULL DEFAULT 'done' CHECK (status IN ('done', 'failed')),
    result_json     TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at      TEXT NOT NULL
);

CREATE TABLE escalations (
    id              INTEGER PRIMARY KEY,
    ticket_id       INTEGER REFERENCES tickets(id),
    customer_id     INTEGER REFERENCES customers(id),
    reason          TEXT NOT NULL,
    proposed_json   TEXT,
    status          TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'approved', 'rejected', 'closed')),
    operator_msg_id INTEGER,
    decided_by      TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE bug_tickets (
    id          INTEGER PRIMARY KEY,
    incident_id INTEGER REFERENCES incidents(id),
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    severity    TEXT NOT NULL,
    priority    TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE vectors (
    collection    TEXT NOT NULL,
    id            TEXT NOT NULL,
    text          TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    embedding     BLOB,
    PRIMARY KEY (collection, id)
);

-- History used by the deterministic risk profile (app/agent/risk.py).
CREATE TABLE refunds (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    amount      REAL NOT NULL,
    reason      TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('approved', 'rejected')),
    created_at  TEXT NOT NULL
);

CREATE TABLE disputes (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    kind        TEXT NOT NULL DEFAULT 'dispute' CHECK (kind IN ('dispute', 'chargeback')),
    outcome     TEXT NOT NULL CHECK (outcome IN ('open', 'won', 'lost')),
    created_at  TEXT NOT NULL
);

CREATE TABLE devices (
    id            INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(id),
    change_type   TEXT NOT NULL CHECK (change_type IN ('new_device', 'phone_number_changed')),
    details       TEXT,
    changed_at    TEXT NOT NULL
);

CREATE TABLE credit_accounts (
    id                 INTEGER PRIMARY KEY,
    customer_id        INTEGER NOT NULL UNIQUE REFERENCES customers(id),
    card_id            INTEGER REFERENCES cards(id),
    credit_limit       REAL NOT NULL,
    balance            REAL NOT NULL,
    overdue_days       INTEGER NOT NULL DEFAULT 0,
    missed_payments_6m INTEGER NOT NULL DEFAULT 0,
    in_restructuring   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE credit_charges (
    id            INTEGER PRIMARY KEY,
    account_id    INTEGER NOT NULL REFERENCES credit_accounts(id),
    customer_id   INTEGER NOT NULL REFERENCES customers(id),
    kind          TEXT NOT NULL CHECK (kind IN ('fee', 'interest')),
    description   TEXT NOT NULL,
    amount        REAL NOT NULL,
    is_erroneous  INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'charged' CHECK (status IN ('charged', 'refunded')),
    created_at    TEXT NOT NULL
);

-- Every money decision: what the LLM proposed and what policy enforced.
CREATE TABLE decisions (
    id               INTEGER PRIMARY KEY,
    ticket_id        INTEGER REFERENCES tickets(id),
    customer_id      INTEGER NOT NULL,
    kind             TEXT NOT NULL,
    llm_decision     TEXT,
    final_decision   TEXT NOT NULL CHECK (final_decision IN ('auto_approve', 'needs_operator', 'info_only')),
    security_review  INTEGER NOT NULL DEFAULT 0,
    risk_level       TEXT NOT NULL,
    risk_score       INTEGER NOT NULL,
    factors_json     TEXT NOT NULL DEFAULT '[]',
    enforced_json    TEXT NOT NULL DEFAULT '[]',
    reasoning        TEXT,
    operator_note    TEXT,
    action_json      TEXT,
    created_at       TEXT NOT NULL
);

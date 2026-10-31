# Аудит DemoBank Proactive Support (31.10, перед здачею о 17:00)

## 1. Підсумок

Проєкт **готовий до демо**. Головний сценарій працює від першої скарги до закриття інциденту без Telegram
(`scripts/demo_check.py`) і в живому боті з Gemma 4 через Google AI:
- троє скаржаться → інцидент → знайдено 15 affected, з них 12 не писали → 25 at_risk без повідомлень;
- план повернень, підтвердження оператором, привʼязка пізніх скарг за гілками a/b/c/d;
- фінальні повідомлення тим, хто писав.

Автоповернення з профілем ризику і `policy.enforce`, яка може рішення лише посилити, працюють для всіх
персонажів, як і захист від prompt injection. 101 тест проходить. Секретів у репозиторії немає.

Від еталонної ідеї **відсутні**:
- симулятор банку (`/break`, `/fix`, вебекран);
- алертер SEV1/2/3 з повтором до «Беру в роботу»;
- вебдашборд з автооновленням;
- денний бюджет і бюджет інциденту;
- retry-логи для відрізнення дубля від двох легітимних покупок.

Дашборд є лише як Telegram-команди `/dashboard`, `/stats`, `/incidents`.

## 2. Вимоги

| Вимога | Статус | Де |
|---|---|---|
| LLMClient: Ollama `gemma4:e4b` і Google `gemma-4-26b-a4b-it`, function calling → строгий JSON + pydantic, таймаути, повтори, circuit breaker, rate limit | ✅ | [app/llm.py:91](app/llm.py#L91), [:134](app/llm.py#L134), [:259](app/llm.py#L259) |
| triage: intent, level 1–4, confidence, risk_flags, signature | ✅ | [app/agent/triage.py:92](app/agent/triage.py#L92) |
| media: фото → поля, відео → кадри ffmpeg, медіа не зберігаються, last4 | ✅ (ffmpeg на машині не встановлено → бот просить скріншот) | [app/agent/media.py:34](app/agent/media.py#L34), [:61](app/agent/media.py#L61) |
| policy.enforce лише посилює рішення | ✅ (тест на всіх комбінаціях) | [app/agent/decision.py:119](app/agent/decision.py#L119) |
| risk.py: профіль, ваги в конфігу, low/medium/high, без персональних ознак | ✅ ваги в `RISK_WEIGHTS` (код, не .env) | [app/agent/risk.py:16](app/agent/risk.py#L16), [:30](app/agent/risk.py#L30) |
| decision.py: строгий JSON, нейтральне customer_message | ✅ | [app/agent/decision.py:31](app/agent/decision.py#L31) |
| Ліміти 500 грн/транзакція, 1000 грн/клієнт/30 днів, денний бюджет, бюджет інциденту, circuit breaker — усе в .env | ⚠️ було: ліміти константами в коді (2 000/дубль, 500/кредит, goodwill 1 000 раз на 30 днів), денного бюджету й бюджету інциденту немає; circuit breaker — у коді | [app/agent/decision.py:24](app/agent/decision.py#L24) |
| detector: гібрид signature + вектори | ✅ | [app/incidents/detector.py:28](app/incidents/detector.py#L28) |
| linker: гілки a–d | ✅ | [app/incidents/linker.py:113](app/incidents/linker.py#L113) |
| impact: affected лише через SQL | ✅ | [app/incidents/impact.py:176](app/incidents/impact.py#L176) |
| planner / executor: аудит, ідемпотентність | ✅ | [app/incidents/planner.py:52](app/incidents/planner.py#L52), [app/actions/executor.py:72](app/actions/executor.py#L72) |
| Векторне сховище без сервера, faq_rag ескалює | ✅ SQLite + numpy; без фрагмента агент спершу уточнює, після 3 спроб ескалює | [app/vector/store.py:25](app/vector/store.py#L25), [app/agent/faq_rag.py:97](app/agent/faq_rag.py#L97) |
| alerter SEV1/2/3, повтор до «Беру в роботу», GitHub Issue | ❌ є лише баг-тікет з severity/priority (P0–P3) у канал розробників | [app/bot/cards.py:208](app/bot/cards.py#L208) |
| dashboard: KPI, топ-3 скарги дня, стрічка дій AI, гроші і бюджет, інциденти, автооновлення | ⚠️ `/stats` (KPI), `/dashboard` (рішення, причини ескалації), `/incidents`; топ-3 скарг і стрічки дій AI немає, вебверсії немає | [app/stats.py:27](app/stats.py#L27), [:67](app/stats.py#L67) |
| bank_sim: /break, /fix, /reset_demo, /simulate, /demo_login, вебекран, фішинг лише `.example` | ⚠️ є `/simulate` (7 сценаріїв + reset), `/demo_login` (22 персонажі); немає /break, /fix, вебекрану; фішинговий домен `demobank-secure.example` (реальна TLD) | [app/bot/simulate.py:69](app/bot/simulate.py#L69), [app/bot/personas.py:8](app/bot/personas.py#L8), [app/db/seed.py:45](app/db/seed.py#L45) |
| Три чати Telegram, кнопки, /stats, /top, /incidents | ⚠️ клієнт ✅, оператори ✅ (+ живий чат, /dashboard), `/top` немає; група розробників не підключена (тікети йдуть операторам) | [app/bot/](app/bot/) |
| README українською (проблема, affected vs at_risk, формула лімітів, mermaid, запуск, демо, безпека, тести) | ✅ | [README.md](README.md) |
| Понад еталон: AI-агент з інструментами (уточнює, яке списання; диспут; скасування підписки; goodwill), живий чат з оператором | ✅ | [app/agent/assistant.py:89](app/agent/assistant.py#L89) |

## 3. Результати `python -m scripts.demo_check --mock-llm`

| Сценарій | Результат |
|---|---|
| A масовий збій | ✅ інцидент, 15 affected (12 не писали), 25 at_risk, 0 повідомлень at_risk, баг-тікет P0 з доказами · ⚠️ SEV1-алертера немає |
| B автоповернення лояльному | ✅ 180 грн, actor=agent, approved_by=auto-policy, reason + evidence |
| C частий поворотник | ✅ needs_operator + operator_note |
| D змінив телефон | ✅ needs_operator + security_review |
| E кредит | ✅ прострочка → оператор; без прострочки → авто; відстрочка / реструктуризація / ліміт → оператор |
| F пізні скарги a/b/c/d | ✅ усі 4 гілки |
| G масовий план | ✅ завжди через оператора · ⚠️ INCIDENT_AUTO_BUDGET немає (поведінка суворіша) |
| H ідемпотентність | ✅ |
| I circuit breaker | ✅ автономію вимкнено · ⚠️ денного бюджету і SEV1 немає |
| J prompt injection | ✅ |
| K підроблений скріншот | ✅ |
| L дві легітимні однакові покупки | ❌ вважаються дублем (у даних немає retry-логів) |
| M закриття / fix | ✅ фінальні повідомлення лише тим, хто писав · ⚠️ /fix немає |
| N без ембеддингів | ✅ |
| O LLM повертає сміття / таймаут | ✅ |

Разом: ✅ 26 · ⚠️ 4 · ❌ 1.

## 4. Проблеми за пріоритетом

| # | P | Файл | Суть | Як виправити | Час |
|---|---|---|---|---|---|
| 1 | P0 | app/agent/decision.py | Автоповернення без лімітів на клієнта і на день: за сценарію L (дві легітимні покупки) агент може повернути до 2 000 грн за раз і без обмежень по сумі | ліміт 500 грн/транзакція (дубль), 1 000 грн/клієнт/30 днів (усі автоповернення), денний бюджет — у .env, блокери в `enforce` | 10 хв |
| 2 | P1 | seed, assets, tests | Фішинговий домен на реальній TLD `.top` | `demobank-secure.example` | 10 хв |
| 3 | P1 | app/stats.py, operator_handlers | Немає «топ-3 скарги дня» і стрічки дій AI | `/top` + стрічка в `/dashboard` | 10 хв |
| 4 | P1 | impact.py | Дубль = однакова сума за < 2 хв, без retry-логів (сценарій L) | потрібне поле `auth_retry` у транзакціях (нова фіча) | 30+ хв |
| 5 | P1 | — | Немає алертера SEV1/2/3 з повтором | нова фіча | 45 хв |
| 6 | P1 | — | Немає /break, /fix, вебекрану банку, вебдашборду | нова фіча | 1–2 год |
| 7 | P2 | .env | Група розробників не підключена | `/chatid` у групі → `DEV_CHAT_ID` | 2 хв (потрібна дія людини) |
| 8 | P2 | media.py | Немає ffmpeg на демо-машині | `brew install ffmpeg` | 3 хв |
| 9 | P2 | risk.py | Ваги ризику в коді, а не в .env | — | 10 хв |

## 5. Критерії журі

| Критерій | Що підтверджує | Слабко | Найдешевша правка |
|---|---|---|---|
| Operational Impact | 1 скарга → знайдено й виправлено 15 випадків; миттєва привʼязка; `/stats` рахує зекономлені хвилини | немає вебдашборду для керівника | показати `/stats` і `/dashboard` на демо |
| Innovation | проактивний пошук affected, у тому числі тих, хто не писав; правило at_risk без повідомлень | — | наголосити в пітчі |
| Use of AI | Gemma 4 — triage, vision, агент з інструментами, рішення про гроші, гіпотеза інциденту; ембеддинги для групування | — | показати живий діалог Софії |
| Reasoning & Context | памʼять розмови, уточнення за реальними операціями, гілки a–d, operator_note з фактами | — | — |
| Technical Execution | 101 тест, policy в коді, аудит, ідемпотентність, graceful degradation | легітимні повтори = дубль | ліміти (виправлення 1) |
| Presentation & Demo | `/simulate`, `/demo_login`, шпаргалка, офлайн-демо | ліміт квоти 14 RPM → затримки | заздалегідь `/simulate reset` |

## 6. Виправлення

_(оновлюється нижче)_

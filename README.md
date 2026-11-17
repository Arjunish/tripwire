# Tripwire

**A smoke alarm for your supply chain that gets smarter every night.**

Overnight agents watch for blacklisted companies and discontinued parts, trace them through your products using a Cognee knowledge graph, make educated guesses (conjectures), test them, and use `cognee.improve()` to learn from the results.

```
metadata ──► graph ──► conjectures ──► test (supplier email + human) ──► improve() ──┐
   ▲                                                                                  │
   └──────────────────────────── next night, smarter ─────────────────────────────────┘
```

## Agents (all share one Cognee memory)

| Agent | What it does | Cognee call |
|---|---|---|
| Watcher | Stores overnight events as dated facts (`[effective_date=...]`) | `remember(..., node_set=["events"])` |
| Analyst | Traces part → supplier → owner → product → contract, writes conjectures | `remember(..., session_id="night-N")` |
| Chaser | Emails the supplier to test an ownership guess and stores the reply | `remember(..., session_id="night-N")` |
| Human | Confirms ✅ or rejects ❌ each conjecture | `remember(..., session_id="night-N")` |
| Learning | Merges the night's session (feedback, confirmed facts, lessons) into the permanent graph | `improve(dataset="tripwire", session_ids=["night-N"])` |
| Night 2 | Asks memory what it now knows | `recall(...)` |

## Website (recommended for the demo)

```bash
python3 server.py            # real Cognee (pip install cognee + LLM_API_KEY set)
python3 server.py --offline  # same site, no Cognee
```
Open **http://localhost:8000** and click the six steps on the left. Each product is a live web of parts, suppliers, owners and countries. Click any circle to see its blast radius.

## Command line

```bash
pip install cognee
export LLM_API_KEY="sk-..."          # OpenAI key by default; see Cognee docs for other providers

python tripwire.py demo              # full run: setup → night 1 → review → improve → night 2
python tripwire.py demo --pause      # waits for Enter between steps (good for recording)
python tripwire.py demo --offline    # no Cognee or API key needed (memory calls logged to out/memory_log.jsonl)
```

Step by step:

```bash
python tripwire.py setup
python tripwire.py night 1
python tripwire.py review 1 --interactive   # answer y/n yourself
python tripwire.py improve 1
python tripwire.py night 2
python tripwire.py ask "Who owns Halden Radar Systems?"
python tripwire.py reset
```

## The demo story

- **Night 1:** Jinlong Holdings is blacklisted. Nothing matches directly, but Halden (the radar supplier) shares a director with a Jinlong subsidiary, so the agent **guesses** Halden may be owned by Jinlong. The Chaser emails Halden, and the reply confirms it. A second blacklisting, Brightwave *Industrial* (China), looks like your camera supplier Brightwave *Electronics* (Taiwan), so it raises a possible false alarm.
- **Review:** ✅ the Halden link, ❌ the Brightwave match (different country).
- **`improve()`:** the confirmed ownership link and the lesson become permanent memory.
- **Night 2:** Corvane Capital, Jinlong's parent, is sanctioned. Tripwire traces **Corvane → Jinlong → Halden → radar → Army product**, a path it could not have found on night 1. Brightwave *Trading* (China) is **suppressed** because of what it learned.

## Shared memory with permissions

Every agent has its own identity and read/write rights on three zones of the shared Cognee memory:
**events** (raw overnight facts), **session** (the night's scratchpad of guesses, replies and feedback) and **graph** (permanent knowledge).
Agents can only *propose* into the session. The Analyst and Chaser are 🔒 blocked from writing guesses or supplier claims as permanent fact.
Only the Learner (`cognee.improve()`) can promote knowledge to the permanent graph, and only after a human has reviewed it.
Every write is tagged `agent:<name>` via Cognee `node_set` for provenance. See the **Agents 🔒** tab and `out/access.jsonl`.
The permission check is enforced in Tripwire's memory layer; in production it maps onto Cognee's multi-user mode (agents as users with a parent owner, per-dataset read/write/share).

## Files

- `tripwire.py`: all agents and commands
- `data/bom.csv`: parts list for a fictional drone startup (Kestrel Robotics)
- `data/products.json`: which product goes to which customer and contract
- `data/companies.json`: supplier profiles, known owners and soft evidence ("hints")
- `data/night1.json`, `data/night2.json`: simulated overnight events
- `data/supplier_replies.json`: the Chaser's simulated email replies
- `data/feedback_night1.json`: canned human review for the non-interactive demo
- `out/`: morning briefs, state and the memory call log

All companies and events are fictional and simulated.

## Notes

- The graph trace is mirrored locally so the demo is deterministic. Cognee holds the shared memory, the session learning and the `recall` answers.
- Dates are embedded in the text as `[effective_date=...]`. Cognee's conflict review reads dates from document metadata (`effective_date` key), so passing real metadata is the natural next step.
- Next steps: live feeds (trade.gov Consolidated Screening List API, Federal Register API, DigiKey/Mouser stock), sending real Chaser emails, and Cognee's MCP server so a Claude or Cursor agent can query the same memory.

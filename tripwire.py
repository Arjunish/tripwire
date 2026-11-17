#!/usr/bin/env python3
"""
Tripwire: overnight supply-chain risk agents that learn, built on Cognee memory.

The loop:  metadata -> graph -> conjectures -> test -> improve() -> repeat

Agents (all share one Cognee memory):
  Watcher  - pulls overnight events (blacklists, end-of-life notices) and stores them as dated facts
  Analyst  - traces the graph (part -> supplier -> owner -> product -> contract) and writes conjectures
  Chaser   - emails suppliers to check conjectures and stores their replies
  Human    - confirms or rejects each conjecture in the morning brief
  improve()- merges confirmed knowledge and feedback from the night's session into the permanent graph

Usage:
  python tripwire.py demo              # run everything end to end (best for the video)
  python tripwire.py setup             # load parts list + company profiles into Cognee
  python tripwire.py night 1           # run one overnight cycle, write the morning brief
  python tripwire.py review 1 [--interactive]   # confirm / reject conjectures
  python tripwire.py improve 1         # cognee.improve() on that night's session
  python tripwire.py ask "Who owns Halden Radar Systems?"
  python tripwire.py reset
Add --offline to any command to run without Cognee (logs memory calls to out/memory_log.jsonl).
"""

import argparse
import asyncio
import csv
import json
import re
import shutil
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"
STATE_FILE = OUT / "state.json"
DATASET = "tripwire"
NAME_MATCH_THRESHOLD = 0.6
AUTO_CONFIRM_AT = 0.9   # at or above this confidence, Tripwire decides by itself; below, it asks a human

# Who may read / write which part of the shared Cognee memory.
#   events  = raw overnight facts      session = the night's scratchpad (guesses, replies, feedback)
#   graph   = permanent knowledge      Only the Learner (cognee.improve, run on human-reviewed sessions)
#                                      and the Owner (initial upload) may write the permanent graph.
PERMS = {
    "Owner":   {"role": "Founder uploading their parts list", "read": ["graph", "events", "session"], "write": ["graph"]},
    "Watcher": {"role": "Reads overnight news",               "read": [],                              "write": ["events"]},
    "Analyst": {"role": "Traces risk, writes guesses",        "read": ["events", "graph"],             "write": ["session"]},
    "Chaser":  {"role": "Verifies guesses with suppliers",    "read": ["session"],                     "write": ["session"]},
    "Collector": {"role": "LLM + web search: live shipping & trade news", "read": [],                    "write": ["events"]},
    "Judge":   {"role": "LLM that decides each guess, with a reason", "read": ["events", "graph", "session"], "write": ["session"]},
    "Human":   {"role": "Can override the Judge",        "read": ["graph", "events", "session"],  "write": ["session"]},
    "Learner": {"role": "cognee.improve() on reviewed nights","read": ["session"],                     "write": ["graph"]},
}


class PermissionDenied(Exception):
    pass


def access_log(agent, action, zone, allowed, note=""):
    OUT.mkdir(exist_ok=True)
    with (OUT / "access.jsonl").open("a") as f:
        f.write(json.dumps({"agent": agent, "action": action, "zone": zone, "allowed": allowed, "note": note}) + "\n")
    if not allowed:
        print(f"🔒 BLOCKED: {agent} tried to {action} '{zone}' memory. {note}")


def check(agent, action, zone, note=""):
    allowed = zone in PERMS.get(agent, {}).get(action, [])
    access_log(agent, action, zone, allowed, note)
    if not allowed:
        raise PermissionDenied(f"{agent} has no {action} permission on {zone}")


# --------------------------------------------------------------------------------------
# Memory layer: thin wrapper over Cognee (remember / recall / improve)
# --------------------------------------------------------------------------------------
class Memory:
    def __init__(self, offline: bool = False):
        self.offline = offline
        self.cognee = None
        if not offline:
            try:
                import cognee  # noqa: F401

                self.cognee = cognee
            except ImportError:
                print("⚠️  cognee isn't installed, so running in offline mode (pip install cognee)\n")
                self.offline = True
        OUT.mkdir(exist_ok=True)
        self.log_path = OUT / "memory_log.jsonl"

    def _log(self, op, **kw):
        OUT.mkdir(exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps({"op": op, **kw}) + "\n")

    async def remember(self, text: str, session_id: str | None = None, node_set: list[str] | None = None,
                       agent: str = "Owner", zone: str | None = None, note: str = ""):
        """Permanent memory if session_id is None, otherwise the night's session memory.
        Every write is permission-checked and tagged with the agent that made it (provenance)."""
        zone = "session" if session_id else (zone or "graph")
        check(agent, "write", zone, note or text[:80])
        # Provenance: permanent writes get an agent tag via node_set. Sessions in Cognee pin one node_set,
        # so inside a session the agent name goes into the text instead.
        if session_id:
            text = f"[agent={agent}] {text}"
            node_set = None
        else:
            node_set = (node_set or []) + [f"agent:{agent}"]
        self._log("remember", agent=agent, zone=zone, session_id=session_id, node_set=node_set, text=text)
        if self.offline:
            return
        kwargs = {"dataset_name": DATASET}
        if node_set and not session_id:
            kwargs["node_set"] = node_set
        if session_id:
            kwargs["session_id"] = session_id
        else:
            # We call improve() ourselves at the end of each night, so the demo shows it explicitly.
            kwargs["self_improvement"] = False
        await self.cognee.remember(text, **kwargs)

    async def recall(self, question: str, session_id: str | None = None, agent: str = "Human") -> str | None:
        check(agent, "read", "graph", question[:80])
        self._log("recall", agent=agent, session_id=session_id, question=question)
        if self.offline:
            return None
        try:
            res = await (self.cognee.recall(question, session_id=session_id) if session_id
                         else self.cognee.recall(question))
        except TypeError:
            res = await self.cognee.recall(question)
        return flatten(res)

    async def improve(self, session_ids: list[str], agent: str = "Learner") -> list[tuple[str, str, str]]:
        check(agent, "read", "session", f"reading {session_ids}")
        check(agent, "write", "graph", f"merging {session_ids} into permanent memory")
        self._log("improve", agent=agent, session_ids=session_ids)
        if self.offline:
            return [("improve", "skipped", "offline mode")]
        result = await self.cognee.improve(dataset=DATASET, session_ids=session_ids)
        try:
            return [(s.stage, s.status, s.reason or "") for s in result.stages]
        except AttributeError:
            return [("improve", "completed", str(result)[:200])]


def flatten(res) -> str:
    if res is None:
        return ""
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        for key in ("answer", "text", "search_result", "content"):
            if key in res:
                return flatten(res[key])
        return json.dumps(res)[:800]
    if isinstance(res, (list, tuple)):
        return "\n".join(flatten(r) for r in res if r)[:1500]
    for attr in ("answer", "text", "search_result", "content"):
        if hasattr(res, attr):
            return flatten(getattr(res, attr))
    return str(res)[:1500]


# --------------------------------------------------------------------------------------
# Local mirror of the graph (deterministic, so the demo is reliable)
# --------------------------------------------------------------------------------------
def load_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def load_bom():
    with (DATA / "bom.csv").open() as f:
        return list(csv.DictReader(f))


def load_state():
    return load_json(STATE_FILE, {"confirmed_links": [], "rules": [], "nights_run": [], "learned_log": []})


def save_state(state):
    OUT.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def norm(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    stop = {"co", "ltd", "inc", "llc", "corp", "limited", "pte", "group", "holdings", "the"}
    return " ".join(w for w in name.split() if w not in stop)


def name_similarity(a: str, b: str) -> float:
    na, nb = norm(a), norm(b)
    score = SequenceMatcher(None, na, nb).ratio()
    if na.split() and nb.split() and na.split()[0] == nb.split()[0]:
        score = max(score, 0.85)  # same distinctive first word, e.g. "Brightwave ..."
    return round(score, 2)


class Graph:
    def __init__(self, state, use_learning=True):
        self.bom = load_bom()
        self.products = load_json(DATA / "products.json")
        self.companies = load_json(DATA / "companies.json")
        self.parent = {n: c.get("parent") for n, c in self.companies.items() if c.get("parent")}
        self.learned_parents = {}
        if use_learning:
            for link in state["confirmed_links"]:
                self.learned_parents[link["child"]] = link["parent"]

    @property
    def suppliers(self):
        return sorted({r["supplier"] for r in self.bom})

    def owner_chain(self, company):
        """[(company, how_we_know)] walking up: supplier -> parent -> grandparent."""
        chain, cur, seen = [(company, "supplier")], company, {company}
        while True:
            if cur in self.parent:
                nxt, how = self.parent[cur], "registry"
            elif cur in self.learned_parents:
                nxt, how = self.learned_parents[cur], "learned"
            else:
                break
            if nxt in seen:
                break
            chain.append((nxt, how))
            seen.add(nxt)
            cur = nxt
        return chain

    def exposure(self, supplier):
        rows = [r for r in self.bom if r["supplier"] == supplier]
        out = {}
        for r in rows:
            p = out.setdefault(r["product"], {"parts": [], **self.products[r["product"]]})
            p["parts"].append(r)
        return out


# --------------------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------------------
async def watcher(mem: Memory, night: int, session: str):
    """Pull overnight events and store them as dated facts (metadata)."""
    events = load_json(DATA / f"night{night}.json", [])
    for e in events:
        if e["type"] == "blacklist_add":
            text = (f"[effective_date={e['effective_date']}] [source={e['source']}] "
                    f"{e['entity']} ({e['country']}) was added to the {e['list']}.")
        else:
            text = (f"[effective_date={e['effective_date']}] [source={e['source']}] "
                    f"Part {e['part_mpn']} status changed to {e['status']}. {e.get('note', '')}")
        await mem.remember(text, node_set=["events"], agent="Watcher", zone="events")
    print(f"👁  Watcher: stored {len(events)} overnight events as dated facts")
    return events


def analyst(events, graph: Graph, state):
    """Trace each event through the graph. Returns (alerts, conjectures, suppressed)."""
    alerts, conjectures, suppressed = [], [], []
    rules = {r["rule"] for r in state["rules"]}

    for e in events:
        if e["type"] == "lifecycle":
            rows = [r for r in graph.bom if r["part_mpn"] == e["part_mpn"]]
            for r in rows:
                weeks = int(r["stock_on_hand"]) / max(int(r["weekly_usage"]), 1)
                alerts.append({
                    "kind": "lifecycle", "event": e, "supplier": r["supplier"],
                    "products": {r["product"]: {"parts": [r], **graph.products[r["product"]]}},
                    "path": [f"{r['part_mpn']} ({r['description']})", r["product"]],
                    "weeks_left": round(weeks, 1), "alternates": r["alternates"],
                    "severity": "MEDIUM", "learned": False,
                })
            continue

        entity = e["entity"]
        hit = False
        # 1) Facts: is the blacklisted entity anywhere in a supplier's ownership chain?
        for s in graph.suppliers:
            chain = graph.owner_chain(s)
            names = [norm(c) for c, _ in chain]
            if norm(entity) in names:
                idx = names.index(norm(entity))
                used_learning = any(how == "learned" for _, how in chain[: idx + 1])
                exp = graph.exposure(s)
                defense = any(p["defense_contract"] for p in exp.values())
                alerts.append({
                    "kind": "blacklist", "event": e, "supplier": s, "products": exp,
                    "path": [c for c, _ in chain[: idx + 1]][::-1],
                    "severity": "HIGH" if defense else "MEDIUM", "learned": used_learning,
                })
                hit = True
        if hit:
            continue

        # 2) Conjecture: soft evidence links a supplier to the entity (e.g. shared director)
        for s in graph.suppliers:
            for h in graph.companies.get(s, {}).get("hints", []):
                if norm(h["entity"]) == norm(entity):
                    conjectures.append({
                        "id": f"ownership|{s}|{entity}", "kind": "ownership", "event": e,
                        "supplier": s, "entity": entity, "confidence": 0.55,
                        "evidence": h["evidence"], "products": graph.exposure(s),
                    })

        # 3) Conjecture: the names look alike, so is it the same company?
        for s in graph.suppliers:
            score = name_similarity(s, entity)
            if score < NAME_MATCH_THRESHOLD:
                continue
            s_country = graph.companies.get(s, {}).get("country")
            if "name_match_requires_same_country" in rules and s_country != e["country"]:
                suppressed.append({
                    "supplier": s, "entity": entity, "score": score,
                    "why": (f"names look alike ({score}) but {s} is in {s_country} and {entity} is in "
                            f"{e['country']}. Last time you rejected a match like this."),
                })
                continue
            conjectures.append({
                "id": f"same_entity|{s}|{entity}", "kind": "same_entity", "event": e,
                "supplier": s, "entity": entity, "confidence": score,
                "evidence": f"Name similarity {score} ({s}, {s_country} vs {entity}, {e['country']})",
                "products": graph.exposure(s),
            })
    return alerts, conjectures, suppressed


async def write_conjectures(mem: Memory, conjectures, session: str):
    """Analyst: guesses go to the night's session. It may not write them as permanent fact."""
    for e in {json.dumps(c["event"], sort_keys=True) for c in conjectures}:
        access_log("Analyst", "read", "events", True, json.loads(e).get("entity", ""))
    for c in conjectures:
        verb = "may be owned by" if c["kind"] == "ownership" else "may be the same company as"
        try:
            await mem.remember(f"FACT: {c['supplier']} {verb.replace('may be ', 'is ')} {c['entity']}.",
                               agent="Analyst", zone="graph", note="Tried to save its own guess as permanent fact")
        except PermissionDenied:
            c["analyst_blocked"] = True
        await mem.remember(
            f"CONJECTURE ({c['id']}): {c['supplier']} {verb} {c['entity']}. "
            f"Confidence {c['confidence']}. Evidence: {c['evidence']}", session_id=session, agent="Analyst")


async def chaser(mem: Memory, conjectures, session: str):
    """Email suppliers to test ownership conjectures; store replies in the night's session."""
    replies = load_json(DATA / "supplier_replies.json", {})
    for c in conjectures:
        if c["kind"] != "ownership":
            continue
        print(f"📧 Chaser: emailing {c['supplier']}: \"Can you confirm whether you are owned by "
              f"{c['entity']} or any of its affiliates?\"")
        reply = replies.get(c["supplier"])
        if not reply:
            c["chaser"] = "No reply yet."
            continue
        c["chaser"] = reply
        if norm(c["entity"]) in norm(reply) and "subsidiary" in reply.lower():
            c["confidence"] = 0.9
        await mem.remember(f"Supplier email reply from {c['supplier']}: {reply}", session_id=session, agent="Chaser")
        print(f"   ↳ reply received, confidence now {c['confidence']}")
        # The Chaser would love to save the supplier's claim as fact. It isn't allowed to.
        try:
            await mem.remember(f"FACT: {c['supplier']} is owned by {c['entity']}.", agent="Chaser", zone="graph",
                               note="Supplier says so, but a supplier's word isn't a verified fact")
        except PermissionDenied:
            c["chaser_blocked"] = True


# --------------------------------------------------------------------------------------
# Morning brief
# --------------------------------------------------------------------------------------
def product_line(products):
    return ", ".join(f"{name} ({p['customer']})" for name, p in products.items())


def write_brief(night, alerts, conjectures, suppressed, cognee_answer=None):
    L = [f"# ☀️ Tripwire morning brief: night {night}", ""]
    L.append("## 🚨 Alerts (confirmed path)")
    if not alerts:
        L.append("_Nothing confirmed._")
    for a in sorted(alerts, key=lambda a: a["severity"] != "HIGH"):
        e = a["event"]
        if a["kind"] == "blacklist":
            L.append(f"- **[{a['severity']}] {e['entity']} was added to the {e['list']}** "
                     f"(effective {e['effective_date']}, {e['source']}).")
            L.append(f"  - Path: {' → '.join(a['path'])} → your parts")
            L.append(f"  - Affects: {product_line(a['products'])}")
            parts = {r['part_mpn'] for p in a['products'].values() for r in p['parts']}
            L.append(f"  - Parts: {', '.join(sorted(parts))}")
            alts = {r['alternates'] for p in a['products'].values() for r in p['parts'] if r['alternates']}
            if alts:
                L.append(f"  - Alternates: {'; '.join(alts)}")
            if a["learned"]:
                L.append("  - 🧠 **Caught because of an ownership link you confirmed on an earlier night.** "
                         "Without it, this would have been missed.")
        else:
            L.append(f"- **[{a['severity']}] {e['part_mpn']}: {e['status']}.** {e.get('note', '')}")
            L.append(f"  - Affects: {product_line(a['products'])}. About {a['weeks_left']} weeks of stock left.")
            if a["alternates"]:
                L.append(f"  - Alternates: {a['alternates']}")
    L.append("")
    L.append("## 🤔 Conjectures (need your call)")
    if not conjectures:
        L.append("_None._")
    for c in conjectures:
        verb = "may be owned by" if c["kind"] == "ownership" else "may be the same company as"
        L.append(f"- `{c['id']}`: **{c['supplier']} {verb} {c['entity']}** "
                 f"(just added to the {c['event']['list']}). Confidence {c['confidence']}.")
        L.append(f"  - Evidence: {c['evidence']}")
        if c.get("chaser"):
            L.append(f"  - Supplier reply: \"{c['chaser']}\"")
        L.append(f"  - If true, this affects: {product_line(c['products'])}")
        L.append("  - ✅ confirm / ❌ reject")
    if suppressed:
        L.append("")
        L.append("## 🧠 Suppressed by what I learned")
        for s in suppressed:
            L.append(f"- Did not flag **{s['supplier']} ≈ {s['entity']}**: {s['why']}")
    if cognee_answer:
        L.append("")
        L.append("## 🧩 Cognee memory says")
        L.append(cognee_answer.strip())
    text = "\n".join(L) + "\n"
    (OUT / f"brief_night{night}.md").write_text(text)
    return text


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------
async def cmd_setup(mem: Memory):
    graph = Graph(load_state())
    for name, p in graph.products.items():
        rows = [r for r in graph.bom if r["product"] == name]
        parts = "; ".join(f"{r['part_mpn']} ({r['description']}) from {r['supplier']}" for r in rows)
        await mem.remember(
            f"Kestrel Robotics product {name} is sold to {p['customer']}. "
            f"{'It is under a US defense contract.' if p['defense_contract'] else 'It is commercial.'} "
            f"It contains: {parts}.", node_set=["bom"])
    for name, c in graph.companies.items():
        txt = f"{name} is a company based in {c['country']}."
        if c.get("parent"):
            txt += f" It is owned by {c['parent']}."
        await mem.remember(txt, node_set=["companies"])
    print(f"📦 Setup: loaded {len(graph.products)} products, {len(graph.bom)} BOM lines and "
          f"{len(graph.companies)} company profiles into Cognee dataset '{DATASET}'")


async def cmd_night(mem: Memory, night: int):
    state = load_state()
    session = f"night-{night}"
    print(f"\n🌙 ===== Night {night} =====")
    events = await watcher(mem, night, session)
    graph = Graph(state)
    alerts, conjectures, suppressed = analyst(events, graph, state)
    print(f"🔎 Analyst: {len(alerts)} confirmed alerts, {len(conjectures)} conjectures, "
          f"{len(suppressed)} suppressed by learning")
    await write_conjectures(mem, conjectures, session)
    await chaser(mem, conjectures, session)

    answer = None
    if night > 1:
        answer = await mem.recall(
            "Who ultimately owns Halden Radar Systems, and which Kestrel products depend on it?")
    state["nights_run"] = sorted(set(state["nights_run"]) | {night})
    save_state(state)
    (OUT / f"conjectures_night{night}.json").write_text(json.dumps(conjectures, indent=2, default=str))
    brief = write_brief(night, alerts, conjectures, suppressed, answer)
    print("\n" + brief)
    print(f"📝 Saved to out/brief_night{night}.md")


async def cmd_review(mem: Memory, night: int, interactive: bool):
    state = load_state()
    session = f"night-{night}"
    conjectures = load_json(OUT / f"conjectures_night{night}.json", [])
    canned = load_json(DATA / f"feedback_night{night}.json", {})
    print(f"\n🧑 Review of night {night}")
    for c in conjectures:
        if interactive:
            ans = input(f"  {c['supplier']} ↔ {c['entity']} (conf {c['confidence']}). Confirm? [y/n] ").strip().lower()
            fb = {"verdict": "confirm" if ans.startswith("y") else "reject",
                  "reason": "country_mismatch" if (c["kind"] == "same_entity" and not ans.startswith("y")) else "",
                  "note": input("  Note (optional): ").strip()}
        else:
            fb = canned.get(c["id"])
            if not fb:
                continue
        icon = "✅" if fb["verdict"] == "confirm" else "❌"
        print(f"  {icon} {c['id']}: {fb.get('note', '')}")
        await mem.remember(
            f"HUMAN FEEDBACK on {c['id']}: {fb['verdict'].upper()}. {fb.get('note', '')}", session_id=session, agent="Human")

        if fb["verdict"] == "confirm" and c["kind"] == "ownership":
            link = {"child": c["supplier"], "parent": c["entity"], "night": night}
            if link not in state["confirmed_links"]:
                state["confirmed_links"].append(link)
            await mem.remember(
                f"CONFIRMED FACT: {c['supplier']} is a subsidiary of {c['entity']} "
                f"(confirmed by supplier email and by the user).", session_id=session, agent="Human")
            state["learned_log"].append(f"Night {night}: learned {c['supplier']} → owned by {c['entity']}")
        if fb["verdict"] == "reject" and fb.get("reason") == "country_mismatch":
            rule = {"rule": "name_match_requires_same_country", "night": night}
            if rule["rule"] not in {r["rule"] for r in state["rules"]}:
                state["rules"].append(rule)
            await mem.remember(
                "LESSON: a similar company name alone is not enough to flag a supplier when the two "
                "companies are in different countries.", session_id=session, agent="Human")
            state["learned_log"].append(f"Night {night}: learned name-only matches across countries are false alarms")
    save_state(state)


async def cmd_improve(mem: Memory, night: int):
    session = f"night-{night}"
    print(f"\n🧠 cognee.improve(dataset='{DATASET}', session_ids=['{session}'])")
    for stage, status, reason in await mem.improve([session]):
        print(f"   {stage:<28} {status:<18} {reason}")
    state = load_state()
    for line in state["learned_log"]:
        print(f"   • {line}")


async def cmd_ask(mem: Memory, question: str):
    ans = await mem.recall(question)
    print(ans if ans else "(offline mode: no Cognee answer)")


def cmd_reset():
    if OUT.exists():
        shutil.rmtree(OUT)
    print("🧹 Reset local state (out/). To clear Cognee too: python -c \"import asyncio,cognee; "
          "asyncio.run(cognee.forget(dataset='tripwire'))\"")


async def cmd_demo(mem: Memory, pause: bool):
    def step(msg):
        if pause:
            input(f"\n⏎  {msg}")
        else:
            print(f"\n▶ {msg}")

    await cmd_setup(mem)
    step("Night 1: run the overnight agents")
    await cmd_night(mem, 1)
    step("Morning: the human reviews the conjectures")
    await cmd_review(mem, 1, interactive=False)
    step("Run cognee.improve() so the system learns from night 1")
    await cmd_improve(mem, 1)
    step("Night 2: same agents, smarter memory")
    await cmd_night(mem, 2)


def main():
    ap = argparse.ArgumentParser(description="Tripwire: supply-chain agents with Cognee memory")
    ap.add_argument("command", choices=["setup", "night", "review", "improve", "ask", "demo", "reset"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--offline", action="store_true", help="run without Cognee")
    ap.add_argument("--interactive", action="store_true", help="answer conjectures yourself")
    ap.add_argument("--pause", action="store_true", help="pause between demo steps (for recording)")
    a = ap.parse_args()

    if a.command == "reset":
        return cmd_reset()
    mem = Memory(offline=a.offline)
    n = int(a.arg) if a.arg and a.arg.isdigit() else 1
    if a.command == "setup":
        asyncio.run(cmd_setup(mem))
    elif a.command == "night":
        asyncio.run(cmd_night(mem, n))
    elif a.command == "review":
        asyncio.run(cmd_review(mem, n, a.interactive))
    elif a.command == "improve":
        asyncio.run(cmd_improve(mem, n))
    elif a.command == "ask":
        if not a.arg:
            sys.exit("Usage: python tripwire.py ask \"your question\"")
        asyncio.run(cmd_ask(mem, a.arg))
    elif a.command == "demo":
        if load_state()["nights_run"]:
            cmd_reset()
        asyncio.run(cmd_demo(mem, a.pause))


if __name__ == "__main__":
    main()

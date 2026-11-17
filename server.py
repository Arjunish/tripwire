#!/usr/bin/env python3
"""
Tripwire web app: serves the website and runs the agents with Cognee.

    python3 server.py            # real Cognee (needs: pip install cognee, LLM_API_KEY set)
    python3 server.py --offline  # no Cognee, same UI

Then open http://localhost:8000
"""

import argparse
import asyncio
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import tripwire as tw
import llm

WEB = tw.ROOT / "web"
LAST_FILE = tw.OUT / "last.json"

LOOP = asyncio.new_event_loop()
threading.Thread(target=LOOP.run_forever, daemon=True).start()
MEM: tw.Memory = None


def run(coro):
    return asyncio.run_coroutine_threadsafe(coro, LOOP).result()


def load_last():
    return tw.load_json(LAST_FILE, {"night": 0, "alerts": [], "conjectures": [], "suppressed": [],
                                    "answer": None, "stages": [], "timeline": [], "blacklisted": [],
                                    "lifecycle": [], "setup": False, "reviewed": {}})


def save_last(last):
    tw.OUT.mkdir(exist_ok=True)
    LAST_FILE.write_text(json.dumps(last, indent=2, default=str))


def log(last, kind, text):
    last["timeline"].insert(0, {"kind": kind, "text": text, "night": last.get("night", 0)})


# ------------------------------------------------------------------ actions
async def do_setup():
    last = load_last()
    await tw.cmd_setup(MEM)
    last["setup"] = True
    log(last, "setup", "Loaded 2 products, their parts, suppliers and owners into Cognee")
    save_last(last)
    return {"ok": True}


async def do_night(n: int):
    last = load_last()
    state = tw.load_state()
    session = f"night-{n}"
    events = await tw.watcher(MEM, n, session)
    graph = tw.Graph(state)
    alerts, conjectures, suppressed = tw.analyst(events, graph, state)
    await tw.write_conjectures(MEM, conjectures, session)
    await tw.chaser(MEM, conjectures, session)
    answer = None
    if n > 1:
        answer = await MEM.recall(
            "Who ultimately owns Halden Radar Systems, and which Kestrel products depend on it?", agent="Analyst")
    state["nights_run"] = sorted(set(state["nights_run"]) | {n})
    tw.save_state(state)
    (tw.OUT / f"conjectures_night{n}.json").write_text(json.dumps(conjectures, indent=2, default=str))
    tw.write_brief(n, alerts, conjectures, suppressed, answer)

    last.update(night=n, alerts=alerts, conjectures=conjectures, suppressed=suppressed,
                answer=answer, stages=[])
    for e in events:
        if e["type"] == "blacklist_add":
            last["blacklisted"].append({"entity": e["entity"], "list": e["list"], "date": e["effective_date"],
                                        "country": e["country"]})
        else:
            last["lifecycle"].append(e)
    log(last, "night", f"Night {n}: {len(events)} events · {len(alerts)} alerts · "
                       f"{len(conjectures)} guesses · {len(suppressed)} suppressed by learning")
    for c in conjectures:
        if c.get("chaser"):
            log(last, "chaser", f"Chaser emailed {c['supplier']}: reply received")
    for s in suppressed:
        log(last, "learned", f"Suppressed {s['supplier']} ≈ {s['entity']} (learned rule)")
    for c in conjectures:
        if c.get("analyst_blocked"):
            log(last, "blocked", f"🔒 Analyst blocked from saving its guess about {c['supplier']} as fact")
        if c.get("chaser_blocked"):
            log(last, "blocked", f"🔒 Chaser blocked from saving {c['supplier']}'s email claim as fact")
    save_last(last)
    # Judge agent (LLM) decides every guess itself, with a reason. You can still override.
    decisions = await asyncio.to_thread(llm.judge, conjectures, graph.companies)
    if decisions:
        last = await do_review(n, {k: v["verdict"] for k, v in decisions.items()}, agent="Judge", auto=True,
                               reasons=decisions)
    # Advisor agent (LLM) writes today's plan.
    last["advice"] = await asyncio.to_thread(llm.advise, n, alerts, conjectures, decisions, suppressed,
                                             last.get("signals"))
    log(last, "advice", f"Advisor wrote today's plan ({last['advice'].get('by')})")
    save_last(last)
    return last


async def do_review(n: int, decisions: dict, agent: str = "Human", auto: bool = False, reasons=None):
    last = load_last()
    last.setdefault("auto", [])
    last.setdefault("reasons", {})
    state = tw.load_state()
    session = f"night-{n}"
    for c in last["conjectures"]:
        verdict = decisions.get(c["id"])
        if not verdict:
            continue
        if auto:
            r = (reasons or {}).get(c["id"], {})
            note = f"Decided by the Judge agent ({r.get('by', 'rules')}): {r.get('reason', '')}"
            last["auto"].append(c["id"])
            last["reasons"][c["id"]] = r
        else:
            if c["id"] in last["auto"]:
                last["auto"].remove(c["id"])
                note_prefix = "Human override of the Judge. "
            else:
                note_prefix = ""
            note = note_prefix + ("Confirmed by the user." if verdict == "confirm" else "Rejected by the user.")
        await MEM.remember(f"{'JUDGE DECISION' if auto else 'HUMAN FEEDBACK'} on {c['id']}: {verdict.upper()}. {note}",
                           session_id=session, agent=agent)
        last["reviewed"][c["id"]] = verdict
        if verdict == "confirm" and c["kind"] == "ownership":
            link = {"child": c["supplier"], "parent": c["entity"], "night": n}
            if link not in state["confirmed_links"]:
                state["confirmed_links"].append(link)
            await MEM.remember(f"CONFIRMED FACT: {c['supplier']} is a subsidiary of {c['entity']} "
                               f"({note})", session_id=session, agent=agent)
            state["learned_log"].append(f"Night {n}: learned {c['supplier']} → owned by {c['entity']}")
            log(last, "confirm", f"{'🤖 Judge confirmed' if auto else '✅ You confirmed'}: {c['supplier']} is owned by {c['entity']}")
        elif verdict == "confirm":
            log(last, "confirm", f"✅ Confirmed: {c['supplier']} = {c['entity']}")
        if verdict == "reject":
            if c["kind"] == "same_entity" and "name_match_requires_same_country" not in {r["rule"] for r in state["rules"]}:
                state["rules"].append({"rule": "name_match_requires_same_country", "night": n})
                await MEM.remember("LESSON: a similar company name alone is not enough to flag a supplier "
                                   "when the two companies are in different countries.", session_id=session, agent=agent)
                state["learned_log"].append(f"Night {n}: learned name-only matches across countries are false alarms")
            log(last, "reject", f"{'🤖 Judge rejected' if auto else '❌ You rejected'}: {c['supplier']} ≈ {c['entity']}")
    tw.save_state(state)
    save_last(last)
    return last


async def do_improve(n: int):
    last = load_last()
    if not last["reviewed"]:
        tw.access_log("Learner", "write", "graph", False, "No human-reviewed guesses yet, so nothing may be promoted")
        raise tw.PermissionDenied("Review at least one guess first: the Learner only promotes human-approved knowledge")
    stages = await MEM.improve([f"night-{n}"])
    last["stages"] = [{"stage": s, "status": st, "reason": r} for s, st, r in stages]
    log(last, "improve", f"cognee.improve() merged night {n} into permanent memory")
    save_last(last)
    return last


async def do_collect():
    last = load_last()
    g = tw.Graph(tw.load_state())
    countries = sorted({c["country"] for n, c in g.companies.items() if n in g.suppliers})
    signals, err = await asyncio.to_thread(llm.collect_signals, countries)
    if err and not signals:
        tw.access_log("Collector", "write", "events", True, f"no live data: {err[:80]}")
        raise RuntimeError(f"Live data collection failed: {err}")
    for s in signals:
        await MEM.remember(f"[live signal {s.get('date', '')}] [source={s.get('source', '')}] Shipping/trade risk for "
                           f"{s.get('country')}: {s.get('risk')}. {s.get('headline')}",
                           node_set=["signals"], agent="Collector", zone="events")
    last["signals"] = signals
    log(last, "collect", f"Collector pulled {len(signals)} live shipping/trade signals from the web")
    save_last(last)
    return last


async def do_ask(q: str):
    ans = await MEM.recall(q, agent="Human")
    last = load_last()
    log(last, "ask", f"Asked Cognee: {q}")
    save_last(last)
    return {"answer": ans or "(Offline mode: start the server without --offline to ask Cognee.)"}


def do_reset():
    if tw.OUT.exists():
        shutil.rmtree(tw.OUT)
    tw.OUT.mkdir(exist_ok=True)
    return {"ok": True}


# ------------------------------------------------------------------ graph for the UI
def build_graph(product: str):
    state = tw.load_state()
    last = load_last()
    g = tw.Graph(state)
    nodes, links, seen = {}, [], set()

    def add(nid, ntype, **kw):
        if nid not in nodes:
            nodes[nid] = {"id": nid, "type": ntype, "label": kw.pop("label", nid), **kw}
        return nodes[nid]

    def link(a, b, kind, **kw):
        key = (a, b, kind)
        if key not in seen:
            seen.add(key)
            links.append({"source": a, "target": b, "kind": kind, **kw})

    products = list(g.products) if product == "all" else [product]
    for p in products:
        info = g.products[p]
        add(p, "product", customer=info["customer"], defense=info["defense_contract"])
        for r in [r for r in g.bom if r["product"] == p]:
            pid = f"part:{r['part_mpn']}"
            weeks = round(int(r["stock_on_hand"]) / max(int(r["weekly_usage"]), 1), 1)
            add(pid, "part", label=r["part_mpn"], desc=r["description"], weeks=weeks,
                alternates=r["alternates"], supplier=r["supplier"])
            link(p, pid, "contains")
            add(r["supplier"], "supplier")
            link(pid, r["supplier"], "made_by")
            chain = g.owner_chain(r["supplier"])
            for (child, _), (parent, how) in zip(chain, chain[1:]):
                add(parent, "owner")
                link(child, parent, "learned" if how == "learned" else "owned_by")

    # pending guesses from the latest night
    for c in last["conjectures"]:
        if c["supplier"] in nodes and c["id"] not in last["reviewed"]:
            add(c["entity"], "owner", country=c["event"]["country"])
            link(c["supplier"], c["entity"], "conjecture", conjecture_id=c["id"])

    # countries
    for nid, n in list(nodes.items()):
        if n["type"] in ("supplier", "owner"):
            country = g.companies.get(nid, {}).get("country") or n.get("country")
            if country:
                n["country"] = country
                add(f"country:{country}", "country", label=country)
                link(nid, f"country:{country}", "located_in")

    sig = {s.get("country", "").lower(): s for s in last.get("signals", [])}
    for n in nodes.values():
        if n["type"] == "country" and n["label"].lower() in sig:
            n["signal"] = sig[n["label"].lower()]
    black = {tw.norm(b["entity"]): b for b in last["blacklisted"]}
    for n in nodes.values():
        if n["type"] in ("supplier", "owner") and tw.norm(n["id"]) in black:
            n["blacklisted"] = black[tw.norm(n["id"])]

    hot_nodes, hot_links = set(), set()
    for a in last["alerts"]:
        if a["kind"] == "blacklist":
            path = a["path"]
            hot_nodes.update(path)
            for x, y in zip(path, path[1:]):
                hot_links.add(tuple(sorted((x, y))))
        for pname, pinfo in a["products"].items():
            if pname not in nodes:
                continue
            hot_nodes.add(pname)
            for r in pinfo["parts"]:
                pid = f"part:{r['part_mpn']}"
                hot_nodes.update({pid})
                hot_links.add(tuple(sorted((pname, pid))))
                if a["kind"] == "blacklist":
                    hot_nodes.add(r["supplier"])
                    hot_links.add(tuple(sorted((pid, r["supplier"]))))
    for n in nodes.values():
        n["hot"] = n["id"] in hot_nodes
        n["pending"] = any(c["entity"] == n["id"] and c["id"] not in last["reviewed"]
                           for c in last["conjectures"])
    for l in links:
        l["hot"] = tuple(sorted((l["source"], l["target"]))) in hot_links
    return {"nodes": list(nodes.values()), "links": links}


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            body = (WEB / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        elif u.path == "/api/state":
            st = tw.load_state()
            acc = []
            if (tw.OUT / "access.jsonl").exists():
                acc = [json.loads(l) for l in (tw.OUT / "access.jsonl").read_text().splitlines() if l.strip()]
            self.send_json({**load_last(), "offline": MEM.offline, "products": list(tw.Graph(st).products),
                            "learned": st["learned_log"], "rules": st["rules"], "perms": tw.PERMS,
                            "access": acc[::-1][:60]})
        elif u.path == "/api/graph":
            self.send_json(build_graph(q.get("product", ["Kestrel Sentinel"])[0]))
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(n) or b"{}")
        try:
            if u.path == "/api/reset":
                self.send_json(do_reset())
            elif u.path == "/api/setup":
                self.send_json(run(do_setup()))
            elif u.path == "/api/night":
                self.send_json(run(do_night(int(data.get("n", 1)))))
            elif u.path == "/api/review":
                self.send_json(run(do_review(int(data.get("n", 1)), data.get("decisions", {}))))
            elif u.path == "/api/improve":
                self.send_json(run(do_improve(int(data.get("n", 1)))))
            elif u.path == "/api/collect":
                self.send_json(run(do_collect()))
            elif u.path == "/api/ask":
                self.send_json(run(do_ask(data.get("q", ""))))
            else:
                self.send_json({"error": "not found"}, 404)
        except Exception as e:  # show errors in the UI instead of crashing
            import traceback
            traceback.print_exc()
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    global MEM
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    MEM = tw.Memory(offline=a.offline)
    print(f"\n⚡ Tripwire is running → open http://localhost:{a.port}  "
          f"({'offline mode' if MEM.offline else 'Cognee live'})\n")
    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

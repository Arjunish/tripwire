"""
LLM agents for Tripwire. Each one falls back to simple rules if the LLM call fails,
so the demo never breaks.

  judge()            - decides each guess (confirm / reject) with a reason, like an analyst would
  advise()           - writes today's action plan from the morning brief
  collect_signals()  - uses web search to pull live shipping / trade disruption news per country
"""

import json
import os
import re

MODEL = os.getenv("TRIPWIRE_MODEL", "gpt-4o-mini")


def _client():
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key)
    except Exception:
        return None


def _parse_json(text):
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    raw = m.group(1) if m else text
    start = min([i for i in (raw.find("{"), raw.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    try:
        return json.loads(raw[start:])
    except Exception:
        end = max(raw.rfind("}"), raw.rfind("]"))
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            return None


def chat_json(system, user):
    c = _client()
    if not c:
        return None
    try:
        r = c.chat.completions.create(
            model=MODEL, temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return _parse_json(r.choices[0].message.content)
    except Exception as e:
        print(f"⚠️  LLM call failed, using fallback rules: {e}")
        return None


# ---------------------------------------------------------------- Judge
JUDGE_SYSTEM = """You are a supply-chain compliance analyst at a drone startup that sells to the US Army.
For each guess, decide whether to CONFIRM it (treat as true) or REJECT it (treat as false), based only on the evidence.
Be careful: a wrong 'confirm' can make the company drop a good supplier; a wrong 'reject' can break US export law.
Supplier written confirmations and corporate registry links are strong evidence. Similar names alone are weak,
especially across different countries.
Return JSON: {"decisions":[{"id":..., "verdict":"confirm"|"reject", "confidence":0-1, "reason":"one short sentence"}]}"""


def judge(conjectures, companies):
    items = [{
        "id": c["id"], "claim": f"{c['supplier']} {'is owned by' if c['kind'] == 'ownership' else 'is the same company as'} {c['entity']}",
        "supplier_country": companies.get(c["supplier"], {}).get("country"),
        "entity_country": c["event"].get("country"), "evidence": c["evidence"],
        "supplier_email_reply": c.get("chaser"), "prior_confidence": c["confidence"],
    } for c in conjectures]
    out = chat_json(JUDGE_SYSTEM, json.dumps(items, indent=2))
    decisions = {}
    if out and isinstance(out.get("decisions"), list):
        for d in out["decisions"]:
            if d.get("id") and d.get("verdict") in ("confirm", "reject"):
                decisions[d["id"]] = {"verdict": d["verdict"], "reason": d.get("reason", ""),
                                      "confidence": d.get("confidence"), "by": "LLM"}
    for c in conjectures:  # fallback rules for anything the LLM didn't decide
        if c["id"] in decisions:
            continue
        same_country = companies.get(c["supplier"], {}).get("country") == c["event"].get("country")
        if c["kind"] == "ownership" and c["confidence"] >= 0.9:
            decisions[c["id"]] = {"verdict": "confirm", "by": "rules", "confidence": c["confidence"],
                                  "reason": "Registry link plus the supplier's own written confirmation."}
        elif c["kind"] == "same_entity" and not same_country:
            decisions[c["id"]] = {"verdict": "reject", "by": "rules", "confidence": 0.8,
                                  "reason": "Only the names match, and the companies are in different countries."}
    return decisions


# ---------------------------------------------------------------- Advisor
ADVISE_SYSTEM = """You are the chief operating officer's assistant at a 10-person drone startup (Kestrel Robotics).
Given this morning's supply-chain brief, write today's plan: the 3 most important actions, most urgent first.
Each action must be concrete (who to contact, what to order, what to tell which customer) and say why.
Return JSON: {"headline":"one sentence summary of the day","actions":[{"priority":"high|medium|low","action":"...","why":"..."}]}"""


def advise(night, alerts, conjectures, decisions, suppressed, signals=None):
    brief = {
        "night": night,
        "alerts": [{"what": (a["event"].get("entity") or a["event"].get("part_mpn")), "kind": a["kind"],
                    "path": a.get("path"), "products": list(a["products"]), "severity": a["severity"],
                    "weeks_of_stock": a.get("weeks_left"),
                    "alternates": sorted({r["alternates"] for p in a["products"].values() for r in p["parts"] if r["alternates"]})}
                   for a in alerts],
        "decided_guesses": [{"claim": f"{c['supplier']} ~ {c['entity']}", **decisions.get(c["id"], {})} for c in conjectures],
        "ignored_false_alarms": [s["why"] for s in suppressed],
        "live_shipping_signals": signals or [],
    }
    out = chat_json(ADVISE_SYSTEM, json.dumps(brief, indent=2, default=str))
    if out and out.get("actions"):
        out["by"] = "LLM"
        return out
    # fallback
    actions = []
    for a in alerts:
        if a["kind"] == "blacklist":
            parts = sorted({r["part_mpn"] for p in a["products"].values() for r in p["parts"]})
            actions.append({"priority": "high", "action": f"Pause new orders from {a['supplier']} and qualify an alternate for {', '.join(parts)}.",
                            "why": f"{a['event']['entity']} is upstream of {a['supplier']} and affects {', '.join(a['products'])}."})
            if any(p.get("defense_contract") for p in a["products"].values()):
                actions.append({"priority": "high", "action": "Tell your Army contracting officer and compliance counsel today.",
                                "why": "Defense contracts require prompt disclosure of restricted-party exposure."})
        else:
            actions.append({"priority": "medium", "action": f"Place a last-time buy for {a['event']['part_mpn']} or switch to {a.get('alternates') or 'an alternate'}.",
                            "why": f"About {a.get('weeks_left')} weeks of stock left."})
    return {"headline": "Here's what needs attention today.", "actions": actions[:3], "by": "rules"}


# ---------------------------------------------------------------- Collector (live web data)
COLLECT_PROMPT = """Search the web for the latest news (last 30 days) on shipping delays, port congestion,
tariffs, export controls or sanctions that would affect a US drone company sourcing electronics,
motors, batteries or sensors from these countries: {countries}.
For each country give the single most relevant current item.
Return ONLY JSON: {{"signals":[{{"country":"...","risk":"low|medium|high","headline":"short summary","source":"url","date":"YYYY-MM-DD or empty"}}]}}"""


def collect_signals(countries):
    """Live data: uses OpenAI's web search tool. Returns (signals, error_or_None)."""
    c = _client()
    if not c:
        return [], "No API key set."
    prompt = COLLECT_PROMPT.format(countries=", ".join(countries))
    last_err = None
    for tool in ("web_search", "web_search_preview"):
        try:
            r = c.responses.create(model=MODEL, tools=[{"type": tool}], input=prompt)
            data = _parse_json(getattr(r, "output_text", "") or "")
            if data and data.get("signals"):
                return data["signals"], None
            last_err = "The model returned no usable data."
        except Exception as e:
            last_err = str(e)
    return [], last_err

"""A window onto a running mind: python3 -m mind dashboard [--state DIR] [--port 8090]

A separate process that only reads the mind's SQLite file (read-only; WAL lets it
read while the mind writes) and serves one page on 127.0.0.1. The only thing it
can do to the mind is drop a message into state/inbox/, which the mind perceives
as raw words like any other input.

It also reads the world's hidden rules so the page can mark which beliefs are
true. That is for the person watching: the mind itself never sees them.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mind.eval import _cause_truth
from mind.params import PARAMS

QUIET_KINDS = ("pulse",)


class Reader:
    def __init__(self, state: Path):
        self.state = state
        self.db_path = state / "mind.sqlite"
        self._cache: dict = {}

    @contextmanager
    def db(self):
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=5)
        con.row_factory = sqlite3.Row
        try:
            yield con
        finally:
            con.close()

    def kv(self, con, key, default=None):
        r = con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def cached(self, key, ttl, fn):
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        val = fn()
        self._cache[key] = (time.time(), val)
        return val

    # ---------------------------------------------------------------- now
    def now(self):
        with self.db() as con:
            ident = self.kv(con, "identity") or {}
            ck = self.kv(con, "checkpoint") or {}
            pulse = con.execute("SELECT tick, t, state, data FROM stream WHERE kind='pulse' "
                                "ORDER BY id DESC LIMIT 1").fetchone()
            last = con.execute("SELECT tick, t, state FROM stream ORDER BY id DESC LIMIT 1").fetchone()
            p = json.loads(pulse["data"]) if pulse else {}
            tick = max(pulse["tick"] if pulse else 0, last["tick"] if last else 0, ck.get("tick", 0))
            state = (last["state"] if last and last["state"] else None) or ck.get("state")
            percepts = con.execute("SELECT source, concept, surprise FROM percepts WHERE tick=(SELECT max(tick) "
                                   "FROM percepts) ORDER BY source, concept").fetchall()
            thoughts = con.execute("SELECT id, focus, step, status, created, updated FROM thoughts "
                                   "WHERE status IN ('active','suspended') ORDER BY updated DESC LIMIT 8").fetchall()
            params = (self.kv(con, "params") or {}).get("values", {})
            interests = sorted(((r[0], c) for c, r in (ck.get("interests") or {}).items()), reverse=True)[:10]
            return dict(
                id=ident.get("id"), born=ident.get("born"), boots=self.kv(con, "boots"), tick=tick, state=state,
                last_t=last["t"] if last else None, server_t=time.time(),
                arousal=p.get("a", ck.get("arousal")), fatigue=p.get("f", ck.get("fatigue")),
                curiosity=p.get("c", ck.get("curiosity")), surprise=p.get("s"),
                workspace=p.get("ws", ck.get("workspace") or {}),
                percepts=[dict(source=r["source"], concept=r["concept"], surprise=r["surprise"]) for r in percepts],
                thoughts=[dict(r) for r in thoughts],
                interests=[dict(concept=c, strength=round(s, 4)) for s, c in interests if s > 0],
                params=[dict(name=k, value=params.get(k, v[0]), start=v[0], lo=v[1], hi=v[2]) for k, v in PARAMS.items()],
                notes=(self.kv(con, "notes") or [])[-3:],
                pulses=pulse is not None)

    # ---------------------------------------------------------------- timeline
    def pulses(self, seconds=3600, points=600):
        with self.db() as con:
            end = con.execute("SELECT max(t) FROM stream").fetchone()[0] or time.time()
            start = end - seconds
            rows = con.execute("SELECT tick, t, state, data FROM stream WHERE kind='pulse' AND t >= ? "
                               "ORDER BY id", (start,)).fetchall()
            if not rows:   # minds born before pulses existed: state changes only
                rows = con.execute("SELECT tick, t, state, data FROM stream WHERE kind='state' AND t >= ? "
                                   "ORDER BY id", (start,)).fetchall()
        bands = []
        for r in rows:
            if bands and bands[-1][2] == r["state"]:
                bands[-1][1] = r["t"]
            else:
                if bands:
                    bands[-1][1] = r["t"]
                bands.append([r["t"], r["t"], r["state"]])
        if bands:
            bands[-1][1] = end
        step = max(1, len(rows) // points)
        pts = []
        for i in range(0, len(rows), step):
            chunk = [json.loads(r["data"]) for r in rows[i:i + step]]
            def mean(key):
                v = [c[key] for c in chunk if c.get(key) is not None]
                return round(sum(v) / len(v), 3) if v else None
            pts.append(dict(t=rows[i]["t"], tick=rows[i]["tick"], a=mean("a"), f=mean("f"), s=mean("s"),
                            c=mean("c")))
        return dict(start=start, end=end, bands=bands, points=pts)

    # ---------------------------------------------------------------- life stream
    def stream(self, after=0, limit=150, hide=()):
        hide = tuple(QUIET_KINDS) + tuple(hide)
        q = (f"SELECT id, tick, t, kind, state, data FROM stream WHERE id > ? AND kind NOT IN "
             f"({','.join('?' * len(hide))}) ORDER BY id DESC LIMIT ?")
        with self.db() as con:
            rows = con.execute(q, (after, *hide, limit)).fetchall()
        return [dict(id=r["id"], tick=r["tick"], t=r["t"], kind=r["kind"], state=r["state"],
                     data=json.loads(r["data"])) for r in reversed(rows)]

    # ---------------------------------------------------------------- what it knows
    def truth(self):
        def load():
            p = self.state / "world" / "rules.json"
            try:
                rules = json.loads(p.read_text())["rules"]
            except (OSError, ValueError, KeyError):
                return None
            causes = {f"{a}\t{b}": sorted(l) for (a, b), l in _cause_truth(rules).items()}
            cycles = {r["feature"]: r["period"] for r in rules if r["kind"] == "cycle"}
            return dict(causes=causes, cycles=cycles)
        return self.cached("truth", 300, load)

    def beliefs(self):
        truth = self.truth()
        with self.db() as con:
            rows = [dict(r) for r in con.execute(
                "SELECT kind, a, b, lag, confidence, support, origin, status, created, updated FROM beliefs "
                "WHERE status != 'rejected' ORDER BY (status='belief') DESC, confidence DESC LIMIT 200")]
            counts = dict(con.execute("SELECT status, count(*) FROM beliefs GROUP BY status").fetchall())
        for r in rows:
            if truth is None:
                r["true"] = None
            elif r["kind"] == "cause":
                r["true"] = r["lag"] in truth["causes"].get(f"{r['a']}\t{r['b']}", [])
            else:
                per = truth["cycles"].get(r["a"].split("=")[0])
                r["true"] = per is not None and abs(per - r["lag"]) <= 1
        return dict(rows=rows, counts=counts, hidden_rules=None if truth is None else
                    dict(causes=len(truth["causes"]), cycles=len(truth["cycles"])))

    def dreams(self):
        def load():
            out = {}
            with self.db() as con:
                for kind in ("dream", "dream_control", "dream_llm"):
                    r = con.execute("SELECT count(*), sum(json_extract(data, '$.passed')) FROM stream "
                                    "WHERE kind=?", (kind,)).fetchone()
                    out[kind] = dict(n=r[0], passed=r[1] or 0)
                recent = con.execute("SELECT tick, t, kind, data FROM stream WHERE kind IN ('dream','dream_control')"
                                     " ORDER BY id DESC LIMIT 24").fetchall()
            out["recent"] = [dict(tick=r["tick"], t=r["t"], kind=r["kind"], **json.loads(r["data"])) for r in recent]
            return out
        return self.cached("dreams", 10, load)

    def message(self, text: str):
        text = text.strip()[:2000]
        if not text:
            return False
        d = self.state / "inbox"
        d.mkdir(parents=True, exist_ok=True)
        name = f"{time.time():.3f}"
        tmp = d / (name + ".tmp")
        tmp.write_text(text)
        tmp.rename(d / (name + ".txt"))   # appears whole: the mind never reads half a message
        return True


def handler(reader: Reader):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                if not reader.db_path.exists():
                    return self._json({"error": "no mind has been born here"}, 404)
                if u.path == "/api/now":
                    return self._json(reader.now())
                if u.path == "/api/pulses":
                    return self._json(reader.pulses(int(q.get("seconds", 3600)), int(q.get("points", 600))))
                if u.path == "/api/stream":
                    hide = tuple(x for x in q.get("hide", "").split(",") if x)
                    return self._json(reader.stream(int(q.get("after", 0)), min(500, int(q.get("limit", 150))), hide))
                if u.path == "/api/beliefs":
                    return self._json(reader.beliefs())
                if u.path == "/api/dreams":
                    return self._json(reader.dreams())
                return self._json({"error": "not found"}, 404)
            except (sqlite3.Error, ValueError) as e:
                return self._json({"error": f"{type(e).__name__}: {e}"}, 503)

        def do_POST(self):
            if urlparse(self.path).path != "/api/message":
                return self._json({"error": "not found"}, 404)
            n = int(self.headers.get("Content-Length") or 0)
            if n > 8192:
                return self._json({"error": "too long"}, 413)
            try:
                text = json.loads(self.rfile.read(n) or b"{}").get("text", "")
            except ValueError:
                return self._json({"error": "bad json"}, 400)
            return self._json({"ok": reader.message(str(text))})

    return H


def serve(state: Path, port=8090, host="127.0.0.1"):
    srv = ThreadingHTTPServer((host, port), handler(Reader(state)))
    print(f"dashboard for {state} on http://{host}:{srv.server_port}", flush=True)
    return srv


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mind</title>
<style>
:root{--bg:#f6f5f1;--panel:#fff;--ink:#1d1d1b;--mute:#6b6a64;--line:#e3e1da;--accent:#2f6fde;
--awake:#e8a33a;--background:#8aa0b8;--resting:#5fae7b;--dreaming:#9b6fd6;--deep_sleep:#3d4a8f;
--ar:#e07a2f;--fa:#c2414b;--su:#178a8a;--ok:#2e8b57;--bad:#b4434b}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1e1e1c;--ink:#ecebe6;--mute:#9b9a93;--line:#33322f;
--accent:#79a6ff;--background:#62748a;--deep_sleep:#6876c9;--ar:#f39a55;--fa:#e26a73;--su:#3cbcbc;--ok:#5cc28a;--bad:#e27a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:16px 20px 8px;display:flex;flex-wrap:wrap;gap:8px 20px;align-items:baseline}
h1{font-size:18px;margin:0;font-weight:650}h2{font-size:12px;margin:0 0 8px;color:var(--mute);font-weight:600;text-transform:uppercase;letter-spacing:.06em}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
.mute{color:var(--mute)}.chip{display:inline-block;padding:2px 10px;border-radius:99px;color:#fff;font-weight:600;font-size:13px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}
main{display:grid;grid-template-columns:repeat(12,1fr);gap:12px;padding:8px 20px 24px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0}
.c12{grid-column:span 12}.c8{grid-column:span 8}.c6{grid-column:span 6}.c4{grid-column:span 4}
@media (max-width:900px){.c8,.c6,.c4{grid-column:span 12}}
.meter{display:grid;grid-template-columns:80px 1fr 44px;gap:8px;align-items:center;margin:4px 0}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar>i{display:block;height:100%;border-radius:4px}
.row{display:grid;grid-template-columns:1fr 70px;gap:8px;align-items:center;margin:3px 0}
#feed{max-height:460px;overflow:auto}.ev{display:grid;grid-template-columns:64px 1fr;gap:8px;padding:3px 0;border-bottom:1px solid var(--line)}
.ev .k{color:var(--mute)}.ev.dim{opacity:.55}
table{width:100%;border-collapse:collapse}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--mute);font-size:12px}td.n{text-align:right;font-variant-numeric:tabular-nums}
.tabs button,.btn{background:none;border:1px solid var(--line);color:var(--ink);border-radius:6px;padding:3px 9px;cursor:pointer;font:inherit;font-size:12.5px}
.tabs button.on{background:var(--ink);color:var(--bg)}
svg text{fill:var(--mute);font-size:11px}.legend span{margin-right:12px;font-size:12px}
textarea{width:100%;min-height:60px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:8px;font:inherit}
.sym{font-family:ui-monospace,Menlo,monospace;font-size:12.5px}.ok{color:var(--ok)}.bad{color:var(--bad)}
.flex{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:space-between}
</style></head><body>
<header><h1 id="title">Mind</h1><span id="state"></span><span id="alive" class="mute"></span><span id="meta" class="mute"></span></header>
<main>
<section class="card c12"><div class="flex"><h2>Timeline</h2>
<div class="tabs" id="range"><button data-s="600">10 min</button><button data-s="3600" class="on">1 h</button><button data-s="21600">6 h</button><button data-s="86400">24 h</button><button data-s="604800">7 d</button></div></div>
<div id="chart"></div>
<div class="legend mute"><span><i class="dot" style="background:var(--ar)"></i>arousal</span><span><i class="dot" style="background:var(--fa)"></i>fatigue (sleep pressure)</span><span><i class="dot" style="background:var(--su)"></i>surprise of the world (bits, scaled)</span><span id="bandlegend"></span></div></section>

<section class="card c4"><h2>Drives</h2><div id="drives"></div><h2 style="margin-top:14px">Perceiving now</h2><div id="percepts" class="mono"></div></section>
<section class="card c4"><h2>Workspace (what holds its attention)</h2><div id="ws"></div><h2 style="margin-top:14px">Thoughts</h2><div id="thoughts"></div></section>
<section class="card c4"><h2>Interests (where it is learning)</h2><div id="interests"></div><h2 style="margin-top:14px">Settings it tuned itself</h2><div id="params"></div></section>

<section class="card c8"><div class="flex"><h2>Life stream</h2><label class="mute"><input type="checkbox" id="showattend"> show every “noticed”</label></div><div id="feed"></div></section>
<section class="card c4"><h2>Dreams</h2><div id="dreamstats"></div><div id="dreams" class="mono" style="max-height:330px;overflow:auto"></div></section>

<section class="card c8"><h2>What it believes</h2><div id="bcounts" class="mute" style="margin-bottom:6px"></div><div style="max-height:420px;overflow:auto"><table id="beliefs"></table></div></section>
<section class="card c4"><h2>Say something</h2><p class="mute" style="margin-top:0">It arrives as plain words among its other percepts. Nothing tells it what words mean or that they matter.</p>
<textarea id="msg" maxlength="2000"></textarea><div class="flex" style="margin-top:6px"><span id="msgstate" class="mute"></span><button class="btn" id="send">Send</button></div>
<div id="notes"></div></section>
</main>
<script>
const $=id=>document.getElementById(id), esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const STATES=["awake","background","resting","dreaming","deep_sleep"];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue("--"+n).trim();
const label=s=>(s||"").replace("_"," ");
const sym=c=>{c=String(c??"");if(c.startsWith("missing:"))return `<span class="sym">${esc(c.slice(8))}</span> <span class="mute">(expected, absent)</span>`;return `<span class="sym">${esc(c)}</span>`};
let range=3600,lastId=0,feed=[];
function ago(s){s=Math.max(0,s);if(s<90)return Math.round(s)+" s";if(s<5400)return Math.round(s/60)+" min";if(s<172800)return (s/3600).toFixed(1)+" h";return (s/86400).toFixed(1)+" days"}
function hhmm(t){const d=new Date(t*1000);return d.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"})}
function meter(name,v,color,max=1){v=v??0;const p=Math.max(0,Math.min(1,v/max))*100;return `<div class="meter"><span>${name}</span><div class="bar"><i style="width:${p}%;background:${color}"></i></div><span class="mono">${(+v).toFixed(2)}</span></div>`}
async function j(u,o){const r=await fetch(u,o);return r.json()}

async function now(){
  const n=await j("/api/now");if(n.error){$("title").textContent=n.error;return}
  $("title").innerHTML=`Mind <span class="mono mute">${esc((n.id||"").slice(0,8))}</span>`;
  $("state").innerHTML=`<span class="chip" style="background:${css(n.state)}">${label(n.state)}</span>`;
  const age=(n.server_t-Date.parse(n.born)/1000), quiet=n.server_t-n.last_t;
  $("alive").innerHTML=quiet<30?`<i class="dot" style="background:var(--ok)"></i>alive`:`<i class="dot" style="background:var(--bad)"></i>silent for ${ago(quiet)}`;
  $("meta").textContent=`tick ${n.tick} · age ${ago(age)} · boot ${n.boots}`;
  $("drives").innerHTML=meter("arousal",n.arousal,css("ar"))+meter("fatigue",n.fatigue,css("fa"))+meter("curiosity",n.curiosity,css("su"),4)+(n.surprise!=null?meter("surprise",n.surprise,css("su"),8):"");
  const ws=Object.entries(n.workspace||{});
  $("ws").innerHTML=ws.length?ws.map(([k,v])=>`<div class="row"><div>${sym(k)}<div class="bar"><i style="width:${Math.min(1,v)*100}%;background:var(--accent)"></i></div></div><span class="mono mute">${(+v).toFixed(2)}</span></div>`).join(""):`<span class="mute">empty</span>`;
  $("thoughts").innerHTML=n.thoughts.length?n.thoughts.map(t=>`<div>${sym(t.focus)} <span class="mute">step ${t.step} · ${t.status} · since tick ${t.created}</span></div>`).join(""):`<span class="mute">none open</span>`;
  const by={};n.percepts.forEach(p=>(by[p.source]=by[p.source]||[]).push(p));
  $("percepts").innerHTML=Object.entries(by).map(([s,ps])=>`<div><span class="mute">${esc(s)}</span> ${ps.map(p=>`<span title="surprise ${p.surprise} bits" style="opacity:${0.45+Math.min(1,p.surprise/6)*0.55}">${esc(p.concept)}</span>`).join(" ")}</div>`).join("")||`<span class="mute">nothing</span>`;
  const mx=Math.max(1e-9,...n.interests.map(i=>i.strength));
  $("interests").innerHTML=n.interests.length?n.interests.map(i=>`<div class="row"><div>${sym(i.concept)}<div class="bar"><i style="width:${i.strength/mx*100}%;background:var(--su)"></i></div></div><span class="mono mute">${i.strength.toFixed(3)}</span></div>`).join(""):`<span class="mute">none yet</span>`;
  $("params").innerHTML=`<table>${n.params.map(p=>{const d=p.value-p.start;return `<tr><td class="mono">${p.name}</td><td class="n mono">${+(+p.value).toFixed(3)}</td><td class="n mute mono">${d? (d>0?"+":"")+(+d.toFixed(3)):"·"}</td></tr>`}).join("")}</table>`;
  $("notes").innerHTML=n.notes.length?`<h2 style="margin-top:14px">Its notes</h2>`+n.notes.map(x=>`<p>${esc(x)}</p>`).join(""):"";
}

async function chart(){
  const d=await j(`/api/pulses?seconds=${range}&points=600`);if(d.error)return;
  const W=$("chart").clientWidth||800,H=170,pad=22,x=t=>pad+(t-d.start)/(d.end-d.start||1)*(W-2*pad),y=v=>H-18-(Math.max(0,Math.min(1,v)))*(H-30);
  let s=`<svg width="${W}" height="${H}" role="img" aria-label="state and drive timeline">`;
  for(const [a,b,st] of d.bands)s+=`<rect x="${x(a)}" y="6" width="${Math.max(1,x(b)-x(a))}" height="${H-24}" fill="${css(st)}" opacity=".22"><title>${label(st)} ${hhmm(a)}–${hhmm(b)}</title></rect>`;
  const line=(k,c,sc=1)=>{const p=d.points.filter(q=>q[k]!=null).map(q=>`${x(q.t).toFixed(1)},${y(q[k]/sc).toFixed(1)}`);return p.length?`<polyline fill="none" stroke="${c}" stroke-width="1.6" points="${p.join(" ")}"/>`:""};
  s+=line("s",css("su"),8)+line("f",css("fa"))+line("a",css("ar"));
  for(let i=0;i<=4;i++){const t=d.start+(d.end-d.start)*i/4;s+=`<text x="${x(t)}" y="${H-3}" text-anchor="${i==0?"start":i==4?"end":"middle"}">${range>86400?new Date(t*1000).toLocaleDateString([], {weekday:"short"})+" ":""}${hhmm(t)}</text>`}
  if(!d.points.length)s+=`<text x="${W/2}" y="${H/2}" text-anchor="middle">${d.bands.length?"drives are recorded from the next restart on; states shown":"nothing in this range"}</text>`;
  $("chart").innerHTML=s+"</svg>";
  $("bandlegend").innerHTML=STATES.map(st=>`<span><i class="dot" style="background:${css(st)};opacity:.6"></i>${label(st)}</span>`).join("");
}

function describe(e){
  const d=e.data||{},L=x=>sym(x);
  switch(e.kind){
    case "birth":return `<b>born</b> <span class="mono mute">${esc(d.id)}</span>`;
    case "boot":return `woke up after a restart (boot ${d.boot}${d.offline_s!=null?`, offline ${ago(d.offline_s)}`:""})`;
    case "shutdown":return `stopped`; case "stop_requested":return `asked to stop`;
    case "state":return `${label(d.old)} → <b>${label(e.state)}</b> <span class="mute">arousal ${d.arousal} · fatigue ${d.fatigue}</span>`;
    case "attend":return `noticed ${L(d.focus)} <span class="mute">${d.activation}</span>`;
    case "thought_start":return `started thinking about ${L(d.focus)}`;
    case "thought_end":return `finished thinking about ${L(d.focus)}: <b>${esc(d.outcome)}</b>`;
    case "hypothesis":return `suspects ${L(d.a)} → ${L(d.b)} after ${d.lag} <span class="mute">conf ${d.conf} · from ${esc(d.origin)}</span>`;
    case "belief_belief":return `<b>now believes</b> ${L(d.a)} → ${L(d.b)} after ${d.lag} <span class="mute">held ${Math.round(d.conf*100)}% of ${d.n} times since</span>`;
    case "belief_doubted":return `<b>doubts</b> ${L(d.a)} → ${L(d.b)} after ${d.lag} <span class="mute">conf ${d.conf}</span>`;
    case "belief_rejected":return `dropped ${L(d.a)} → ${L(d.b)} after ${d.lag}`;
    case "belief_hypothesis":return `back to a hunch: ${L(d.a)} → ${L(d.b)} after ${d.lag}`;
    case "belief_new":return `<b>believes</b> ${L(d.concept)} comes round every ${d.period} ticks <span class="mute">from ${esc(d.origin)}</span>`;
    case "dream":return `dreamt ${L(d.a)} ~ ${L(d.b)} <span class="${d.passed?"ok":"mute"}">${d.passed?"held up in memory":"nothing"} · lift ${d.lift}</span>`;
    case "dream_control":return `<span class="mute">control pair ${esc(d.a)} ~ ${esc(d.b)}: ${d.passed?"passed":"nothing"}</span>`;
    case "consolidate":return `deep sleep: forgot ${d.forgotten} dull memories <span class="mute">${d.edges} links · ${d.trackers} trackers</span>`;
    case "self_tune":{const t=d.trying||{};return `tuned itself${d.param?`: ${esc(d.param)} ${d.old}→${d.new} ${d.kept?"<span class=ok>kept</span>":"<span class=bad>undone</span>"}`:""}; now trying ${esc(t.param)} ${t.old}→${t.new} <span class="mute">surprise ${d.mean_surprise}</span>`}
    case "tier_call":return `<span class="mute">asked the ${esc(d.tier)} model (${esc(d.kind)}) · ${d.seconds}s${d.error?" · "+esc(d.error):""}</span>`;
    case "reflection":return `note: “${esc(d.note)}”`;
    default:return `${esc(e.kind)} <span class="mute mono">${esc(JSON.stringify(d)).slice(0,160)}</span>`;
  }
}
async function stream(){
  const hide=$("showattend").checked?"":"attend,dream_control";
  const ev=await j(`/api/stream?after=${lastId}&limit=150&hide=${hide}`);if(ev.error)return;
  if(ev.length){lastId=ev[ev.length-1].id;feed=feed.concat(ev).slice(-300)}
  const box=$("feed"),stick=box.scrollTop<8;
  box.innerHTML=feed.slice().reverse().map(e=>`<div class="ev ${e.kind=="attend"||e.kind=="dream_control"?"dim":""}"><span class="k mono" title="${new Date(e.t*1000).toLocaleString()}">${e.tick}</span><span>${describe(e)}</span></div>`).join("");
  if(stick)box.scrollTop=0;
}
async function beliefs(){
  const b=await j("/api/beliefs");if(b.error)return;
  const c=b.counts;$("bcounts").textContent=`${c.belief||0} beliefs · ${c.hypothesis||0} hunches · ${c.doubted||0} doubted · ${c.rejected||0} dropped`+(b.hidden_rules?` · the hidden world has ${b.hidden_rules.causes} causes and ${b.hidden_rules.cycles} cycles (✓ = matches one; it never sees these)`:"");
  $("beliefs").innerHTML=`<tr><th>status</th><th>relation</th><th>conf</th><th>seen</th><th>origin</th><th>✓</th></tr>`+b.rows.map(r=>`<tr><td>${esc(r.status)}</td><td>${r.kind=="cause"?`${sym(r.a)} → ${sym(r.b)} <span class="mute">after ${r.lag}</span>`:`${sym(r.a)} <span class="mute">every ${r.lag}</span>`}</td><td class="n">${(+r.confidence).toFixed(2)}</td><td class="n">${r.support}</td><td class="mute">${esc(r.origin)}</td><td>${r.true==null?"":r.true?"<span class=ok>✓</span>":"<span class=mute>·</span>"}</td></tr>`).join("");
}
async function dreams(){
  const d=await j("/api/dreams");if(d.error)return;
  const pct=x=>x.n?`${(100*x.passed/x.n).toFixed(1)}% of ${x.n}`:"none yet";
  $("dreamstats").innerHTML=`<div>dreams that held up: <b>${pct(d.dream)}</b></div><div class="mute">random pairs (control): ${pct(d.dream_control)}</div>${d.dream_llm.n?`<div class="mute">language-model dreams: ${pct(d.dream_llm)}</div>`:""}<div style="height:8px"></div>`;
  $("dreams").innerHTML=d.recent.map(x=>`<div class="${x.kind=="dream_control"?"mute":""}">${x.passed?"<span class=ok>●</span>":"○"} ${esc(x.a)} ~ ${esc(x.b)} <span class="mute">${x.kind=="dream_control"?"control":"lag "+x.lag}</span></div>`).join("")||`<span class="mute">it has not dreamt yet</span>`;
}
$("range").onclick=e=>{const b=e.target.closest("button");if(!b)return;range=+b.dataset.s;[...$("range").children].forEach(x=>x.classList.toggle("on",x==b));chart()};
$("showattend").onchange=()=>{lastId=0;feed=[];stream()};
$("send").onclick=async()=>{const t=$("msg").value.trim();if(!t)return;const r=await j("/api/message",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:t})});$("msgstate").textContent=r.ok?"delivered to its inbox":"not delivered";if(r.ok)$("msg").value=""};
const safe=f=>()=>f().catch(()=>{});
[now,stream,chart,beliefs,dreams].forEach(f=>safe(f)());
setInterval(safe(now),2000);setInterval(safe(stream),2500);setInterval(safe(chart),10000);setInterval(safe(beliefs),15000);setInterval(safe(dreams),10000);
addEventListener("resize",safe(chart));
</script></body></html>"""

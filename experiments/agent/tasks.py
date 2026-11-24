"""C0 tasks: small "build X with tests" ideas for a coding agent.

Each task has the spec the agent sees (with the module and function names the hidden tests
import), hidden acceptance tests the agent never sees, and a reference solution used only by
`agent_bench.py --fake` to check the harness and the hidden tests themselves.
"""

TASKS = [
    {
        "id": "roman",
        "level": 1,
        "spec": """Write `roman.py` with two functions:
- `to_roman(n: int) -> str` converts 1..3999 to a Roman numeral (e.g. 1994 -> "MCMXCIV").
- `from_roman(s: str) -> int` converts back, accepting upper case only.
Both raise ValueError for anything out of range or not a valid Roman numeral (e.g. 0, 4000, "IIII", "IC", "").""",
        "hidden": """
import unittest
from roman import to_roman, from_roman

class T(unittest.TestCase):
    def test_examples(self):
        for n, s in [(1, "I"), (4, "IV"), (9, "IX"), (14, "XIV"), (40, "XL"), (90, "XC"),
                     (400, "CD"), (1994, "MCMXCIV"), (2024, "MMXXIV"), (3999, "MMMCMXCIX")]:
            self.assertEqual(to_roman(n), s)
            self.assertEqual(from_roman(s), n)
    def test_round_trip(self):
        for n in range(1, 4000):
            self.assertEqual(from_roman(to_roman(n)), n)
    def test_errors(self):
        for n in (0, 4000, -1):
            with self.assertRaises(ValueError):
                to_roman(n)
        for s in ("", "IIII", "IC", "VV", "ABC", "MMMM"):
            with self.assertRaises(ValueError):
                from_roman(s)
""",
        "reference": {"roman.py": """
VALS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
        (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]

def to_roman(n):
    if not isinstance(n, int) or not 1 <= n <= 3999:
        raise ValueError(n)
    out = ""
    for v, s in VALS:
        while n >= v:
            out += s
            n -= v
    return out

def from_roman(s):
    total, i = 0, 0
    for v, sym in VALS:
        while s.startswith(sym, i):
            total += v
            i += len(sym)
    if i != len(s) or not 1 <= total <= 3999 or to_roman(total) != s:
        raise ValueError(s)
    return total
"""},
    },
    {
        "id": "lru",
        "level": 1,
        "spec": """Write `lru.py` with a class `LRUCache(capacity: int)`:
- `get(key)` returns the value, or None if missing, and marks the key as recently used.
- `put(key, value)` inserts or updates; when over capacity, evicts the least recently used key.
- `len(cache)` gives the number of items.
capacity < 1 raises ValueError.""",
        "hidden": """
import unittest
from lru import LRUCache

class T(unittest.TestCase):
    def test_basic(self):
        c = LRUCache(2)
        c.put("a", 1); c.put("b", 2)
        self.assertEqual(c.get("a"), 1)
        c.put("c", 3)
        self.assertIsNone(c.get("b"))
        self.assertEqual(c.get("a"), 1)
        self.assertEqual(c.get("c"), 3)
        self.assertEqual(len(c), 2)
    def test_update_refreshes(self):
        c = LRUCache(2)
        c.put("a", 1); c.put("b", 2); c.put("a", 10); c.put("c", 3)
        self.assertEqual(c.get("a"), 10)
        self.assertIsNone(c.get("b"))
    def test_missing_and_bad(self):
        self.assertIsNone(LRUCache(1).get("x"))
        with self.assertRaises(ValueError):
            LRUCache(0)
""",
        "reference": {"lru.py": """
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError(capacity)
        self.capacity = capacity
        self.d = OrderedDict()
    def get(self, key):
        if key not in self.d:
            return None
        self.d.move_to_end(key)
        return self.d[key]
    def put(self, key, value):
        self.d[key] = value
        self.d.move_to_end(key)
        while len(self.d) > self.capacity:
            self.d.popitem(last=False)
    def __len__(self):
        return len(self.d)
"""},
    },
    {
        "id": "csv_stats",
        "level": 2,
        "spec": """Write `stats.py` with `summarize(csv_text: str) -> dict`.
The text is CSV with a header row. For every column whose non-empty values are all numbers,
return `{column: {"count": n, "mean": m, "min": lo, "max": hi}}` (floats; empty cells are skipped,
count is the number of non-empty cells). Columns with any non-numeric value are left out.
Empty input or a header with no rows returns {}.""",
        "hidden": """
import unittest
from stats import summarize

class T(unittest.TestCase):
    def test_mixed(self):
        text = "name,age,score\\nann,30,7.5\\nbob,,9\\ncat,40,8.5\\n"
        r = summarize(text)
        self.assertEqual(set(r), {"age", "score"})
        self.assertEqual(r["age"]["count"], 2)
        self.assertAlmostEqual(r["age"]["mean"], 35.0)
        self.assertEqual(r["age"]["min"], 30.0)
        self.assertEqual(r["age"]["max"], 40.0)
        self.assertAlmostEqual(r["score"]["mean"], 25 / 3)
    def test_quoted_and_negative(self):
        text = 'label,value\\n"a, b",-2\\nc,4\\n'
        r = summarize(text)
        self.assertEqual(list(r), ["value"])
        self.assertAlmostEqual(r["value"]["mean"], 1.0)
    def test_empty(self):
        self.assertEqual(summarize(""), {})
        self.assertEqual(summarize("a,b\\n"), {})
""",
        "reference": {"stats.py": """
import csv, io

def summarize(csv_text):
    rows = list(csv.reader(io.StringIO(csv_text)))
    if len(rows) < 2:
        return {}
    head, body = rows[0], rows[1:]
    out = {}
    for i, col in enumerate(head):
        vals = [r[i] for r in body if i < len(r) and r[i].strip() != ""]
        try:
            nums = [float(v) for v in vals]
        except ValueError:
            continue
        if nums:
            out[col] = {"count": len(nums), "mean": sum(nums) / len(nums), "min": min(nums), "max": max(nums)}
    return out
"""},
    },
    {
        "id": "calc",
        "level": 2,
        "spec": """Write `calc.py` with `evaluate(expr: str) -> float` for arithmetic expressions with
+ - * /, parentheses, unary minus, decimals and spaces, with normal precedence (left to right for
equal precedence). Do not use eval or exec. Raise ValueError for malformed input (e.g. "1 +",
"(2", "2 3", "", "a+1") and ZeroDivisionError for division by zero.""",
        "hidden": """
import unittest
from calc import evaluate

class T(unittest.TestCase):
    def test_values(self):
        for e, v in [("1+2*3", 7), ("(1+2)*3", 9), ("10/4", 2.5), ("-3+5", 2), ("2*-3", -6),
                     ("8-3-2", 3), ("16/4/2", 2), (" 1.5 * ( 2 + 2 ) ", 6), ("-(2+3)*2", -10), ("42", 42)]:
            self.assertAlmostEqual(evaluate(e), v, msg=e)
    def test_errors(self):
        for e in ["1 +", "(2", "2 3", "", "a+1", "2)"]:
            with self.assertRaises(ValueError, msg=e):
                evaluate(e)
        with self.assertRaises(ZeroDivisionError):
            evaluate("1/(2-2)")
    def test_no_eval(self):
        import calc, inspect
        src = inspect.getsource(calc)
        self.assertNotIn("eval(", src.replace("evaluate(", ""))
""",
        "reference": {"calc.py": """
import re

TOK = re.compile(r"\\s*(?:(\\d+\\.?\\d*|\\.\\d+)|(.))")

def evaluate(expr):
    toks = []
    for num, op in TOK.findall(expr):
        if num:
            toks.append(float(num))
        elif op.strip():
            if op not in "+-*/()":
                raise ValueError(op)
            toks.append(op)
    pos = [0]
    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None
    def take():
        t = peek(); pos[0] += 1; return t
    def expr_():
        v = term()
        while peek() in ("+", "-"):
            v = v + term() if take() == "+" else v - term()
        return v
    def term():
        v = factor()
        while peek() in ("*", "/"):
            if take() == "*":
                v *= factor()
            else:
                d = factor()
                if d == 0:
                    raise ZeroDivisionError
                v /= d
        return v
    def factor():
        t = take()
        if t == "-":
            return -factor()
        if t == "(":
            v = expr_()
            if take() != ")":
                raise ValueError("missing )")
            return v
        if isinstance(t, float):
            return t
        raise ValueError(t)
    if not toks:
        raise ValueError("empty")
    v = expr_()
    if pos[0] != len(toks):
        raise ValueError("trailing")
    return v
"""},
    },
    {
        "id": "todo_sqlite",
        "level": 2,
        "spec": """Write `todo.py` with a class `TodoStore(path: str)` backed by sqlite3 (use ":memory:" in tests):
- `add(title: str, due: str | None = None) -> int` returns the new id; due is "YYYY-MM-DD" or None; empty title raises ValueError.
- `complete(todo_id: int)`; an unknown id raises KeyError.
- `open_items() -> list[dict]` with keys id, title, due, done, sorted by due date (None last), then id.
- `search(text: str) -> list[dict]` case-insensitive substring match on title, any state, sorted by id.
Data must survive reopening the same file path.""",
        "hidden": """
import os, tempfile, unittest
from todo import TodoStore

class T(unittest.TestCase):
    def test_flow(self):
        s = TodoStore(":memory:")
        a = s.add("Buy milk", "2026-12-02")
        b = s.add("Call Bob")
        c = s.add("Pay rent", "2026-12-01")
        self.assertEqual([t["id"] for t in s.open_items()], [c, a, b])
        s.complete(c)
        self.assertEqual([t["title"] for t in s.open_items()], ["Buy milk", "Call Bob"])
        self.assertEqual([t["id"] for t in s.search("BOB")], [b])
        self.assertEqual(len(s.search("PAY")), 1)
        self.assertTrue(s.search("pay")[0]["done"])
    def test_errors(self):
        s = TodoStore(":memory:")
        with self.assertRaises(ValueError):
            s.add("")
        with self.assertRaises(KeyError):
            s.complete(99)
    def test_persists(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            TodoStore(p).add("Keep me")
            self.assertEqual(TodoStore(p).open_items()[0]["title"], "Keep me")
""",
        "reference": {"todo.py": """
import sqlite3

class TodoStore:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("create table if not exists todo (id integer primary key, title text, due text, done int default 0)")
        self.db.commit()
    def _rows(self, sql, args=()):
        return [{"id": r[0], "title": r[1], "due": r[2], "done": bool(r[3])} for r in self.db.execute(sql, args)]
    def add(self, title, due=None):
        if not title.strip():
            raise ValueError("empty title")
        cur = self.db.execute("insert into todo (title, due) values (?, ?)", (title, due))
        self.db.commit()
        return cur.lastrowid
    def complete(self, todo_id):
        cur = self.db.execute("update todo set done = 1 where id = ?", (todo_id,))
        self.db.commit()
        if cur.rowcount == 0:
            raise KeyError(todo_id)
    def open_items(self):
        return self._rows("select * from todo where done = 0 order by due is null, due, id")
    def search(self, text):
        return [r for r in self._rows("select * from todo order by id") if text.lower() in r["title"].lower()]
"""},
    },
    {
        "id": "adventure",
        "level": 3,
        "spec": """Write `adventure.py` with a class `Game(world: dict, start: str)` for a tiny text adventure.
`world` maps room names to {"exits": {direction: room}, "items": [names]}. Methods:
- `command(text: str) -> str` handles "go <dir>", "take <item>", "drop <item>", "look", "inventory" (case-insensitive, extra spaces ok).
  - go: moves and returns the new room's look text; unknown direction returns "You can't go that way."
  - take/drop move an item between the room and the inventory, returning "Taken." / "Dropped." or "No <item> here." / "You don't have <item>."
  - look returns "<room>. Exits: <dirs sorted, comma+space>. Items: <items sorted, comma+space or 'none'>."
  - inventory returns "You carry: <sorted items, comma+space>." or "You carry nothing."
  - anything else returns "I don't understand."
- attributes `room` (current room name) and `inventory` (list).
The game must not change the `world` dict passed in.""",
        "hidden": """
import copy, unittest
from adventure import Game

W = {"hall": {"exits": {"north": "library", "east": "kitchen"}, "items": ["lamp"]},
     "library": {"exits": {"south": "hall"}, "items": ["book", "map"]},
     "kitchen": {"exits": {"west": "hall"}, "items": []}}

class T(unittest.TestCase):
    def test_look_and_move(self):
        g = Game(W, "hall")
        self.assertEqual(g.command("look"), "hall. Exits: east, north. Items: lamp.")
        self.assertEqual(g.command("GO  north"), "library. Exits: south. Items: book, map.")
        self.assertEqual(g.room, "library")
        self.assertEqual(g.command("go up"), "You can't go that way.")
    def test_items(self):
        w = copy.deepcopy(W)
        g = Game(w, "hall")
        self.assertEqual(g.command("take lamp"), "Taken.")
        self.assertEqual(g.command("take lamp"), "No lamp here.")
        self.assertEqual(g.command("inventory"), "You carry: lamp.")
        g.command("go east")
        self.assertEqual(g.command("drop lamp"), "Dropped.")
        self.assertEqual(g.command("look"), "kitchen. Exits: west. Items: lamp.")
        self.assertEqual(g.command("drop lamp"), "You don't have lamp.")
        self.assertEqual(g.command("inventory"), "You carry nothing.")
        self.assertEqual(w, W)
    def test_unknown(self):
        self.assertEqual(Game(W, "hall").command("dance"), "I don't understand.")
""",
        "reference": {"adventure.py": """
import copy

class Game:
    def __init__(self, world, start):
        self.world = copy.deepcopy(world)
        self.room = start
        self.inventory = []
    def look(self):
        r = self.world[self.room]
        items = ", ".join(sorted(r["items"])) or "none"
        return f"{self.room}. Exits: {', '.join(sorted(r['exits']))}. Items: {items}."
    def command(self, text):
        words = text.lower().split()
        if words == ["look"]:
            return self.look()
        if words == ["inventory"]:
            return f"You carry: {', '.join(sorted(self.inventory))}." if self.inventory else "You carry nothing."
        if len(words) == 2:
            verb, obj = words
            r = self.world[self.room]
            if verb == "go":
                if obj not in r["exits"]:
                    return "You can't go that way."
                self.room = r["exits"][obj]
                return self.look()
            if verb == "take":
                if obj not in r["items"]:
                    return f"No {obj} here."
                r["items"].remove(obj); self.inventory.append(obj)
                return "Taken."
            if verb == "drop":
                if obj not in self.inventory:
                    return f"You don't have {obj}."
                self.inventory.remove(obj); r["items"].append(obj)
                return "Dropped."
        return "I don't understand."
"""},
    },
    {
        "id": "bank",
        "level": 3,
        "spec": """Build a small bank in two modules:
- `accounts.py`: class `Account(owner: str, balance: int = 0)` (amounts are whole pence, never floats) with
  `deposit(amount)` and `withdraw(amount)`; amount <= 0 raises ValueError; withdrawing more than the balance
  raises `InsufficientFunds` (an Exception subclass defined in accounts.py).
- `ledger.py`: class `Bank` with `open(owner: str) -> Account` (owner names are unique: a repeat raises ValueError),
  `transfer(from_owner, to_owner, amount)` which moves money atomically (on failure neither balance changes;
  unknown owners raise KeyError), and `history(owner) -> list[tuple]` of ("deposit"|"withdraw"|"transfer_in"|"transfer_out", amount)
  in order. Deposits and withdrawals made on the Account objects the bank returned must appear in the history.""",
        "hidden": """
import unittest
from accounts import Account, InsufficientFunds
from ledger import Bank

class T(unittest.TestCase):
    def test_account(self):
        a = Account("ann", 100)
        a.deposit(50); a.withdraw(30)
        self.assertEqual(a.balance, 120)
        with self.assertRaises(InsufficientFunds):
            a.withdraw(500)
        with self.assertRaises(ValueError):
            a.deposit(0)
        self.assertTrue(issubclass(InsufficientFunds, Exception))
    def test_bank(self):
        b = Bank()
        ann, bob = b.open("ann"), b.open("bob")
        ann.deposit(1000)
        b.transfer("ann", "bob", 300)
        self.assertEqual((ann.balance, bob.balance), (700, 300))
        with self.assertRaises(InsufficientFunds):
            b.transfer("bob", "ann", 301)
        self.assertEqual((ann.balance, bob.balance), (700, 300))
        with self.assertRaises(KeyError):
            b.transfer("ann", "zed", 1)
        self.assertEqual(ann.balance, 700)
        with self.assertRaises(ValueError):
            b.open("ann")
        bob.withdraw(100)
        self.assertEqual(b.history("ann"), [("deposit", 1000), ("transfer_out", 300)])
        self.assertEqual(b.history("bob"), [("transfer_in", 300), ("withdraw", 100)])
""",
        "reference": {
            "accounts.py": """
class InsufficientFunds(Exception):
    pass

class Account:
    def __init__(self, owner, balance=0):
        self.owner, self.balance, self.events = owner, balance, []
    def _check(self, amount):
        if amount <= 0:
            raise ValueError(amount)
    def deposit(self, amount, kind="deposit"):
        self._check(amount)
        self.balance += amount
        self.events.append((kind, amount))
    def withdraw(self, amount, kind="withdraw"):
        self._check(amount)
        if amount > self.balance:
            raise InsufficientFunds(amount)
        self.balance -= amount
        self.events.append((kind, amount))
""",
            "ledger.py": """
from accounts import Account

class Bank:
    def __init__(self):
        self.accounts = {}
    def open(self, owner):
        if owner in self.accounts:
            raise ValueError(owner)
        self.accounts[owner] = Account(owner)
        return self.accounts[owner]
    def transfer(self, src, dst, amount):
        a, b = self.accounts[src], self.accounts[dst]
        a.withdraw(amount, "transfer_out")
        b.deposit(amount, "transfer_in")
    def history(self, owner):
        return list(self.accounts[owner].events)
""",
        },
    },
    {
        "id": "markdown_toc",
        "level": 3,
        "spec": """Write `toc.py` with `toc(markdown: str) -> str` that builds a table of contents from ATX headings
("#" to "######" followed by a space). Ignore headings inside fenced code blocks (``` lines).
Each line is two spaces of indent per level below the smallest level present, then "- [Title](#slug)".
Slug: lower case, keep letters, digits, spaces and hyphens only, spaces become hyphens. Repeated slugs get
"-1", "-2", ... suffixes in order of appearance. Lines are joined with "\\n"; no headings gives "".""",
        "hidden": """
import unittest
from toc import toc

DOC = '''# Intro
Some text.
## Install & Setup
```
# not a heading
```
### On Linux
## Usage
#nospace is not a heading
## Usage
'''

class T(unittest.TestCase):
    def test_doc(self):
        self.assertEqual(toc(DOC), "\\n".join([
            "- [Intro](#intro)",
            "  - [Install & Setup](#install--setup)",
            "    - [On Linux](#on-linux)",
            "  - [Usage](#usage)",
            "  - [Usage](#usage-1)",
        ]))
    def test_min_level(self):
        self.assertEqual(toc("## A\\n### B"), "- [A](#a)\\n  - [B](#b)")
    def test_none(self):
        self.assertEqual(toc("just text"), "")
""",
        "reference": {"toc.py": """
import re

def toc(markdown):
    heads, fence = [], False
    for line in markdown.splitlines():
        if line.startswith("```"):
            fence = not fence
            continue
        m = re.match(r"(#{1,6}) (.+)", line)
        if m and not fence:
            heads.append((len(m.group(1)), m.group(2).strip()))
    if not heads:
        return ""
    low = min(h[0] for h in heads)
    seen, out = {}, []
    for level, title in heads:
        slug = re.sub(r"[^a-z0-9 -]", "", title.lower()).replace(" ", "-")
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        if n:
            slug = f"{slug}-{n}"
        out.append("  " * (level - low) + f"- [{title}](#{slug})")
    return "\\n".join(out)
"""},
    },
]

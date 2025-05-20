"""gto_helper.py – advanced GTO advisor (Python 3)
Fully interactive, no syntax errors (May 2025)
Features
• Strict & bets modes with EV
• Villain range %, equity histogram, CSV logging
• Game‑flow loop: Continue / New game / Exit
"""
import csv, json, math, sys, random
from pathlib import Path
from datetime import datetime
from collections import Counter
import eval7
try:
    import numpy as np
except ImportError:
    np = None

SUITS, RANKS = "shdc", "23456789TJQKA"

# Older versions of eval7.Card don't expose a `rank_char` attribute. Rather
# than monkey-patching the immutable class, provide a helper that extracts the
# rank from ``str(card)`` when needed.
def card_rank_char(card: eval7.Card) -> str:
    rc = getattr(card, "rank_char", str(card)[0])
    rc = rc.upper()
    return "T" if rc == "0" else rc

DEFAULTS_PATH = Path.home() / ".gto_defaults.json"

def load_defaults():
    if DEFAULTS_PATH.exists():
        try:
            return json.loads(DEFAULTS_PATH.read_text())
        except Exception:
            return {}
    return {}

def save_defaults(d):
    try:
        DEFAULTS_PATH.write_text(json.dumps(d))
    except Exception:
        pass

COLORS = {
    "green": "\033[92m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "end": "\033[0m",
}

def color(text: str, c: str) -> str:
    return f"{COLORS.get(c,'')}" + text + COLORS["end"]

# ── helpers ──────────────────────────────────────────

def cards(txt: str):
    """Parse cards. Accepts spaced ("Ah Ks") or concatenated ("AhKs7c")."""
    txt = txt.strip()
    if not txt:
        return []
    if " " in txt:
        raw = txt.split()
    else:  # chunk every 2 chars
        if len(txt) % 2:
            raise ValueError("Card string length must be even (pairs of chars).")
        raw = [txt[i:i+2] for i in range(0, len(txt), 2)]
    # normalize and validate
    toks = []
    for t in raw:
        if len(t)!=2:
            raise ValueError(f"Bad card: {t}")
        rank, suit = t[0].upper(), t[1].lower()
        if rank=='0': rank='T'
        if rank not in RANKS or suit not in SUITS:
            raise ValueError(f"Bad card: {t}")
        toks.append(rank + suit)
    if len(toks) != len(set(toks)):
        raise ValueError("Duplicate cards detected.")
    return [eval7.Card(t) for t in toks]


# build crude strength order for 169 combos
_order, ORDER = 169, {}
for r1 in RANKS[::-1]:
    for r2 in RANKS[::-1]:
        if r1 < r2:
            continue
        for suited in (True, False):
            ORDER[(r1, r2, suited)] = _order; _order -= 1

def top_range(p):
    keep = math.ceil(169 * p / 100)
    return set(k for k, _ in zip(sorted(ORDER, key=ORDER.get, reverse=True), range(keep)))

def load_range_file(path_or_file):
    """Load custom range from a text file of two-card combos."""
    rng = set()
    if hasattr(path_or_file, "read"):
        raw = path_or_file.read()
        text = raw.decode("utf-8") if hasattr(raw, "decode") else raw
    else:
        p = Path(path_or_file)
        if not p.exists():
            raise ValueError(f"Range file not found: {path_or_file}")
        text = p.read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cs = cards(line)
        except ValueError:
            continue
        if len(cs) != 2:
            continue
        r1 = max(card_rank_char(cs[0]), card_rank_char(cs[1]))
        r2 = min(card_rank_char(cs[0]), card_rank_char(cs[1]))
        s = cs[0].suit == cs[1].suit
        rng.add((r1, r2, s))
    if not rng:
        raise ValueError("No valid combos in range file")
    return rng

def load_weighted_range(path_or_file):
    """Load weighted ranges from a text file ``hand weight`` per line."""
    data = {}
    if hasattr(path_or_file, "read"):
        raw = path_or_file.read()
        text = raw.decode("utf-8") if hasattr(raw, "decode") else raw
    else:
        p = Path(path_or_file)
        if not p.exists():
            raise ValueError(f"Range file not found: {path_or_file}")
        text = p.read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            combo, w = line.split()
            weight = float(w)
            cs = cards(combo)
            if len(cs) != 2:
                continue
            r1 = max(card_rank_char(cs[0]), card_rank_char(cs[1]))
            r2 = min(card_rank_char(cs[0]), card_rank_char(cs[1]))
            s = cs[0].suit == cs[1].suit
            data[(r1, r2, s)] = weight
        except Exception:
            continue
    if not data:
        raise ValueError("No valid weighted combos in file")
    return data

PRESET_SCENARIOS = {
    "flush_draw": ("AhKh", "QhTh2c"),
    "set_vs_draw": ("9c9d", "JdTd9h"),
}

# ── equity simulation ─────────────────────────────────

def _simulate(hero, board, villains, rng, weighted, iters):
    wins, buckets = 0.0, Counter()
    for _ in range(iters):
        deck = eval7.Deck(); [deck.cards.remove(c) for c in hero + board]; deck.shuffle()
        opp = []
        while len(opp) < villains:
            a, b = deck.deal(2)
            r1, r2, s = max(card_rank_char(a), card_rank_char(b)), min(card_rank_char(a), card_rank_char(b)), a.suit == b.suit
            if weighted is not None:
                w = weighted.get((r1, r2, s), 0.0)
                if random.random() > w:
                    deck.cards.extend([a, b]); deck.shuffle(); continue
            elif rng:
                if (r1, r2, s) not in rng:
                    deck.cards.extend([a, b]); deck.shuffle(); continue
            opp.append([a, b])
        sim_board = board + deck.deal(5 - len(board))
        hero_sc = eval7.evaluate(hero + sim_board)
        best, ties, hero_best = hero_sc, 1, True
        for v in opp:
            vs = eval7.evaluate(v + sim_board)
            if vs > best:
                best, ties, hero_best = vs, 1, False
            elif vs == best:
                ties += 1; hero_best |= vs == hero_sc
        if hero_best and hero_sc == best:
            wins += 1 / ties
            buckets[int((1 / ties) * 10)] += 1
        else:
            buckets[0] += 1
    return wins, buckets


def equity(hero, board, villains, pct=None, custom=None, iters=25000, weighted=None, multiprocess=False):
    """Simulate equity.

    pct    -- top X percent of hands to keep (ignored if ``custom`` provided)
    custom -- optional set defining a range
    weighted -- optional dict mapping combos to weights (0-1)
    multiprocess -- use multiple processes if True
    """
    rng = custom if custom is not None else (top_range(pct) if (pct or 0) > 0 else None)
    if multiprocess:
        try:
            import multiprocessing as mp
            procs = min(mp.cpu_count(), 4)
            chunk = iters // procs
            todo = [chunk + (1 if i < iters % procs else 0) for i in range(procs)]
            with mp.Pool(procs) as pool:
                args = [(hero, board, villains, rng, weighted, n) for n in todo]
                results = pool.starmap(_simulate, args)
        except Exception:
            results = [_simulate(hero, board, villains, rng, weighted, iters)]
    else:
        results = [_simulate(hero, board, villains, rng, weighted, iters)]
    wins, buckets = 0.0, Counter()
    for w, b in results:
        wins += w; buckets.update(b)
    eq = wins / iters
    hist = (np.bincount([min(k, 9) for k in buckets.elements()], minlength=10) / iters).tolist() if np else [buckets[i]/iters for i in range(10)]
    return eq, hist

# ── decisions ───────────────────────────────────────

def strict_action(eq):
    return "RAISE" if eq >= 0.65 else "CHECK" if eq >= 0.4 else "FOLD"

def equilibrium_solver(hero, board, villains, pct=None, custom=None, iters=10000):
    """Very naive equilibrium solver using equity as proxy."""
    eq, _ = equity(hero, board, villains, pct, custom, iters=iters)
    bet = round(min(max(eq, 0.0), 1.0), 2)
    check = round(1 - bet, 2)
    return {"bet_freq": bet, "check_freq": check}

RAISE_SIZES = {"0.5": .5, "1": 1.0, "2": 2.0, "shove": None}

def decide_bets(eq, pot, bet, stack, pref):
    call_ev = eq * (pot + bet) - (1 - eq) * bet
    fold_ev = 0.0
    raise_total = stack if pref == "shove" or stack <= bet else min(bet + (pot + 2*bet) * RAISE_SIZES[pref], stack)
    raise_ev = eq * (pot + bet + raise_total) - (1 - eq) * raise_total
    best = max(fold_ev, call_ev, raise_ev)
    if best == raise_ev and raise_total > bet:
        act = "ALL_IN" if raise_total == stack else "RAISE"
        mv = {"call": bet, "raise": round(raise_total - bet,2), "total": round(raise_total,2),
              "breakdown": f"{bet} + {round(raise_total - bet,2)} = {round(raise_total,2)}"}
    elif best == call_ev and stack >= bet:
        act, mv = "CALL", {"call": bet, "raise": 0, "total": bet, "breakdown": f"{bet} = {bet}"}
    else:
        act, mv = "FOLD", {}
    return act, fold_ev, call_ev, raise_ev, mv

# ── logging ─────────────────────────────────────────

def log_row(d):
    f = "gto_history.csv"
    head = ["ts","hand","board","villains","range","eq","act","ev_fold","ev_call","ev_raise"]
    need_header = not (PathExists := __import__('pathlib').Path(f).exists())
    with open(f,'a',newline='') as csvfile:
        w = csv.writer(csvfile)
        if need_header:
            w.writerow(head)
        w.writerow([d.get(h) for h in head])

# ── main loop ───────────────────────────────────────

def prompt_bets():
    try:
        pot = float(input("Current pot: "))
        bet = float(input("Facing bet: "))
        stack = float(input("Your stack: "))
    except ValueError as e:
        raise ValueError("Bad numeric value") from e
    pref = input("Raise size (0.5 / 1 / 2 / shove): ")
    if pref not in RAISE_SIZES:
        raise ValueError("Bad raise size")
    return pot, bet, stack, pref

def play():
    if any(arg in sys.argv for arg in ("-h", "--help", "help")):
        print("""\nQuickGTO command line usage:\n\n"
              "Run: python gto_helper.py\n"
              "You will be prompted for opponents, range (percent or path to\n"
              "custom range file), your hand and board cards. Example:\n"
              "  Opponents (1-9) [2]: 3\n"
              "  Villain range % or file [0]: 20\n"
              "  Hero hand (H/S/D/C): AhKs\n"
              "  Board cards (0/3/4/5): 7c8d9s\n"
              "  Mode strict/bets [s/b]: s\n"
              "Use Ctrl+C to exit at any time.""")
        return

    defaults = load_defaults()
    print("Welcome to GTO Helper – advanced edition\n")
    while True:
        try:
            v_in = input(f"Opponents (1–9) [{defaults.get('villains',2)}]: ")
            villains = int(v_in) if v_in.strip() else int(defaults.get("villains", 2))
            if not 1 <= villains <= 9:
                raise ValueError("Opponents must be between 1 and 9")
            if villains == 1:
                print("Heads‑up mode enabled")
                r_in = input(f"Villain range % or file [{defaults.get('range','0')}]: ")
                if not r_in.strip():
                    r_in = str(defaults.get("range", "0"))
                if Path(r_in).exists():
                    rng_pct = None
                    rng_custom = load_range_file(r_in)
                else:
                    rng_pct = float(r_in)
                    rng_custom = None
                if input("Save as defaults? [y/N]: ").lower().startswith('y'):
                    save_defaults({"villains": villains, "range": r_in})
                break
        except ValueError as e:
            print(color(f"Error: {e}", "red") + "\n")
    hero = board = None
    if input("Load example scenario? [y/N]: ").lower().startswith('y'):
        print("Available examples:")
        for i, name in enumerate(PRESET_SCENARIOS, 1):
            print(f"  {i}. {name}")
        try:
            sel = int(input("Choose example #: ")) - 1
            name = list(PRESET_SCENARIOS.keys())[sel]
        except Exception:
            print(color("Bad selection. Using first example.", "red"))
            name = list(PRESET_SCENARIOS.keys())[0]
        hero_raw, board_raw = PRESET_SCENARIOS[name]
        hero = cards(hero_raw)
        board = cards(board_raw)
    while True:
        try:
            if hero is None:
                hero = cards(input("Hero hand (H/S/D/C): "))
                if len(hero) != 2:
                    raise ValueError("Hero hand must be exactly 2 cards")
            if board is None:
                board = cards(input("Board cards (0/3/4/5): "))
                if len(board) not in (0, 3, 4, 5):
                    raise ValueError("Board must be 0, 3, 4, or 5 cards")
            mode = input("Mode strict/bets [s/b]: ").lower().strip()
            if mode.startswith('b'):
                pot, bet, stack, pref = prompt_bets()
                mode = "bets"
            else:
                mode = "strict"
                eq, hist = equity(hero, board, villains, rng_pct, rng_custom)
        except ValueError as e:
            print(f"Error: {e}. Restarting current hand.\n")
            hero = board = None
            continue
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        ts = datetime.now().isoformat()
        if mode == "strict":
            act = strict_action(eq)
            res = {"equity": round(eq,3), "histogram": hist, "recommended_action": act}
            print(json.dumps(res, indent=2))
            print(color(f"Recommended: {act}", "green"))
            log_row({"ts":ts,"hand":" ".join(str(c) for c in hero),"board":" ".join(str(c) for c in board),
                     "villains":villains,"range":rng_pct,"eq":round(eq,3),"act":act,
                     "ev_fold":0,"ev_call":0,"ev_raise":0})
        else:
            act, evf, evc, evr, mv = decide_bets(eq,pot,bet,stack,pref)
            res = {"equity":round(eq,3),"histogram":hist,"chosen_action":act,
                   "EV_fold":round(evf,2),"EV_call":round(evc,2),"EV_raise":round(evr,2)}
            res.update(mv)
            print(json.dumps(res, indent=2))
            print(color(f"Chosen: {act}", "yellow"))
            log_row({"ts":ts,"hand":" ".join(str(c) for c in hero),"board":" ".join(str(c) for c in board),
                     "villains":villains,"range":rng_pct,"eq":round(eq,3),"act":act,
                     "ev_fold":round(evf,2),"ev_call":round(evc,2),"ev_raise":round(evr,2)})
        print(f"Current hero: {' '.join(str(c) for c in hero)}")
        print(f"Current board: {' '.join(str(c) for c in board) if board else '(none)'}")
        nxt = input("\n[C]ontinue / [N]ew game / [E]xit: ").lower().strip()
        if nxt.startswith('c'):
            add = cards(input("Add board cards: "))
            board.extend(add)
            continue
        if nxt.startswith('n'):
            hero = board = None
            continue
        break

if __name__ == "__main__":
    play()

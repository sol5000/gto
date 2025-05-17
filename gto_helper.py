"""gto_helper.py – advanced GTO advisor (Python 3)
Fully interactive, no syntax errors (May 2025)
Features
• Strict & bets modes with EV
• Villain range %, equity histogram, CSV logging
• Game‑flow loop: Continue / New game / Exit
"""
import csv, json, sys, math
from datetime import datetime
from collections import Counter
import eval7
try:
    import numpy as np
except ImportError:
    np = None

SUITS, RANKS = "shdc", "23456789TJQKA"

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
            sys.exit("Card string length must be even (pairs of chars).")
        raw = [txt[i:i+2] for i in range(0, len(txt), 2)]
    # normalize and validate
    toks = []
    for t in raw:
        if len(t)!=2:
            sys.exit(f"Bad card: {t}")
        rank, suit = t[0].upper(), t[1].lower()
        if rank=='0': rank='T'
        if rank not in RANKS or suit not in SUITS:
            sys.exit(f"Bad card: {t}")
        toks.append(rank + suit)
    if len(toks) != len(set(toks)):
        sys.exit("Duplicate cards detected.")
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

# ── equity simulation ─────────────────────────────────

def equity(hero, board, villains, pct, iters=25000):
    rng = top_range(pct) if pct > 0 else None
    wins, buckets = 0.0, Counter()
    for _ in range(iters):
        deck = eval7.Deck(); [deck.cards.remove(c) for c in hero + board]; deck.shuffle()
        opp = []
        while len(opp) < villains:
            a, b = deck.deal(2)
            if rng:
                r1, r2, s = max(a.rank_char, b.rank_char), min(a.rank_char, b.rank_char), a.suit == b.suit
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
    eq = wins / iters
    hist = (np.bincount([min(k, 9) for k in buckets.elements()], minlength=10) / iters).tolist() if np else [buckets[i]/iters for i in range(10)]
    return eq, hist

# ── decisions ───────────────────────────────────────

def strict_action(eq):
    return "RAISE" if eq >= 0.65 else "CHECK" if eq >= 0.4 else "FOLD"

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
    pot = float(input("Current pot: "))
    bet = float(input("Facing bet: "))
    stack = float(input("Your stack: "))
    pref = input("Raise size (0.5 / 1 / 2 / shove): ")
    if pref not in RAISE_SIZES: sys.exit("Bad raise size.")
    return pot, bet, stack, pref

def play():
    print("Welcome to GTO Helper – advanced edition\n")
    villains = int(input("Opponents (2–9): "))
    rng_pct = float(input("Villain range % (0=random): "))
    hero = board = None
    while True:
        if hero is None:
            hero = cards(input("Hero hand (H/S/D/C): "))
        if board is None:
            board = cards(input("Board cards (0/3/4/5): "))
        mode = input("Mode strict/bets [s/b]: ").lower().strip()
        if mode.startswith('b'):
            pot, bet, stack, pref = prompt_bets()
            mode = "bets"
        else:
            mode = "strict"
        eq, hist = equity(hero, board, villains, rng_pct)
        ts = datetime.now().isoformat()
        if mode == "strict":
            act = strict_action(eq)
            res = {"equity": round(eq,3), "histogram": hist, "recommended_action": act}
            print(json.dumps(res, indent=2))
            log_row({"ts":ts,"hand":" ".join(str(c) for c in hero),"board":" ".join(str(c) for c in board),
                     "villains":villains,"range":rng_pct,"eq":round(eq,3),"act":act,
                     "ev_fold":0,"ev_call":0,"ev_raise":0})
        else:
            act, evf, evc, evr, mv = decide_bets(eq,pot,bet,stack,pref)
            res = {"equity":round(eq,3),"histogram":hist,"chosen_action":act,
                   "EV_fold":round(evf,2),"EV_call":round(evc,2),"EV_raise":round(evr,2)}
            res.update(mv)
            print(json.dumps(res, indent=2))
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

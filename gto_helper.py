"""gto_helper.py – advanced GTO advisor (Python 3)
Fully interactive, no syntax errors (May 2025)
Features
• Strict & bets modes with EV
• Villain range %, equity histogram, CSV logging
• Game‑flow loop: Continue / New game / Exit
"""
import csv, json, math, sys, random, argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
import eval7

# ── compatibility shims ----------------------------------------------
try:
    _ = eval7.Card("Ah").rank_char  # new versions provide rank_char
except AttributeError:  # older eval7 versions only expose numeric rank
    def _rank_char(self):
        """Return rank as a single character (e.g. 'A')."""
        return str(self)[0]

try:
    import numpy as np
except ImportError:
    np = None

SUITS, RANKS = "shdc", "23456789TJQKA"

# simple in-memory cache for equity calculations
CACHE = {}


# Older versions of eval7.Card don't expose a `rank_char` attribute. Rather
# than monkey-patching the immutable class, provide a helper that extracts the
# rank from ``str(card)`` when needed.
def card_rank_char(card: eval7.Card) -> str:
    rc = getattr(card, "rank_char", str(card)[0])
    rc = rc.upper()
    return "T" if rc == "0" else rc


DEFAULTS_PATH = Path.home() / ".gto_defaults.json"

# default strict-mode thresholds (can be overridden in defaults file)
STRICT_THRESH = {"raise": 0.65, "check": 0.4}

# solver accuracy presets
ACCURACY_ITERS = {"Fast": 10000, "Balanced": 25000, "Detailed": 100000}

# available games
GAMES = ["Holdem", "Short Deck"]

def load_defaults():
    if DEFAULTS_PATH.exists():
        try:
            data = json.loads(DEFAULTS_PATH.read_text())
            STRICT_THRESH.update({
                "raise": data.get("strict_raise", STRICT_THRESH["raise"]),
                "check": data.get("strict_check", STRICT_THRESH["check"]),
            })
            return data
        except Exception:
            return {}
    return {}

def save_defaults(d):
    try:
        existing = load_defaults()
        existing.update(d)
        existing["strict_raise"] = STRICT_THRESH["raise"]
        existing["strict_check"] = STRICT_THRESH["check"]
        DEFAULTS_PATH.write_text(
            json.dumps(existing, indent=2)
        )
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

def deck_for_game(game: str) -> eval7.Deck:
    """Return a deck for the given game type."""
    deck = eval7.Deck()
    if game.lower().startswith("short"):
        for r in "2345":
            for s in SUITS:
                card = eval7.Card(r + s)
                deck.cards.remove(card)
    return deck

PRESET_SCENARIOS = {
    "flush_draw": ("AhKh", "QhTh2c"),
    "set_vs_draw": ("9c9d", "JdTd9h"),
    "btn_vs_bb": ("AhKd", ""),
    "sb_limp_pot": ("7s6s", ""),
}

# ── equity simulation ─────────────────────────────────

def _simulate(hero, board, villains, rng, weighted, iters, game):
    wins, buckets = 0.0, Counter()
    for _ in range(iters):
        deck = deck_for_game(game)
        [deck.cards.remove(c) for c in hero + board]
        deck.shuffle()
        opp = []
        while len(opp) < villains:
            a, b = deck.deal(2)
            r1 = max(card_rank_char(a), card_rank_char(b))
            r2 = min(card_rank_char(a), card_rank_char(b))
            s = a.suit == b.suit
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


def equity(hero, board, villains, pct=None, custom=None, iters=25000, weighted=None, multiprocess=False, show_progress=False, game="Holdem"):
    """Simulate equity.

    pct    -- top X percent of hands to keep (ignored if ``custom`` provided)
    custom -- optional set defining a range
    weighted -- optional dict mapping combos to weights (0-1)
    multiprocess -- use multiple processes if True
    """
    rng = custom if custom is not None else (top_range(pct) if (pct or 0) > 0 else None)

    key = (
        tuple(str(c) for c in hero),
        tuple(str(c) for c in board),
        villains,
        pct if custom is None else tuple(sorted(custom)),
        iters,
        game,
    )
    if key in CACHE:
        return CACHE[key]
    if multiprocess:
        try:
            import multiprocessing as mp
            procs = min(mp.cpu_count(), 4)
            chunk = iters // procs
            todo = [chunk + (1 if i < iters % procs else 0) for i in range(procs)]
            with mp.Pool(procs) as pool:
                args = [(hero, board, villains, rng, weighted, n, game) for n in todo]
                results = pool.starmap(_simulate, args)
        except Exception:
            results = [_simulate(hero, board, villains, rng, weighted, iters, game)]
    else:
        if show_progress:
            try:
                from tqdm import trange
                rng_iter = trange(iters, desc="Sim")
            except Exception:
                rng_iter = range(iters)
            wins, buckets = 0.0, Counter()
            for _ in rng_iter:
                w, b = _simulate(hero, board, villains, rng, weighted, 1, game)
                wins += w; buckets.update(b)
            results = [(wins, buckets)]
        else:
            results = [_simulate(hero, board, villains, rng, weighted, iters, game)]
    wins, buckets = 0.0, Counter()
    for w, b in results:
        wins += w; buckets.update(b)
    eq = wins / iters
    hist = (np.bincount([min(k, 9) for k in buckets.elements()], minlength=10) / iters).tolist() if np else [buckets[i]/iters for i in range(10)]
    CACHE[key] = (eq, hist)
    return eq, hist

# ── decisions ───────────────────────────────────────

def strict_action(eq):
    return (
        "RAISE"
        if eq >= STRICT_THRESH["raise"]
        else "CHECK" if eq >= STRICT_THRESH["check"] else "FOLD"
    )

def equilibrium_solver(hero, board, villains, pct=None, custom=None, iters=10000, game="Holdem"):
    """Very naive equilibrium solver using equity as proxy."""
    eq, _ = equity(hero, board, villains, pct, custom, iters=iters, game=game)
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
            game = input(f"Game type Holdem/Short [{defaults.get('game','Holdem')}]: ").strip().title() or defaults.get('game','Holdem')
            if game not in GAMES:
                raise ValueError("Invalid game type")
            acc = input(f"Solver accuracy Fast/Balanced/Detailed [{defaults.get('accuracy','Balanced')}]: ").strip().title() or defaults.get('accuracy','Balanced')
            if acc not in ACCURACY_ITERS:
                raise ValueError("Invalid accuracy option")
            rng_pct = float(defaults.get("range", 0))
            rng_custom = None
            r_in = str(rng_pct)
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
            if input("Save as defaults? [y/N]: ").lower().startswith('y'):
                save_defaults({"villains": villains, "range": r_in, "game": game, "accuracy": acc})
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
            iters = ACCURACY_ITERS[acc]
            if mode.startswith('b'):
                pot, bet, stack, pref = prompt_bets()
                mode = "bets"
                eq, hist = equity(hero, board, villains, rng_pct, rng_custom, iters=iters, game=game)
            else:
                mode = "strict"
                sr = input(f"Strict raise threshold [{STRICT_THRESH['raise']}] : ") or str(STRICT_THRESH['raise'])
                sc = input(f"Strict check threshold [{STRICT_THRESH['check']}] : ") or str(STRICT_THRESH['check'])
                try:
                    STRICT_THRESH['raise'] = float(sr)
                    STRICT_THRESH['check'] = float(sc)
                except ValueError:
                    print("Bad threshold values, using defaults")
                eq, hist = equity(hero, board, villains, rng_pct, rng_custom, iters=iters, game=game)
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

def run_simulation_from_args(args):
    hero = cards(args.hero)
    board = cards(args.board) if args.board else []
    villains = args.villains or 2
    d = load_defaults()
    if args.v_range:
        if Path(args.v_range).exists():
            rng_pct = None
            rng_custom = load_range_file(args.v_range)
        else:
            rng_pct = float(args.v_range)
            rng_custom = None
    else:
        rng_pct = float(d.get("range", 0))
        rng_custom = None
        if args.mode == "strict":
            STRICT_THRESH["raise"] = float(args.strict_raise if args.strict_raise is not None else d.get("strict_raise", STRICT_THRESH["raise"]))
            STRICT_THRESH["check"] = float(args.strict_check if args.strict_check is not None else d.get("strict_check", STRICT_THRESH["check"]))

    game = getattr(args, "game", d.get("game", "Holdem"))
    acc = getattr(args, "accuracy", d.get("accuracy", "Balanced"))
    iters = args.iters or ACCURACY_ITERS.get(acc, 25000)

    eq, hist = equity(
        hero,
        board,
        villains,
        rng_pct,
        rng_custom,
        iters=iters,
        multiprocess=args.multiprocess,
        show_progress=True,
        game=game,
    )

    if args.mode == "strict":
        act = strict_action(eq)
        print(json.dumps({"equity": round(eq,3), "action": act}, indent=2))
    else:
        act, evf, evc, evr, mv = decide_bets(
            eq, args.pot, args.bet, args.stack, args.raise_size
        )
        res = {
            "equity": round(eq,3),
            "chosen_action": act,
            "EV_fold": round(evf,2),
            "EV_call": round(evc,2),
            "EV_raise": round(evr,2),
        }
        res.update(mv)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuickGTO")
    parser.add_argument("--hero")
    parser.add_argument("--board", default="")
    parser.add_argument("--villains", type=int)
    parser.add_argument("--range", dest="v_range")
    parser.add_argument("--mode", choices=["strict", "bets"], default="strict")
    parser.add_argument("--pot", type=float, default=0.0)
    parser.add_argument("--bet", type=float, default=0.0)
    parser.add_argument("--stack", type=float, default=0.0)
    parser.add_argument("--raise-size", default="1", choices=list(RAISE_SIZES.keys()))
    parser.add_argument("--iters", type=int)
    parser.add_argument("--game", choices=GAMES, default=None)
    parser.add_argument("--accuracy", choices=list(ACCURACY_ITERS.keys()), default=None)
    parser.add_argument("--strict-raise", type=float)
    parser.add_argument("--strict-check", type=float)
    parser.add_argument("--multiprocess", action="store_true")
    parser.add_argument("--batch")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if (args.hero or args.batch) and not args.interactive:
        if args.batch:
            for line in Path(args.batch).read_text().splitlines():
                if not line.strip():
                    continue
                hero_raw, board_raw = line.split(",")[:2]
                args.hero = hero_raw.strip()
                args.board = board_raw.strip()
                run_simulation_from_args(args)
        else:
            run_simulation_from_args(args)
    else:
        play()

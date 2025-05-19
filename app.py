"""app.py – Streamlit GUI for GTO Helper (May 2025)
Keeps **all** earlier features and adds:
  • Quick text entry _and_ dropdown pickers for cards
  • Real‑time validation inside a form
  • Plotly histogram, CSV download
  • Theme instructions in sidebar
"""
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

from gto_helper import (
    cards,
    equity,
    strict_action,
    decide_bets,
    RAISE_SIZES,
    log_row,
    SUITS,
    RANKS,
    load_range_file,
    PRESET_SCENARIOS,
)

# ── card utilities ──────────────────────────────────────────────────────────
SUIT_EMOJI = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
EMOJI_TO_SUIT = {v: k for k, v in SUIT_EMOJI.items()}
CARD_CODES = [r + s for r in RANKS for s in SUITS]

def label(code):
    return f"{code[0]}{SUIT_EMOJI[code[1]]}"

# ── sidebar form ────────────────────────────────────────────────────────────
screen = st.sidebar.radio("Screen", ["Simulation", "Training"], key="screen")
example_choice = st.sidebar.selectbox("Example", list(PRESET_SCENARIOS.keys()), key="ex_select")

def load_example():
    hero_raw, board_raw = PRESET_SCENARIOS[example_choice]
    st.session_state.hero_text = hero_raw
    st.session_state.board_text = board_raw
    st.toast(f"Loaded example '{example_choice}'")

if st.sidebar.button("Load example scenario"):
    load_example()

with st.sidebar.form("config"):
    st.header("GTO Helper settings")

    # Hero
    st.markdown("#### Hero hand (enter text OR pick 2 cards)")
    hero_text = st.text_input("Quick hero entry (e.g. AhKs)", key="hero_text")
    hero_pick = st.multiselect("Hero picker (max 2)", [label(c) for c in CARD_CODES], max_selections=2)

    # Board
    st.markdown("#### Board cards")
    board_text = st.text_input("Quick board entry (e.g. 7c8d9s)", key="board_text")
    board_pick = st.multiselect("Board picker (0–5)", [label(c) for c in CARD_CODES], max_selections=5)

    villains   = st.slider("Opponents", 2, 9, 3)
    range_pct  = st.slider("Villain range %", 0, 50, 0)
    range_file = st.file_uploader("Custom range file", type="txt")

    mode = st.radio("Mode", ["Strict", "Bets"])
    if mode == "Bets":
        pot   = st.number_input("Current pot", 0.0, step=10.0, value=100.0)
        bet   = st.number_input("Facing bet", 0.0, step=5.0,  value=20.0)
        stack = st.number_input("Your stack", 0.0, step=10.0, value=200.0)
        pref  = st.selectbox("Raise size", list(RAISE_SIZES.keys()), index=1)
    submit = st.form_submit_button("Run simulation")

# ── helper to merge text / picker input -------------------------------------

def merge_inputs(text: str, picks: list[str]):
    text = text.strip().replace(" ", "")
    if text:
        return text  # user typed something → highest priority
    if picks:
        # convert emoji labels back to codes
        raw_codes = []
        for lbl in picks:
            rank = lbl[0].upper()
            suit = EMOJI_TO_SUIT[lbl[1]]
            raw_codes.append(rank + suit)
        return "".join(raw_codes)
    return ""

# ── run simulation if submitted --------------------------------------------
if screen == "Simulation" and submit:
    hero_raw  = merge_inputs(hero_text, hero_pick)
    board_raw = merge_inputs(board_text, board_pick)

    try:
        hero  = cards(hero_raw)
        board = cards(board_raw)
    except SystemExit as e:
        st.error(str(e)); st.stop()

    if len(hero) != 2:
        st.error("Hero hand must be exactly 2 cards."); st.stop()
    if len(board) not in (0,3,4,5):
        st.error("Board must be 0, 3, 4, or 5 cards."); st.stop()

    rng_custom = None
    if range_file is not None:
        try:
            rng_custom = load_range_file(range_file)
        except Exception as e:
            st.error(str(e)); st.stop()
    eq, hist = equity(hero, board, villains, range_pct, rng_custom)

    if mode == "Strict":
        act = strict_action(eq)
        ev_fold = ev_call = ev_raise = 0
        extra = {}
    else:
        act, ev_fold, ev_call, ev_raise, extra = decide_bets(eq, pot, bet, stack, pref)

    # ── show results --------------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Equity", f"{eq:.2%}")
        st.subheader(f"**{act}**")
        if extra:
            st.json(extra)
    with col2:
        df = pd.DataFrame({"bucket": [f"{i*10}-{(i+1)*10}%" for i in range(10)], "freq": hist})
        fig = px.bar(df, x="bucket", y="freq", title="Equity distribution", labels={"bucket":"Equity %","freq":"Frequency"})
        st.plotly_chart(fig, use_container_width=True)

    # tooltips for transparency
    st.help(strict_action)
    st.help(decide_bets)

    # log
    log_row({
        "ts": datetime.now().isoformat(),
        "hand": hero_raw,
        "board": board_raw,
        "villains": villains,
        "range": range_pct,
        "eq": round(eq,3),
        "act": act,
        "ev_fold": round(ev_fold,2),
        "ev_call": round(ev_call,2),
        "ev_raise": round(ev_raise,2),
    })

    # CSV download
    if Path("gto_history.csv").exists():
        st.download_button("Download session CSV", open("gto_history.csv","rb").read(),"gto_history.csv")

if screen == "Training":
    st.header("Training mode")
    if "train_hero" not in st.session_state or st.button("New training hand"):
        import eval7
        deck = eval7.Deck(); deck.shuffle()
        st.session_state.train_hero = deck.deal(2)
        st.session_state.train_board = deck.deal(3)
    hero = st.session_state.train_hero
    board = st.session_state.train_board
    st.write("Hero:", " ".join(str(c) for c in hero))
    st.write("Board:", " ".join(str(c) for c in board))
    choice = st.radio("Your action?", ["CHECK", "RAISE", "FOLD"], horizontal=True, key="train_choice")
    if st.button("Submit answer"):
        eq, _ = equity(hero, board, villains, range_pct)
        best = strict_action(eq)
        if choice.upper() == best:
            st.success("Correct!")
        else:
            st.error(f"Incorrect. Best action: {best}")
        st.write(f"Equity was {eq:.2%}")

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
import random
import eval7
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="QuickGTO", layout="wide")

if "css_injected" not in st.session_state:        # run only on first render
    st.markdown(
        """
        <style>
        /*  Title bar  */
        #banner{font-size:2.6rem;font-weight:700;border:none;}

        /*  Sticky footer action-bar  */
        #footer{
            position:fixed;bottom:0;left:0;right:0;
            background:#222;padding:0.6rem 1rem;z-index:999;
            box-shadow:0 -2px 6px rgba(0,0,0,.4);
        }
        #footer .stButton>button{
            width:100%!important;height:3rem;
            border-radius:6px;font-size:1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["css_injected"] = True

from gto_helper import (
    cards,
    equity,
    strict_action,
    STRICT_THRESH,
    decide_bets,
    RAISE_SIZES,
    log_row,
    SUITS,
    RANKS,
    load_range_file,
    load_weighted_range,
    equilibrium_solver,
    PRESET_SCENARIOS,
    top_range,
    load_defaults,
    save_defaults,
)

# ── load defaults and intro handling ───────────────────────────────────────
defaults = load_defaults()

if "show_intro" not in st.session_state:
    st.session_state["show_intro"] = not defaults.get("intro_seen", False)

def show_intro(d):
    """Display a glassmorphic welcome screen with preference setup."""
    css = """
    <style>
    div[data-testid="stModal"] {
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(6px);
    }
    div[data-testid="stModal"] > div {
        background: rgba(255, 255, 255, 0.35) !important;
        backdrop-filter: blur(30px);
        border-radius: 20px;
        padding: 2rem 3rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        animation: pop 0.7s cubic-bezier(.175,.885,.32,1.275);
        width: 90%;
        max-width: 600px;
    }
    @keyframes pop {
        from {transform: scale(0.8); opacity: 0;}
        to   {transform: scale(1);   opacity: 1;}
    }
    </style>
    """

    features = [
        "Monte Carlo equity simulations",
        "Equilibrium solver",
        "Training mode & analytics",
        "Strategy pack import",
    ]

    if hasattr(st, "modal"):
        st.markdown(css, unsafe_allow_html=True)
        with st.modal("Welcome to QuickGTO 🎉"):
            st.write("QuickGTO analyses poker spots with Monte Carlo simulations.")
            st.markdown("**Key features**")
            for f in features:
                st.markdown(f"✔️ {f}")
            st.markdown("---")
            st.write("Configure your default preferences below.")
            v = st.slider("Default opponents", 1, 9, int(d.get("villains", 2)), key="intro_villains")
            r = st.slider("Default villain range %", 0, 50, int(d.get("range", 0)), key="intro_range")
            g = st.selectbox(
                "Default game", ["Holdem", "Short Deck"],
                index=0 if d.get("game", "Holdem") == "Holdem" else 1,
                key="intro_game"
            )
            a = st.selectbox(
                "Default solver accuracy", ["Fast", "Balanced", "Detailed"],
                index=["Fast", "Balanced", "Detailed"].index(d.get("accuracy", "Balanced")),
                key="intro_acc"
            )
            sr = st.number_input(
                "Default raise threshold", 0.0, 1.0, float(d.get("strict_raise", 0.65)), 0.05,
                key="intro_sr"
            )
            sc = st.number_input(
                "Default check threshold", 0.0, 1.0, float(d.get("strict_check", 0.4)), 0.05,
                key="intro_sc"
            )
            if st.button("Save preferences and start", key="intro_submit"):
                save_defaults(
                    {
                        "villains": v,
                        "range": r,
                        "game": g,
                        "accuracy": a,
                        "strict_raise": sr,
                        "strict_check": sc,
                        "intro_seen": True,
                    }
                )
                st.session_state.show_intro = False
                st.experimental_rerun()
    else:
        # basic fallback without modal
        st.write("## Welcome to QuickGTO 🎉")
        st.write(
            "QuickGTO analyses poker spots with Monte Carlo simulations. Configure your default preferences below."
        )
        with st.form("intro_form"):
            v = st.slider("Default opponents", 1, 9, int(d.get("villains", 2)))
            r = st.slider("Default villain range %", 0, 50, int(d.get("range", 0)))
            g = st.selectbox(
                "Default game", ["Holdem", "Short Deck"],
                index=0 if d.get("game", "Holdem") == "Holdem" else 1,
            )
            a = st.selectbox(
                "Default solver accuracy", ["Fast", "Balanced", "Detailed"],
                index=["Fast", "Balanced", "Detailed"].index(d.get("accuracy", "Balanced")),
            )
            sr = st.number_input(
                "Default raise threshold", 0.0, 1.0, float(d.get("strict_raise", 0.65)), 0.05
            )
            sc = st.number_input(
                "Default check threshold", 0.0, 1.0, float(d.get("strict_check", 0.4)), 0.05
            )
            submitted = st.form_submit_button("Save preferences and start")
        if submitted:
            save_defaults(
                {
                    "villains": v,
                    "range": r,
                    "game": g,
                    "accuracy": a,
                    "strict_raise": sr,
                    "strict_check": sc,
                    "intro_seen": True,
                }
            )
            st.session_state.show_intro = False
            st.experimental_rerun()


if st.session_state.get("show_intro"):
    show_intro(defaults)
    st.stop()

# ── card utilities ──────────────────────────────────────────────────────────
SUIT_EMOJI = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
EMOJI_TO_SUIT = {v: k for k, v in SUIT_EMOJI.items()}
CARD_CODES = [r + s for r in RANKS for s in SUITS]

def label(code):
    return f"{code[0]}{SUIT_EMOJI[code[1]]}"

def build_range_heatmap(rng_set):
    """Return dataframe for 13x13 heat map given a set of combos."""
    matrix = []
    ranks = list(RANKS)[::-1]
    for r1 in ranks:
        row = []
        for r2 in ranks:
            if r1 == r2:
                key = (r1, r2, False)
            elif ranks.index(r1) < ranks.index(r2):
                key = (r2, r1, True)
            else:
                key = (r1, r2, False)
            row.append(1 if key in rng_set else 0)
        matrix.append(row)
    df = pd.DataFrame(matrix, index=ranks, columns=ranks)
    return df

if "score" not in st.session_state:
    # use dict-style assignment; it’s always safe
    st.session_state["score"] = {"correct": 0, "total": 0}
st.markdown(
    """
    <style>
    #banner{position:sticky;top:0;background:#333;padding:4px;color:white;z-index:999;}
    #footer{position:sticky;bottom:0;background:#222;padding:4px;z-index:999;}
    .flip{animation:flip 0.5s;}
    @keyframes flip{from{transform:rotateY(90deg);}to{transform:rotateY(0);}}
    </style>
    <div id='banner'>QuickGTO</div>
    """,
    unsafe_allow_html=True,
)

# ── history for undo/redo ─────────────────────────────────────────────────--
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.future = []

# ── sidebar form ────────────────────────────────────────────────────────────
screen = st.sidebar.radio("Screen", ["Simulation", "Training"], key="screen")
example_choice = st.sidebar.selectbox("Example", list(PRESET_SCENARIOS.keys()), key="ex_select")
ui_level = st.sidebar.radio("Mode", ["Basic", "Advanced", "Equilibrium"], key="ui_level")
pack_data = json.load(open("strategy_packs.json")) if Path("strategy_packs.json").exists() else {}
pack_choice = st.sidebar.selectbox("Strategy pack", ["None"] + list(pack_data.keys()), key="pack_select")

def load_example():
    hero_raw, board_raw = PRESET_SCENARIOS[example_choice]
    st.session_state.hero_text = hero_raw
    st.session_state.board_text = board_raw
    st.toast(f"Loaded example '{example_choice}'")

if st.sidebar.button("Load example scenario"):
    load_example()
if st.sidebar.button("Load strategy pack") and pack_choice != "None":
    st.session_state.strategy_pack = pack_data.get(pack_choice)
    st.toast(f"Loaded strategy pack '{pack_choice}'")

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

    villains   = st.slider("Opponents", 2, 9, int(defaults.get("villains", 2)), help="Number of villains")
    range_pct  = st.slider("Villain range %", 0, 50, int(defaults.get("range", 0)), help="Top percent of hands")
    game_type = st.selectbox("Game", ["Holdem", "Short Deck"], index=0 if defaults.get("game", "Holdem") == "Holdem" else 1, help="Choose deck type")
    accuracy = st.selectbox("Solver accuracy", ["Fast","Balanced","Detailed"], index=["Fast","Balanced","Detailed"].index(defaults.get("accuracy", "Balanced")), help="Iteration preset")
    range_file = st.file_uploader("Custom range file", type="txt")
    weighted_file = None
    iters = 25000
    multiproc = False
    if ui_level == "Advanced":
        iters = st.number_input("Simulation iterations", 1000, 200000, 25000, step=5000)
        weighted_file = st.file_uploader("Weighted range file", type="txt")
        multiproc = st.checkbox("Enable multiprocessing")

    if ui_level != "Equilibrium":
        mode = st.radio("Mode", ["Strict", "Bets"], help="Decision algorithm")
        strict_raise = STRICT_THRESH["raise"]
        strict_check = STRICT_THRESH["check"]
        if mode == "Bets":
            pot   = st.number_input("Current pot", 0.0, step=10.0, value=100.0)
            bet   = st.number_input("Facing bet", 0.0, step=5.0,  value=20.0)
            stack = st.number_input("Your stack", 0.0, step=10.0, value=200.0)
            pref  = st.selectbox("Raise size", list(RAISE_SIZES.keys()), index=1)
        else:
            strict_raise = st.number_input("Raise threshold", 0.0, 1.0, strict_raise, 0.05, help=">= equity to raise")
            strict_check = st.number_input("Check threshold", 0.0, 1.0, strict_check, 0.05, help=">= equity to check")
    else:
        mode = "Equilibrium"
    submit_sidebar = st.form_submit_button("Run simulation")

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


with st.container():
    st.markdown("<div id='footer'>", unsafe_allow_html=True)

    col_run, col_undo, col_redo = st.columns(3, gap="medium")
    with col_run:
        run_click  = st.button("\u25B6 Run Simulation", key="run_footer",
                               use_container_width=True)
    with col_undo:
        undo_click = st.button("\u21BA Undo", key="undo",
                               use_container_width=True)
    with col_redo:
        redo_click = st.button("\u21BB Redo", key="redo",
                               use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# combine submit signals from sidebar and footer
submit = submit_sidebar or run_click

# handle undo/redo actions
if undo_click and st.session_state.history:
    state = st.session_state.history.pop()
    st.session_state.future.append({
        "hero_text": st.session_state.hero_text,
        "board_text": st.session_state.board_text,
    })
    # use dict-style assignment to avoid widget update errors
    st.session_state["hero_text"] = state["hero_text"]
    st.session_state["board_text"] = state["board_text"]

if redo_click and st.session_state.future:
    state = st.session_state.future.pop()
    st.session_state.history.append({
        "hero_text": st.session_state.hero_text,
        "board_text": st.session_state.board_text,
    })
    st.session_state["hero_text"] = state["hero_text"]
    st.session_state["board_text"] = state["board_text"]

# ── run simulation if submitted --------------------------------------------
if screen == "Simulation" and submit:
    st.session_state.history.append({
        "hero_text": st.session_state.hero_text,
        "board_text": st.session_state.board_text,
    })
    st.session_state.future.clear()
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
    weighted_rng = None
    if weighted_file is not None:
        try:
            weighted_rng = load_weighted_range(weighted_file)
        except Exception as e:
            st.error(str(e)); st.stop()

    prog = st.progress(0)
    with st.spinner("Running simulation..."):
        iters = {"Fast":10000,"Balanced":25000,"Detailed":100000}[accuracy]
        STRICT_THRESH["raise"] = strict_raise if ui_level != "Equilibrium" and mode == "Strict" else STRICT_THRESH["raise"]
        STRICT_THRESH["check"] = strict_check if ui_level != "Equilibrium" and mode == "Strict" else STRICT_THRESH["check"]
        if mode == "Equilibrium":
            result = equilibrium_solver(hero, board, villains, range_pct, rng_custom, game=game_type)
            st.subheader("Equilibrium frequencies")
            st.json(result)
            eq, hist = equity(hero, board, villains, range_pct, rng_custom, iters=iters, game=game_type)
            act = "N/A"
            ev_fold = ev_call = ev_raise = 0
            extra = {}
        else:
            eq, hist = equity(hero, board, villains, range_pct, rng_custom, iters=iters, weighted=weighted_rng, multiprocess=multiproc, game=game_type)
            if mode == "Strict":
                act = strict_action(eq)
                ev_fold = ev_call = ev_raise = 0
                extra = {}
            else:
                act, ev_fold, ev_call, ev_raise, extra = decide_bets(eq, pot, bet, stack, pref)

    # ── show results --------------------------------------------------------
    prog.progress(1)
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
        
        if ui_level == "Advanced":
            # Create additional graphs for Advanced mode
            df["cum"] = df["freq"].cumsum()
            fig2 = px.line(df, x="bucket", y="cum", title="Cumulative equity")
            
            rng_set = rng_custom if rng_custom is not None else top_range(range_pct)
            heat_df = build_range_heatmap(rng_set)
            fig_h = px.imshow(heat_df, color_continuous_scale=["white", "red"], aspect="auto", title="Range heat map")
            
            streets = ["Pre", "Flop", "Turn", "River"]
            boards = [[], board[:3], board[:4], board[:5]]
            eqs = []
            for b in boards:
                e, _ = equity(hero, b, villains, range_pct, rng_custom, iters=max(iters//5,1000), game=game_type)
                eqs.append(e)
            df_e = pd.DataFrame({"street": streets, "equity": eqs})
            fig_e = px.line(df_e, x="street", y="equity", markers=True, title="Equity by street")
            
            # Display the additional 3 graphs in a responsive grid layout
            st.markdown("---")  # Add a visual separator
            graph_col1, graph_col2 = st.columns(2)
            
            with graph_col1:
                st.plotly_chart(fig2, use_container_width=True)
                st.plotly_chart(fig_e, use_container_width=True)
            
            with graph_col2:
                st.plotly_chart(fig_h, use_container_width=True)
        if "strategy_pack" in st.session_state:
            st.subheader("Strategy pack")
            pack_chart = st.session_state.strategy_pack.get("chart", {})
            if pack_chart:
                df_pack = pd.DataFrame(list(pack_chart.items()), columns=["Hand", "Action"])
                st.dataframe(df_pack)

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
    if "score" not in st.session_state:
        st.session_state.score = {"correct": 0, "total": 0}
    if "train_history" not in st.session_state:
        st.session_state.train_history = []
    diff = st.selectbox("Difficulty", ["Easy", "Normal", "Pro"], key="train_diff")
    if st.button("New Question") or "hero_q" not in st.session_state:
        deck = eval7.Deck(); deck.shuffle()
        hero_cards = deck.deal(2)
        board_cards = deck.deal(3)
        st.session_state.hero_q = "".join(str(c) for c in hero_cards)
        st.session_state.board_q = "".join(str(c) for c in board_cards)
        st.session_state.vill_q = 1 if diff == "Easy" else 2
        st.session_state.range_q = 50 if diff == "Easy" else (20 if diff == "Normal" else 10)
    st.write(f"**Hero:** {st.session_state.hero_q}  **Board:** {st.session_state.board_q}")
    choice = st.radio("Your action?", ["RAISE", "CHECK", "FOLD"], key="train_choice")
    if st.button("Submit Answer"):
        hero = cards(st.session_state.hero_q)
        board = cards(st.session_state.board_q)
        eq, _ = equity(hero, board, st.session_state.vill_q, st.session_state.range_q, show_progress=False)
        correct = strict_action(eq)
        st.session_state.score["total"] += 1
        st.session_state.train_history.append(1 if choice == correct else 0)
        if choice == correct:
            st.success("Correct!")
            st.session_state.score["correct"] += 1
        else:
            st.error(f"Wrong. Best action: {correct}")
    st.write(f"Score: {st.session_state.score['correct']} / {st.session_state.score['total']}")
    if st.session_state.train_history:
        df = pd.DataFrame({"q": list(range(1, len(st.session_state.train_history)+1)),
                           "result": st.session_state.train_history})
        df["acc"] = df["result"].expanding().mean()
        fig = px.line(df, x="q", y="acc", range_y=[0,1], title="Training accuracy")
        st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    <script>
    document.addEventListener('keydown',function(e){
        if((e.metaKey||e.ctrlKey) && e.key==='Enter'){
            document.querySelector('button[data-baseweb="button"]').click();
        }
        if((e.metaKey||e.ctrlKey) && e.key==='n'){
            document.querySelector('button[data-baseweb="button"]').click();
        }
    });
    </script>
    """,
    unsafe_allow_html=True,
)

# QuickGTO

QuickGTO is a open source poker command line and Streamlit GUI tool that assists with poker
decisions using Monte Carlo equity simulations. It supports a quick, textual
interface and an interactive GUI, keeping a CSV history of your sessions.

## Features

- **Strict mode** – suggests _RAISE/CHECK/FOLD_ based on equity.
- **Bets mode** – evaluates EV for folding, calling and raising and chooses the
  highest EV action.
- Opponent range percentage filtering.
- Equity histogram output and CSV logging.
- Streamlit GUI with card pickers, real‑time validation and Plotly histogram.
- Mode selector (Basic/Advanced/Equilibrium). Advanced adds adjustable iterations,
  weighted ranges, multiprocessing and extra charts. Equilibrium runs a simple solver.
- Toggle between Hold'em and Short‑Deck.
- Solver accuracy presets (Fast/Balanced/Detailed) controlling iteration count.
- Editable strict‑mode thresholds and persistent defaults in `~/.gto_defaults.json`.

### Pack 2 – Training & visual analytics

- **Training difficulty levels** – Easy, Normal and Pro modes with score history
  tracking and a performance graph.
- **Range-matrix heat maps** – 13×13 grids plus extra charts when using Advanced
  mode.
- **Animated equity graphs** – real-time plots that show equity by street.
- **Pre-flop strategy packs & push/fold charts** – one-click import of common
  solutions.
- **Range grids** – colour-coded layouts for both pre- and post-flop ranges.
- **Library of common spots** – BTN vs BB, SB limp and other setups for instant
  loading.

## Requirements

- Python 3 with `pip` available.
- Libraries: `cython`, `streamlit`, `eval7`, `numpy`, `plotly`.

### Install Python and pip

#### macOS
1. Install Homebrew from <https://brew.sh> if it is not installed.
2. Run `brew install python` to install Python 3 and pip.

#### Linux/Unix
1. Use your package manager, for example on Debian/Ubuntu:
   `sudo apt-get install python3 python3-pip`.
2. Other distributions provide similar packages (`python3` and `python3-pip`).

#### Windows
1. Download Python from <https://www.python.org/downloads/>.
2. Run the installer and ensure "Add Python to PATH" is checked.
3. Open Command Prompt and run `py --version` to confirm installation.

### Install dependencies

After Python and pip are available, install the required packages:

```bash
pip install cython streamlit eval7 numpy plotly
```

`pip` may be named `pip3` on some systems. Use whichever is available.

## Usage

### Command Line

Run QuickGTO interactively in a terminal:


```bash
python gto_helper.py
# or
python3 gto_helper.py
```

You will be prompted for:

1. **Opponents** – number of villains (1–9). Enter `1` for heads‑up mode.
2. **Villain range %** – e.g. `0` for random hands, `20` for top 20%.
3. **Hero hand** – exactly two cards, spaced (`Ah Ks`) or concatenated (`AhKs`).
4. **Board cards** – 0, 3, 4 or 5 cards in the same format.
5. **Mode** – enter `s` for strict or `b` for bets.

If you choose bets mode, you will also enter:

- **Current pot**
- **Facing bet**
- **Your stack**
- **Raise size** (`0.5`, `1`, `2` or `shove`)

The program prints JSON output with the recommended action, equity and
histogram. Session data is appended to `gto_history.csv`.

### GUI with Streamlit

Launch the GUI with:

```bash
streamlit run app.py
```

The web interface allows card entry via text or dropdown pickers and shows the
same results with a Plotly equity distribution graph. The sidebar contains form
controls and a theme configuration hint. A CSV download button appears after a
simulation.
Select **Basic**, **Advanced** or **Equilibrium** mode from the sidebar. Advanced exposes
extra controls like iteration count and weighted ranges, while Equilibrium runs a
simple solver demonstration.

## Strict vs Bets mode

- **Strict** – uses preset thresholds. `RAISE` if equity ≥ 65%, `CHECK` if equity
  ≥ 40%, else `FOLD`.
- **Bets** – calculates EV for folding, calling and raising using your pot and
  stack sizes. The action with the highest EV is recommended and EV values are
  displayed.

Both modes log results to `gto_history.csv` for later review.

• CC BY-NC-SA 4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Course project (情報工学特論, group GW9班): an UNO environment plus five tabular RL learners. Python 3.10+, standard library only (no install step, no linter). The MDP definition follows the slides in `../GW1_9班.pptx`; the algorithms follow the figures in `../tokuron-0*.pdf` (textbook fig numbers are cited in each algo's docstring).

## Commands

```
python test_env.py                                   # the only test; must print ALL OK (15 s to ~1 min)
python run.py --algo all --episodes 100000 --opp rule --jobs 5
python run.py --algo sarsa --episodes 1000000 --seed 3
python run.py --algo all --episodes 100000 --opp random+rule --jobs 5   # curriculum: --episodes per stage, one agent
python watch.py --algo sarsa --episodes 20000        # learning curve + one greedy game move by move with Q values
python watch.py --algo sarsa --episodes 200000 --trace-seed 3 --html   # animated replay -> replay_sarsa_200000.html (open in a browser)
python train_all.py                                  # model library: 5 methods x 2 training bots x 7 sizes -> models/*.json (about 3 min, resumable)
python build_page.py                                 # bundle models + web/engine.js into uno_player.html (8 MB, open in a browser)
python check_js.py                                   # replay Python games in node: web/engine.js must match game.py/env.py/opponents.py exactly
node web/model_test.js                               # saved models played in JS reproduce the win rates measured in Python
python analyze.py --vs rule                          # behaviour table from results/*_q.json -> results/analysis.csv
python run.py --algo sarsa --rule plus4_needs_no_color=True --rule "stacking=()"   # --rule values go through ast.literal_eval
```

There is no single-test runner: `test_env.py` is one script. Edit its `VARIANTS` dict to narrow it down.

## Architecture

Three layers; each change belongs in exactly one:

- `game.py` — engine. `Rules` defaults on this EXPERIMENT branch: 2 colours, numbers 0–9, skip/+2/+4, no hand limit (deck = 50 cards, 5,120 states); `main` uses numbers 1–2 and a 12-card limit (20 cards); the full 108-card game is `Rules(**FULL)` as defined in `test_env.py`. `Rules` is a frozen dataclass; every rule is a field, so change a rule by changing a field, never the engine. `Game.legal_moves(p)` / `Game.apply(move)` with moves `("play", card, colour)`, `("draw",)`, `("pass",)`. Cards are `(colour, value)`; colour 4 = wild. Invariant: `total_cards() == n_cards` at all times.
- `env.py` — MDP layer for seat 0 (the agent). `Abstraction.encode` maps a game to a bucketed state tuple and groups legal moves by abstract action (card kind × colour *rank*, plus draw/pass); `UnoEnv.step(a)` picks a random concrete move among those equivalent under the abstraction, then runs the opponents until it is seat 0's turn again. Reward is +1 win / −1 loss / 0 otherwise (including draws at `max_turns`). Changing state buckets or actions happens only here, and `test_env.py` asserts the state/action sizes, so update its numbers when the MDP changes on purpose.
- `algos/<name>.py` — each learner is a module exposing `HP` (dict) and `episode(ag, env) -> final reward`. Registered in `algos/__init__.py` (`ALGOS`). Shared `Agent` (Q table as `dict state -> [value per action]`, Boltzmann `select` over *legal* actions only, own RNG), `train` (geometric τ decay from `tau0` to `tau_min`) and `evaluate` (greedy, no learning) live in `algos/common.py`. Optional module flag `BID_SCALE = True` makes `select` rank by `c_bid * Q` (bucket brigade).

`run.py` is the glue: builds `Rules` from `--rule`, trains, evaluates on a separate seed (`seed + 10_000`), evaluates against BOTH opponents, and writes `results/<algo>_<opp>_s<seed>_r<rulehash>.csv` (learning curve), `.json` (summary, full rules, hyper-parameters) and `_q.json` (Q table, read by `analyze.py`). The rule hash is in the file name so runs under different rules never overwrite each other. With `--algo all`, each algorithm gets seed `--seed + index`. `run_one` must stay a top-level function returning plain dicts (ProcessPoolExecutor pickling).

`watch.py --html` fills `replay_template.html` (plain HTML/JS, no libraries) with a recorded greedy game (per decision: Q of every legal action, shown as bars and as a badge on each hand card); it records via the optional `Game.observer` / `UnoEnv.observer` hooks (called after each applied move; no effect when unset). 2-player games only.

`uno_player.html` (built by `build_page.py` from `web/player_template.html`, `web/engine.js`, `models/*.json`) lets you pick method, episodes, training bot and opponent and watch the model play live. `web/engine.js` is a hand port of the Python game, abstraction and bots: **any change to `game.py`, `env.py` or `opponents.py` must be mirrored there**, then run `python check_js.py` and, since the state/action layout may change, `python train_all.py` (delete `models/` first) and `python build_page.py`. The model files store the Q table as base64 int16 (Q*10000; bucket brigade stores the bid c*Q) indexed by the mixed-radix state number (`Abstraction.ranges`). Only the default `Rules` and 2 players are supported by the page. The page also has a human-vs-model mode (`startHuman` and the `h*` functions): the human is seat 0, the model plays seat 1 via `Abstraction.encode(g, seat)` (mirrored in `env.py`, checked by `check_js.py`), hints are the model's Q for the human's state, and undo uses `Game.snapshot/restore`. UI text lives in the `I18N` / `NAMES` / `ALGO` tables at the top of the page script in `web/player_template.html` (English and Japanese, switch in the header, remembered in `localStorage`); add any new UI string to both languages.

`opponents.py`: `random` and `rule` bots, signature `f(game, seat, legal_moves, rng)`; they obey the same rules as the agent. Add new ones to `OPPONENTS` and `--opp` picks them up.

## Gotchas

- Profit sharing and Monte Carlo treat a draw (reward 0) as "no reward event" and skip the update; this is deliberate (matches the slides).
- In 2-player games REV is folded into SKIP in both the engine and the abstraction; with ≥3 players `Abstraction.kinds` gains `REV`, so state/action counts differ.
- `n_actions` and the state tuple shape depend on `Rules.n_players`; don't hard-code them outside `env.py`.

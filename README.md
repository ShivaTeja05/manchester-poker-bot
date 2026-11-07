# Team axion: heads-up NLHE bot

**For the organisers.** Our one bot is at the repo root, flat next to
`model.py`, exactly as `python -m submit --team axion` lists it:

| File | What it is |
|---|---|
| `axion.pkl` | the pickled `model.Model` (team name: `axion`) |
| `model.py` | the bot |
| `fasteval.py` | fast hand evaluator used by the bot |
| `pftable.py` | precomputed preflop equity table used by the bot |

Checks run on a fresh clone of this repo: `arena.vet` review OK (nothing
refused or flagged), plays legally in `python -m arena` worker processes with
0 faults, `tests/test_model.py` passes, and it runs on Python 3.10 and 3.13.
Standard library only (plus the pinned `pydantic` from `requirements.txt`);
no network, no files, no engine internals. The template files
(`card_utils.py`, `engine.py`, `game_state.py`, ...) are unchanged from the
official repo.

To try it: `python -m arena axion.pkl equity tight --deals 40`

---

# Poker bot hackathon

Write a bot that plays no-limit Texas hold'em. Your bot is a `Model` in
`model.py`. It is pickled and submitted, and the organisers' engine plays it
against other teams' bots.

## 1. Fork and set up

1. **Fork the repo.** Fork https://github.com/Manchester-AI-Society/poker_bot
   on GitHub (your own copy, under your account) and clone **your fork**. Do
   not clone the original: you would have nowhere to keep your work.
2. Create a virtual environment and install the dependencies (Python 3.10+;
   Windows, macOS and Linux all work):

```sh
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt          # Windows: .venv\Scripts\pip ...
```

Run everything with the venv's Python from the repo root: `.venv/bin/python`
(Windows: `.venv\Scripts\python`). In VS Code, select `.venv` as the
interpreter. If `venv` is missing on Ubuntu:
`sudo apt install python3-venv python3-pip`. The commands below say
`.venv/bin/python`; on Windows read it as `.venv\Scripts\python`.

## 2. Implement `model.py`

**`model.py` is the one file you must implement.** Add your own files next to
it if you need them. The template calls every hand, so fill in:

- `seed(hand, player_id)`: your two hole cards and your seat, once per hand.
  Store them.
- `run(actions, game_state)`: on each of your turns, return one
  `PlayerAction`: `fold`, `call` (also checks) or `raise_to`.
- `end_game(actions, game_state)` (optional): called after every hand. Learn
  from it, for example about opponents, using `game_state.teams`.

Rules that matter:

- Keep the model's attributes plain data (numbers, lists, dicts, numpy
  arrays, weights). No lambdas, open files, sockets or threads: the model is
  pickled. Load weights and data in `__init__`.
- Do not edit `card_utils.py`, `player_action.py`, `game_state.py` or
  `game_interface.py`. The organisers' copies are what play.
- Only use the `GameStateView` names listed below: any other name raises
  `AttributeError`.

## 3. What your bot sees

### A hand, step by step

1. Your `Model()` is built and pickled once, before the event. The same
   object then plays hand after hand, so anything you keep in its attributes
   between hands is what it has learned.
2. At the start of each hand, `seed(hand, player_id)` gives you your two hole
   cards and your seat. Store both: `run` is not given them again. Seats
   change between hands; team names (`game_state.teams`) do not.
3. On each of your turns, `run(actions, game_state)` is called:
   - `actions`: every `PlayerAction` since your last turn this hand, oldest
     first, your own previous move and the dealer's cards included;
   - `game_state`: a read-only `GameStateView` of the hand, already updated
     with `actions`.

   Return one `PlayerAction` with your own `player_id`. Anything else (an
   exception, someone else's id, an illegal move, a move over the time limit)
   is a fault: the engine checks for you, or folds if you owe chips.
4. After every hand, `end_game(actions, game_state)` is called (optional):
   the last actions, the cards shown at a showdown and who won what, with the
   final state. Learn from the hand here; `seed` overwrites the per-hand
   fields next hand, so there is nothing to clear.

### Rules as implemented

- Players are numbered `0..n-1`; every per-player tuple is indexed by id.
  The dealer is player `-1` (`player_action.DEALER`).
- Blinds are posted automatically (default 10/20, stacks 500). Heads-up, the
  button posts the small blind and acts first preflop; after the flop the
  first player after the button acts first.
- `raise_to(total)` names your **total bet this street**, not the increment.
  Legal totals are `min_raise_to..max_raise_to`. A raise must be at least the
  last full raise (`last_raise`); an all-in smaller than that does not reopen
  betting for players who already acted.
- Fold and call are always legal on your turn. `call` checks when nothing is
  owed and calls all-in for less when your stack is short.
- At a showdown, every player still in shows both hole cards. A player who
  wins because everyone else folded shows nothing.

### `PlayerAction` (`player_action.py`)

What you return from `run`, always built with a constructor:

| Constructor | Meaning |
|---|---|
| `PlayerAction.fold(player_id)` | fold |
| `PlayerAction.call(player_id)` | call, or check when `call_amount == 0` |
| `PlayerAction.raise_to(player_id, total)` | raise to a street total |

What you receive in `actions`, besides those three from other players:

| `kind` | `player_id` | Meaning |
|---|---|---|
| `"reveal_card"` | `DEALER` | a board card: the 3rd, 4th and 5th open the flop, turn and river |
| `"reveal_card"` | a player | a hole card that player shows at a showdown (after the board) |
| `"win"` | a player | chips that player won this hand (`amount`), from every pot, own uncalled chips included; one per winner |

Fields: `player_id`, `kind` (`"fold"`, `"call"`, `"raise"`, `"reveal_card"`,
`"win"`), `amount` (raise total, or chips won), `card` (for `reveal_card`).
Only the engine builds `reveal_card` and `win`: returning them is illegal.

### `GameStateView` (`game_state.py`)

Every name your bot can read; any other raises `AttributeError`. Lists come
back as tuples, indexed by player id.

| The table | |
|---|---|
| `button`, `small_blind`, `big_blind` | player id; chips; chips |
| `stacks`, `bets`, `folded`, `acted` | per player: chips behind, chips bet this street, bools |
| `teams` | per player: team name. Your team is `teams[self.player_id]`; a team's seat is `teams.index(name)` |
| `board` | Cards dealt so far |
| `street` | `"preflop"`, `"flop"`, `"turn"`, `"river"` |
| `pot` | every chip put in this hand, this street's bets included |
| `last_raise` | size of the last full raise this street |
| `to_act` | player id to act, `None` when the street is closed |
| `num_players`, `players_in_hand`, `active_players` | count; ids not folded; ids not folded and not all-in |
| `is_street_closed`, `is_hand_over` | bools |
| `current_bet` | the street total everyone must reach |

| Any player (methods) | |
|---|---|
| `status(player_id)` | `"active"`, `"folded"` or `"all_in"` |
| `effective_stack(player_id)` | their chips behind, capped by the biggest other stack still in (chips already bet excluded) |
| `spr(player_id)` | `effective_stack / pot` |

| Your turn (inside `run`; not meaningful in `end_game`) | |
|---|---|
| `call_amount` | chips a call adds (0 = check) |
| `min_raise_to`, `max_raise_to`, `raise_range` | legal raise totals; `raise_range` is `None` when you cannot raise |
| `can_raise` | bool |
| `legal_actions` | property, no `()`: e.g. `("fold", "call", "raise")` |
| `pot_odds` | `call_amount / (pot + call_amount)`, `0.0` when checking |
| `validate_action(action)` | raises `InvalidAction` if `action` is illegal |

### Cards and hand strength (`card_utils.py`)

- `Card.parse("AS")`; `card.rank` (`"2".."9"`, `"T"`, `"J"`, `"Q"`, `"K"`,
  `"A"`), `card.suit` (`"C"`, `"D"`, `"H"`, `"S"`), `card.value` (2 to 14, ace
  high), `str(card) == "AS"`. Compare ranks with `.value`.
- `evaluate(cards)` takes 5 to 7 Cards and returns a tuple that compares with
  `<`/`>` (higher wins, equal splits). `rank[0]` indexes `CATEGORIES`, from
  `0` (high card) to `8` (straight flush); `category(rank)` names it. Typical
  use: `evaluate(self.hand + list(game_state.board))`.

### Example bot

```python
from card_utils import evaluate
from player_action import PlayerAction


class Model:
    def __init__(self):
        self.hand = []
        self.player_id = None

    def seed(self, hand, player_id):
        self.hand = hand
        self.player_id = player_id

    def run(self, actions, game_state):
        me = self.player_id
        if len(game_state.board) >= 3:
            # Two pair or better (see card_utils.CATEGORIES).
            strong = evaluate(self.hand + list(game_state.board))[0] >= 2
        else:
            high, low = sorted(card.value for card in self.hand)[::-1]
            strong = high == low or low >= 10  # a pair, or two high cards
        if strong and "raise" in game_state.legal_actions:
            low, high = game_state.raise_range
            # Raise to about the pot, kept within the legal range.
            return PlayerAction.raise_to(me, min(high, max(low, game_state.pot)))
        if game_state.call_amount == 0 or game_state.pot_odds < 0.25:
            return PlayerAction.call(me)
        return PlayerAction.fold(me)
```

### Common mistakes

- Treating `raise_to(x)` as "raise by x": `x` is your total bet this street.
- Calling `legal_actions()` with brackets: it is a property.
- Treating every `reveal_card` as a board card: the board's come from
  `DEALER`, players' shown cards from their own `player_id`.
- Forgetting to store `hand` and `player_id` in `seed`.

## 4. Check and measure your bot

```sh
# Is it a valid submission? (pickles, plays legally, no faults, in time)
.venv/bin/python -m pytest tests/test_model.py -q

# How strong is it? bb/100 against each dummy bot, with a 95% interval
.venv/bin/python -m evals
.venv/bin/python -m evals --opponents tight equity --deals 1000 --table
```

Beating the dummy bots shows your bot works, not that it will win the
tournament: that depends on the other teams. Results are only meaningful when
the interval does not include 0, so add `--deals` to narrow it.

## For the organiser

Everything is under `event/` (`--root` to change it). Python 3.10+ on Windows,
macOS or Linux.

```sh
.venv/bin/python -m arena.event import DOWNLOADS   # the form's downloads
.venv/bin/python -m arena.event status             # who is ready / waiting
.venv/bin/python -m arena.event review TEAM        # findings, then the code
.venv/bin/python -m arena.event approve TEAM       # after reading it
.venv/bin/python -m arena.event run --deals 200    # every bot plays every other
.venv/bin/python -m arena.event export --out site_out --delay 120 --spacing 20 --loop 30
```

`import` reads either one folder per team or the flat folder Google Forms
makes (files named `<file> - <respondent>`, one respondent's files belong
together; a team's name is its `.pkl` name). Each upload is vetted
(`arena/vet.py`) before its code runs: a pickle naming anything but the team's
own modules and plain data is `REFUSED`; a team whose code has flagged lines
(or every team, with `--review-all`) is `NEEDS_REVIEW` until you have read it
with `review` and run `approve`; the rest are played against dummy bots and
become `READY`. A newer upload that fails never replaces one that works.

`run` plays one round: every ready team against every other, heads-up, each
pairing as duplicate deals (seats swapped), so card luck mostly cancels; the
best bb/100 is the champion. It saves every hand, can be stopped and started
again (finished pairings are skipped), and `--concurrency` sets how many
pairings play at once.

`export` writes the results site (the organiser's pages, kept outside this
repository in `site/`, plus `data/`). With
`--delay` and `--spacing` each finished pairing is shown (with a replay of its
biggest hands) some time after it ended and at least that far after the
previous one, so the site feels live but runs behind; standings are computed
from what is shown and the champion appears with the last pairing. `--loop`
keeps exporting; `--git-push` commits and pushes the folder (a git clone that
a static host serves). The Home page text is `event.json`.

Unpickling runs the team's code. Vetting keeps out the obvious attacks and
shows you the code, but it is not a sandbox: for real isolation run the
workers under `--launcher` (for example `bwrap` on Linux, or a container or
VM), on a machine you can spare.

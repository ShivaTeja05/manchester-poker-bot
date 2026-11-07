"""
Sparring bots: plausible styles other hackathon teams might submit, from
weak to strong, for testing our model against more than the stock dummies.

Every class is built as Cls(seed=None) like arena/dummies.py, and only reads
GameStateView names listed in the README. SPARRING_BOTS at the bottom maps a
short name to each class.
"""

import random
from itertools import combinations

from arena.dummies import _Bot
from card_utils import evaluate, full_deck
from player_action import PlayerAction

# --------------------------------------------------------------------------
# Preflop hand ranking: Chen formula over the 169 starting hands, turned into
# "top X%" percentiles over all 1326 combos.
# --------------------------------------------------------------------------

_CHEN_HIGH = {14: 10.0, 13: 8.0, 12: 7.0, 11: 6.0}


def _chen(high: int, low: int, suited: bool) -> float:
    score = _CHEN_HIGH.get(high, high / 2.0)
    if high == low:
        return max(5.0, score * 2)
    if suited:
        score += 2
    gap = high - low - 1
    score -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and high < 12:
        score += 1
    return score


def _key(high: int, low: int, suited: bool) -> tuple:
    # Chen first; ties broken by high card then kicker (Ax ahead in HU).
    return (_chen(high, low, suited), high, low, suited)


def _build_percentiles() -> dict:
    classes = {}
    for hi in range(2, 15):
        for lo in range(2, hi + 1):
            if hi == lo:
                classes[(hi, lo, False)] = 6
            else:
                classes[(hi, lo, True)] = 4
                classes[(hi, lo, False)] = 12
    ordered = sorted(classes, key=lambda c: _key(*c), reverse=True)
    pct, seen = {}, 0
    for cls in ordered:
        seen += classes[cls]
        pct[cls] = seen / 1326.0  # share of combos at least this good
    return pct


_PERCENTILE = _build_percentiles()


def preflop_pct(hand) -> float:
    """Top-X fraction this hand falls in: 0.0045 for AA, 1.0 for the worst."""
    a, b = hand
    hi, lo = max(a.value, b.value), min(a.value, b.value)
    suited = a.suit == b.suit and hi != lo
    return _PERCENTILE[(hi, lo, suited)]


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


class _Base(_Bot):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.rng = random.Random(seed)
        self._eq_cache = None

    def seed(self, hand, player_id):
        super().seed(list(hand), player_id)
        self._eq_cache = None

    # -- actions --
    def _call(self):
        return PlayerAction.call(self.player_id)

    def _shove(self, gs):
        return self._raise_to(gs, gs.max_raise_to if gs.can_raise else 0)

    def _bet_frac(self, gs, frac):
        """Bet/raise to current_bet + frac * (pot after calling)."""
        total = gs.current_bet + int(frac * (gs.pot + gs.call_amount))
        return self._raise_to(gs, total)

    # -- reads --
    def _is_sb(self, gs):
        return gs.button == self.player_id

    def _facing_raise_pf(self, gs):
        return gs.current_bet > gs.big_blind

    def _start_stack(self, gs):
        me = self.player_id
        return gs.stacks[me] + gs.bets[me]

    def _facing_big(self, gs, share=0.35):
        """Facing a bet that commits a big share of our stack (e.g. a shove)."""
        return gs.call_amount >= share * max(1, self._start_stack(gs)) or not gs.can_raise

    def equity(self, board, samples):
        """Win share vs one random hand; cached per street."""
        key = (len(board), samples)
        if self._eq_cache and self._eq_cache[0] == key:
            return self._eq_cache[1]
        hand = self.hand
        unseen = [c for c in full_deck() if c not in hand and c not in board]
        missing = 5 - len(board)
        won = 0.0
        for _ in range(samples):
            cards = self.rng.sample(unseen, missing + 2)
            full = board + cards[:missing]
            mine = evaluate(hand + full)
            theirs = evaluate(cards[missing:] + full)
            if mine > theirs:
                won += 1
            elif mine == theirs:
                won += 0.5
        eq = won / samples
        self._eq_cache = (key, eq)
        return eq

    def _made_hand(self, board):
        """(category, uses_hole_card) for the current board (3+ cards)."""
        cat = evaluate(self.hand + board)[0]
        ranks = {c.rank for c in board}
        uses = (
            self.hand[0].rank == self.hand[1].rank
            or any(c.rank in ranks for c in self.hand)
            or cat >= 4
        )
        if len(board) == 5 and evaluate(board) >= evaluate(self.hand + board):
            uses = False  # playing the board
        return cat, uses

    def _has_draw(self, board):
        cards = self.hand + board
        for suit in "CDHS":
            if sum(c.suit == suit for c in cards) >= 4 and any(
                c.suit == suit for c in self.hand
            ):
                return True
        values = {c.value for c in cards}
        if 14 in values:
            values.add(1)
        hole = {c.value for c in self.hand} | ({1} if any(c.value == 14 for c in self.hand) else set())
        for start in range(1, 11):
            window = set(range(start, start + 5))
            if len(window & values) >= 4 and window & hole:
                return True
        return False


# --------------------------------------------------------------------------
# 1. Nash push/fold
# --------------------------------------------------------------------------


class NashPushFoldBot(_Base):
    """HU push/fold at ~25bb: SB shoves top 33%, BB calls/reshoves top 22%;
    postflop check/calls pair+ else check/folds."""

    PUSH = 0.33
    CALL = 0.22

    def run(self, actions, gs):
        if gs.street == "preflop":
            pct = preflop_pct(self.hand)
            if not self._facing_raise_pf(gs):
                # Unopened (SB first in) or BB after a limp.
                limit = self.PUSH if self._is_sb(gs) else self.CALL
                if pct <= limit:
                    return self._shove(gs)
                return self._check_or_fold(gs)
            if pct <= self.CALL:
                return self._shove(gs) if gs.can_raise else self._call()
            return self._check_or_fold(gs)
        cat, uses = self._made_hand(list(gs.board))
        if cat >= 1 and uses:
            return self._call()
        return self._check_or_fold(gs)


# --------------------------------------------------------------------------
# 2. Tight-aggressive
# --------------------------------------------------------------------------


class TagBot(_Base):
    """Tight-aggressive: chart preflop (open 60% to 2.5bb, 3-bet 12%, 5-bet
    shove 6%), MC equity postflop with 2/3-pot value bets and c-bets."""

    SAMPLES = 150

    def seed(self, hand, player_id):
        super().seed(hand, player_id)
        self.aggressor = False
        self.cbet_street = None

    def run(self, actions, gs):
        if gs.street == "preflop":
            return self._preflop(gs)
        return self._postflop(gs)

    def _raise(self, gs, total):
        self.aggressor = True
        return self._raise_to(gs, total)

    def _preflop(self, gs):
        pct = preflop_pct(self.hand)
        bb = gs.big_blind
        me = self.player_id
        if not self._facing_raise_pf(gs):
            if self._is_sb(gs):  # first in
                if pct <= 0.60:
                    return self._raise(gs, int(2.5 * bb))
                return self._check_or_fold(gs)
            # BB vs limp
            if pct <= 0.25:
                return self._raise(gs, 3 * bb)
            return self._call()
        # Facing a raise.
        if self._facing_big(gs):
            if pct <= 0.15:
                return self._shove(gs) if gs.can_raise else self._call()
            self.aggressor = False
            return self._check_or_fold(gs)
        i_raised = gs.bets[me] > bb  # we opened / raised already
        if i_raised:  # facing a 3-bet
            if pct <= 0.06:
                return self._shove(gs)
            if pct <= 0.12:
                self.aggressor = False
                return self._call()
            return self._check_or_fold(gs)
        if pct <= 0.12:
            return self._raise(gs, 3 * gs.current_bet)
        if pct <= 0.45:
            self.aggressor = False
            return self._call()
        return self._check_or_fold(gs)

    def _postflop(self, gs):
        board = list(gs.board)
        eq = self.equity(board, self.SAMPLES)
        if gs.call_amount == 0:
            if eq > 0.65:
                return self._bet_frac(gs, 2 / 3)
            if (
                self.aggressor
                and gs.street == "flop"
                and self.cbet_street is None
            ):
                self.cbet_street = "flop"
                if self.rng.random() < 0.6:
                    return self._bet_frac(gs, 2 / 3)
            return self._call()
        if eq > 0.85 and gs.can_raise:
            return self._bet_frac(gs, 1.0)
        if eq > gs.pot_odds + 0.05:
            return self._call()
        return self._check_or_fold(gs)


# --------------------------------------------------------------------------
# 3. Loose-aggressive
# --------------------------------------------------------------------------


class LagBot(_Base):
    """Loose-aggressive: opens 90% to 3bb, 3-bets 30%, bets pot 70% when
    checked to, bluff-raises 20%, folds weak hands to big bets."""

    SAMPLES = 100

    def run(self, actions, gs):
        bb = gs.big_blind
        if gs.street == "preflop":
            pct = preflop_pct(self.hand)
            if not self._facing_raise_pf(gs):
                if self._is_sb(gs):
                    if pct <= 0.90:
                        return self._raise_to(gs, 3 * bb)
                    return self._check_or_fold(gs)
                if pct <= 0.5 or self.rng.random() < 0.2:
                    return self._raise_to(gs, 3 * bb)
                return self._call()
            if self._facing_big(gs):
                if pct <= 0.30:
                    return self._shove(gs) if gs.can_raise else self._call()
                return self._check_or_fold(gs)
            if pct <= 0.20 or self.rng.random() < 0.10:
                return self._raise_to(gs, 3 * gs.current_bet)
            if pct <= 0.75:
                return self._call()
            return self._check_or_fold(gs)
        eq = self.equity(list(gs.board), self.SAMPLES)
        if gs.call_amount == 0:
            if eq > 0.6 or self.rng.random() < 0.7:
                return self._bet_frac(gs, 1.0)
            return self._call()
        if eq > 0.75 and gs.can_raise:
            return self._bet_frac(gs, 1.0)
        big = gs.pot_odds > 0.3  # bet of ~3/4 pot or more
        if not big and gs.can_raise and self.rng.random() < 0.2:
            return self._bet_frac(gs, 1.0)  # bluff raise
        if big and eq < 0.5:
            return self._check_or_fold(gs)
        if eq > gs.pot_odds * 0.8:
            return self._call()
        return self._check_or_fold(gs)


# --------------------------------------------------------------------------
# 4. Calling station
# --------------------------------------------------------------------------


class StationBot(_Base):
    """Loose-passive: calls with any pair, draw or ace-high, raises only
    with two pair+, folds pure air to bets over half pot."""

    def run(self, actions, gs):
        if gs.street == "preflop":
            pct = preflop_pct(self.hand)
            if gs.call_amount <= 3 * gs.big_blind:
                return self._call()
            if self._facing_big(gs, 0.5):
                return self._call() if pct <= 0.35 else self._check_or_fold(gs)
            return self._call() if pct <= 0.6 else self._check_or_fold(gs)
        board = list(gs.board)
        cat, uses = self._made_hand(board)
        if cat >= 2 and uses:
            return self._bet_frac(gs, 1.0)
        has_ace = any(c.value == 14 for c in self.hand)
        if (cat >= 1 and uses) or self._has_draw(board) or has_ace:
            return self._call()
        if gs.call_amount == 0:
            return self._call()
        if gs.pot_odds <= 0.25:  # bet of at most half pot
            return self._call()
        return self._check_or_fold(gs)


# --------------------------------------------------------------------------
# 5. Min-raiser
# --------------------------------------------------------------------------


class MinRaiserBot(_Base):
    """Min-raises every street with top-half hands (equity vs random >= 0.5,
    at most 3 raises a street), calls with >35% equity, else check/folds."""

    SAMPLES = 150

    def seed(self, hand, player_id):
        super().seed(hand, player_id)
        self.raises = {}

    def run(self, actions, gs):
        eq = self.equity(list(gs.board), self.SAMPLES)
        n = self.raises.get(gs.street, 0)
        if eq >= 0.5 and gs.can_raise and n < 3:
            self.raises[gs.street] = n + 1
            return self._raise_to(gs, gs.min_raise_to)
        if eq >= 0.5 or eq > 0.35:
            return self._call()
        return self._check_or_fold(gs)


# --------------------------------------------------------------------------
# 6. Shove top 50%
# --------------------------------------------------------------------------


class Shover50Bot(_Base):
    """Shoves preflop with the top 50%, folds the rest; calls a shove (or
    reshoves over a raise) with the top 30%. Postflop: pair+ calls."""

    def run(self, actions, gs):
        if gs.street == "preflop":
            pct = preflop_pct(self.hand)
            if not self._facing_raise_pf(gs):
                if pct <= 0.50:
                    return self._shove(gs)
                return self._check_or_fold(gs)
            if pct <= 0.30:
                return self._shove(gs) if gs.can_raise else self._call()
            return self._check_or_fold(gs)
        cat, uses = self._made_hand(list(gs.board))
        if cat >= 1 and uses:
            return self._call()
        return self._check_or_fold(gs)


# --------------------------------------------------------------------------
# 7. Pot-odds equity bot
# --------------------------------------------------------------------------


class PotOddsBot(_Base):
    """EquityBot, stronger: 400 samples, shoves >0.85, raises 3/4 pot
    >0.7, calls when equity beats pot odds."""

    SAMPLES = 400

    def run(self, actions, gs):
        eq = self.equity(list(gs.board), self.SAMPLES)
        if eq > 0.85 and gs.can_raise:
            return self._shove(gs)
        if eq > 0.7 and gs.can_raise:
            return self._bet_frac(gs, 0.75)
        if eq > gs.pot_odds:
            return self._call()
        return self._check_or_fold(gs)


SPARRING_BOTS: dict[str, type] = {
    "nash": NashPushFoldBot,
    "tag": TagBot,
    "lag": LagBot,
    "station": StationBot,
    "minraiser": MinRaiserBot,
    "shover50": Shover50Bot,
    "potodds": PotOddsBot,
}

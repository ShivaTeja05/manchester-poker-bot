import math
import random
import time

from card_utils import Card
from fasteval import all_combos, card_to_int, class_index, eval_cards
from game_state import GameStateView
from pftable import PF_EQUITY
from player_action import PlayerAction

COMBOS = all_combos()
NCOMBOS = len(COMBOS)

FOLD, CALL, RAISE, SHOVE = 0, 1, 2, 3

# Pseudo-counts (fold, call, raise, shove) for what an unknown opponent
# does in each spot: the starting guess before we have seen them play.
PRIORS = {
    "sb_open": (1.2, 0.9, 3.0, 0.9),
    "bb_vs_limp": (0.0, 4.0, 1.4, 0.6),
    "bb_vs_open": (2.4, 2.1, 0.6, 0.9),
    "vs_3bet": (2.4, 1.5, 0.3, 1.8),
    "vs_shove": (3.3, 2.7, 0.0, 0.0),
    "post_open": (0.0, 3.3, 2.3, 0.4),
    "post_vs_bet": (2.4, 2.7, 0.4, 0.5),
}


def _pf_strength_table():
    """Per class: preflop strength in [0, 1], the share of all combos it
    beats in the equity-vs-random ranking (so it is a percentile)."""
    weight = []
    for ci in range(169):
        hi, lo = divmod(ci, 13)
        weight.append(6 if hi == lo else (4 if hi > lo else 12))
    order = sorted(range(169), key=lambda ci: PF_EQUITY[ci])
    strength = [0.0] * 169
    below = 0
    for ci in order:
        strength[ci] = (below + weight[ci] / 2) / 1326
        below += weight[ci]
    return strength


PF_STRENGTH = _pf_strength_table()
COMBO_PF = [PF_STRENGTH[class_index(a, b)] for a, b in COMBOS]


def _sig(x):
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


class Model:
    """
    Heads-up no-limit hold'em bot.

    Each hand it keeps a weighted range over the opponent's 1326 hands,
    narrowed by every move they make (using how often this opponent
    folds, calls and raises in that spot), and picks the move with the
    best expected value: its equity against that range, the chance the
    opponent folds, and its equity against the hands that continue.
    """

    def __init__(self):
        self.hand = []
        self.player_id = None
        self.rng = random.Random(20261107)
        self.stats = {}  # team -> context -> [fold, call, raise, shove]
        self.hands_seen = 0
        self.errors = 0
        self._reset_hand()

    # ----- hand bookkeeping -----

    def _reset_hand(self):
        self.h = []
        self.ready = False
        self.opp = None
        self.opp_team = None
        self.board = []
        self.street = 0
        self.bets = [0, 0]
        self.invested = [0, 0]
        self.stack = [0, 0]
        self.raises = 0
        self.acted = [False, False]
        self.w = None
        self.s = None
        self.s_street = -1
        self.button = 0

    def seed(self, hand: list[Card], player_id: int):
        self.hand = hand
        self.player_id = player_id
        self._reset_hand()
        self.h = [card_to_int(c) for c in hand]

    def _init_hand(self, gs):
        me = self.player_id
        n = gs.num_players
        self.n = n
        self.button = gs.button
        if n != 2:
            self.ready = True
            return
        opp = 1 - me
        self.opp = opp
        self.opp_team = gs.teams[opp]
        if self.opp_team not in self.stats:
            self.stats[self.opp_team] = {
                k: list(v) for k, v in PRIORS.items()
            }
        # Preflop every chip put in is in bets, so this is the start.
        start = [gs.stacks[p] + gs.bets[p] for p in range(2)]
        sb_player = self.button
        bb_player = 1 - self.button
        self.stack = list(start)
        self.bets = [0, 0]
        sb = min(gs.small_blind, start[sb_player])
        bb = min(gs.big_blind, start[bb_player])
        self.bets[sb_player] = sb
        self.bets[bb_player] = bb
        self.stack[sb_player] -= sb
        self.stack[bb_player] -= bb
        self.invested = list(self.bets)
        self.bb = gs.big_blind
        h0, h1 = self.h
        board = set()
        self.w = [
            0.0 if (a == h0 or a == h1 or b == h0 or b == h1) else 1.0
            for a, b in COMBOS
        ]
        self.s = list(COMBO_PF)
        self.s_street = 0
        self.board = []
        self.street = 0
        self.raises = 0
        self.ready = True

    def _freqs(self, ctx):
        c = self.stats[self.opp_team][ctx]
        total = c[0] + c[1] + c[2] + c[3]
        return [x / total for x in c]

    def _context(self, p):
        """The spot player p is in, for the move they are about to make."""
        other = 1 - p
        facing = max(self.bets) - self.bets[p]
        if facing > 0 and self.stack[other] == 0:
            return "vs_shove"
        if self.street == 0:
            if self.raises == 0:
                if p == self.button and not self.acted[p]:
                    return "sb_open"
                if facing == 0:
                    return "bb_vs_limp"
            if self.raises == 1 and p != self.button and facing > 0:
                return "bb_vs_open"
            if facing > 0:
                return "vs_3bet"
            return None
        if facing == 0:
            return "post_open"
        return "post_vs_bet"

    def _apply(self, a, learn=True):
        """Replays one action onto our copy of the hand."""
        kind = a.kind
        if kind == "reveal_card":
            if a.player_id == -1:
                c = card_to_int(a.card)
                self.board.append(c)
                nb = len(self.board)
                new_street = 1 if nb <= 3 else (2 if nb == 4 else 3)
                if new_street != self.street:
                    self.street = new_street
                    self.bets = [0, 0]
                    self.raises = 0
                    self.acted = [False, False]
                if self.w is not None:
                    w = self.w
                    for i, (x, y) in enumerate(COMBOS):
                        if x == c or y == c:
                            w[i] = 0.0
            return
        if kind == "win":
            return
        p = a.player_id
        if p not in (0, 1):
            return
        ctx = None
        outcome = None
        if p == self.opp and learn:
            ctx = self._context(p)
        if kind == "fold":
            outcome = FOLD
        elif kind == "call":
            need = min(max(self.bets) - self.bets[p], self.stack[p])
            self.bets[p] += need
            self.invested[p] += need
            self.stack[p] -= need
            outcome = CALL
        elif kind == "raise":
            add = a.amount - self.bets[p]
            add = max(0, min(add, self.stack[p]))
            self.bets[p] += add
            self.invested[p] += add
            self.stack[p] -= add
            self.raises += 1
            outcome = SHOVE if self.stack[p] == 0 else RAISE
        self.acted[p] = True
        if ctx is not None and outcome is not None:
            try:
                self._update_range(ctx, outcome)
            except Exception:
                pass
            self.stats[self.opp_team][ctx][outcome] += 1.0

    # ----- the opponent's range -----

    def _strengths(self):
        """Strength of every opponent combo on the current street, cached:
        preflop percentile, or made-hand percentile plus a draw bonus."""
        if self.s_street == self.street:
            return self.s
        board = self.board
        w = self.w
        vals = []
        for i, (a, b) in enumerate(COMBOS):
            if w[i] > 0:
                vals.append((eval_cards([a, b] + board), i))
        vals.sort()
        s = [0.0] * NCOMBOS
        n = len(vals)
        j = 0
        while j < n:
            k = j
            while k + 1 < n and vals[k + 1][0] == vals[j][0]:
                k += 1
            pct = ((j + k) / 2 + 0.5) / max(1, n)
            for t in range(j, k + 1):
                s[vals[t][1]] = pct
            j = k + 1
        if self.street < 3:
            fd_bonus = 0.22 if self.street == 1 else 0.12
            sd_bonus = 0.14 if self.street == 1 else 0.07
            bsuits = [0, 0, 0, 0]
            bmask = 0
            for c in board:
                bsuits[c & 3] += 1
                bmask |= 1 << (c >> 2)
            for _, i in vals:
                a, b = COMBOS[i]
                if s[i] > 0.9:
                    continue
                bonus = 0.0
                sa = a & 3
                sb = b & 3
                if sa == sb:
                    if bsuits[sa] == 2:
                        bonus += fd_bonus
                else:
                    if bsuits[sa] == 3 and (a >> 2) >= 9:
                        bonus += fd_bonus * 0.8
                    if bsuits[sb] == 3 and (b >> 2) >= 9:
                        bonus += fd_bonus * 0.8
                m = bmask | 1 << (a >> 2) | 1 << (b >> 2)
                hole = 1 << (a >> 2) | 1 << (b >> 2)
                wins = 0
                for lo in range(-1, 9):
                    if lo == -1:
                        win = 0b1000000001111
                    else:
                        win = 0b11111 << lo
                    if bin(m & win).count("1") == 4 and (hole & win) and (
                            bin(bmask & win).count("1") < 4):
                        wins += 1
                if wins >= 2:
                    bonus += sd_bonus
                elif wins == 1:
                    bonus += sd_bonus * 0.45
                s[i] = min(0.97, s[i] + bonus)
        self.s = s
        self.s_street = self.street
        return s

    def _wquantile(self, p):
        """The strength below which a share p of the range's weight lies."""
        s = self._strengths()
        w = self.w
        items = sorted((s[i], w[i]) for i in range(NCOMBOS) if w[i] > 0)
        total = sum(x[1] for x in items)
        if total <= 0:
            return 0.5
        target = p * total
        acc = 0.0
        for v, wt in items:
            acc += wt
            if acc >= target:
                return v
        return items[-1][0]

    def _update_range(self, ctx, outcome):
        if outcome == FOLD or self.w is None:
            return
        f = self._freqs(ctx)
        s = self._strengths()
        w = self.w
        temp = 0.05 if self.street == 0 else 0.07
        aggressive = f[2] + f[3]
        if outcome == SHOVE:
            share = f[3] if f[3] > 0.02 else aggressive
            share = min(0.98, max(0.03, share))
            q = self._wquantile(1 - share)
            floor = 0.06
            for i in range(NCOMBOS):
                if w[i] > 0:
                    w[i] *= floor + (1 - floor) * _sig((s[i] - q) / temp)
        elif outcome == RAISE:
            share = min(0.98, max(0.03, aggressive))
            q = self._wquantile(1 - share)
            floor = 0.1
            for i in range(NCOMBOS):
                if w[i] > 0:
                    w[i] *= floor + (1 - floor) * _sig((s[i] - q) / temp)
        elif outcome == CALL:
            facing_free = ctx in ("bb_vs_limp", "post_open")
            if facing_free:
                q = self._wquantile(1 - min(0.95, max(0.03, aggressive)))
                for i in range(NCOMBOS):
                    if w[i] > 0:
                        w[i] *= 1 - 0.55 * _sig((s[i] - q) / temp)
            else:
                lo = min(0.95, max(0.0, f[0]))
                q_lo = self._wquantile(lo) if lo > 0.02 else -1.0
                q_hi = self._wquantile(1 - min(0.95, max(0.02, aggressive)))
                for i in range(NCOMBOS):
                    if w[i] > 0:
                        keep = 0.05 + 0.95 * _sig((s[i] - q_lo) / temp)
                        keep *= 1 - 0.45 * _sig((s[i] - q_hi) / temp)
                        w[i] *= keep

    # ----- equity -----

    def _equity(self, budget):
        """Samples (strength, outcome) against the weighted range, most
        likely first: E against any slice of it by strength."""
        s = self._strengths()
        w = self.w
        board = self.board
        hero = self.h
        idx = [i for i in range(NCOMBOS) if w[i] > 0]
        if not idx:
            idx = [i for i, (a, b) in enumerate(COMBOS)
                   if a not in hero and b not in hero
                   and a not in board and b not in board]
            for i in idx:
                w[i] = 1.0
        need = 5 - len(board)
        res = []
        if need == 0:
            mine = eval_cards(hero + board)
            for i in idx:
                a, b = COMBOS[i]
                o = eval_cards([a, b] + board)
                res.append((s[i], w[i],
                            1.0 if mine > o else (0.5 if mine == o else 0.0)))
        else:
            dead = set(hero) | set(board)
            deck = [c for c in range(52) if c not in dead]
            weights = [w[i] for i in idx]
            deadline = time.perf_counter() + budget
            rng = self.rng
            batch = 200
            total = 0
            while total < 3000:
                picks = rng.choices(idx, weights=weights, k=batch)
                for i in picks:
                    a, b = COMBOS[i]
                    run = rng.sample(deck, need + 2)
                    if a in run or b in run:
                        run = [c for c in run if c != a and c != b]
                    full = board + run[:need]
                    mine = eval_cards(hero + full)
                    o = eval_cards([a, b] + full)
                    res.append((s[i], 1.0, 1.0 if mine > o else (
                        0.5 if mine == o else 0.0)))
                total += batch
                if total >= 600 and time.perf_counter() > deadline:
                    break
        # Strongest first, ties in random order.
        rng = self.rng
        res = [(x[0] + rng.random() * 1e-6, x[1], x[2]) for x in res]
        res.sort(key=lambda x: -x[0])
        tw = sum(x[1] for x in res)
        cum = []
        acc_w = 0.0
        acc_o = 0.0
        for _, wt, o in res:
            acc_w += wt
            acc_o += wt * o
            cum.append((acc_w / tw, acc_o, acc_w))
        return cum

    @staticmethod
    def _band(cum, lo, hi):
        """Equity against the slice of the range from the top lo share to
        the top hi share (0 = strongest)."""
        if not cum:
            return 0.5
        lo = max(0.0, min(1.0, lo))
        hi = max(0.0, min(1.0, hi))
        if hi - lo < 1e-9:
            hi = min(1.0, lo + 0.02)
            if hi - lo < 1e-9:
                lo = max(0.0, hi - 0.02)

        def at(x):
            # (cumulative outcome, cumulative weight) at share x
            if x <= 0:
                return 0.0, 0.0
            n = len(cum)
            l, r = 0, n - 1
            while l < r:
                m = (l + r) // 2
                if cum[m][0] >= x:
                    r = m
                else:
                    l = m + 1
            return cum[l][1], cum[l][2]

        o1, w1 = at(lo)
        o2, w2 = at(hi)
        if w2 - w1 <= 0:
            return cum[-1][1] / cum[-1][2]
        return (o2 - o1) / (w2 - w1)

    # ----- deciding -----

    def run(self, actions: list[PlayerAction],
            game_state: GameStateView) -> PlayerAction:
        me = self.player_id
        try:
            if not self.ready:
                self._init_hand(game_state)
            if self.n != 2:
                return self._multiway(game_state)
            for a in actions:
                self._apply(a)
            return self._decide(game_state)
        except Exception:
            self.errors += 1
            return self._fallback(game_state)

    def _fallback(self, gs):
        me = self.player_id
        try:
            if gs.call_amount == 0:
                return PlayerAction.call(me)
            if gs.pot_odds < 0.25:
                return PlayerAction.call(me)
        except Exception:
            pass
        return PlayerAction.fold(me)

    def _realize(self, eq, ip, street):
        if street == 0:
            base = 0.98 if ip else 0.84
        elif street == 1:
            base = 0.97 if ip else 0.89
        elif street == 2:
            base = 0.99 if ip else 0.94
        else:
            return 1.0
        r = base + (eq - 0.5) * 0.3
        return max(0.55, min(1.15, r))

    def _decide(self, gs):
        me = self.player_id
        opp = self.opp
        pot = gs.pot
        to_call = gs.call_amount
        my_stack = gs.stacks[me]
        opp_stack = gs.stacks[opp]
        my_bet = gs.bets[me]
        opp_bet = gs.bets[opp]
        street = self.street
        ip = me == self.button
        bb = self.bb

        cum = self._equity(0.12 if street == 0 else 0.18)
        eq = self._band(cum, 0.0, 1.0)

        invested_me = self.invested[me]
        invested_opp = self.invested[opp]
        eff_total = min(invested_me + my_stack, invested_opp + opp_stack)

        def allin_ev(e):
            # Both all-in: we win e of the final pot, paying the rest of
            # our effective stack.
            return e * 2 * eff_total - (eff_total - invested_me)

        options = {}
        if to_call > 0:
            options["fold"] = 0.0
        # Call or check.
        c = min(to_call, my_stack)
        pot_eff = pot - (to_call - c)
        if opp_stack == 0 or c >= my_stack or street == 3 and to_call > 0:
            options["call"] = eq * (pot_eff + c) - c
        else:
            r = self._realize(eq, ip, street)
            if street == 3:
                r = 1.0
            options["call"] = r * eq * (pot_eff + c) - c

        raises = {}
        if gs.can_raise and opp_stack > 0:
            low, high = gs.raise_range
            cap = min(high, opp_bet + opp_stack)
            if cap >= low:
                for x in self._sizes(gs, low, cap):
                    raises[x] = self._raise_ev(
                        x, gs, cum, eq, cap, allin_ev, ip)
        for x, v in raises.items():
            options[x] = v

        # Small noise so close spots are mixed.
        scale = 0.004 * pot
        best = None
        best_v = None
        for k, v in options.items():
            v2 = v + self.rng.gauss(0, scale)
            if best_v is None or v2 > best_v:
                best, best_v = k, v2
        if best == "fold":
            return PlayerAction.fold(me)
        if best == "call":
            return PlayerAction.call(me)
        return PlayerAction.raise_to(me, int(best))

    def _sizes(self, gs, low, cap):
        me = self.player_id
        pot = gs.pot
        bb = self.bb
        my_bet = gs.bets[me]
        opp_bet = gs.bets[self.opp]
        street = self.street
        out = set()
        if street == 0:
            if self.raises == 0 and me == self.button:
                out.add(int(2.5 * bb))
                out.add(2 * bb)
            elif self.raises == 0:
                out.add(4 * bb)
            elif self.raises == 1:
                out.add(int(opp_bet * 3))
        else:
            if opp_bet == 0:
                out.add(int(pot * 0.4))
                out.add(int(pot * 0.7))
                out.add(int(pot * 1.0))
            else:
                out.add(int(opp_bet * 2.8 + (pot - opp_bet - my_bet) * 0.3))
        out.add(cap)
        sizes = set()
        for x in out:
            x = max(low, min(cap, x))
            # Leftover stack too small to fold: just go all-in.
            left = (gs.stacks[me] + my_bet) - x
            if left < 0.3 * (pot + 2 * x - my_bet - opp_bet):
                x = cap
            sizes.add(int(x))
        return sorted(sizes)

    def _raise_ev(self, x, gs, cum, eq, cap, allin_ev, ip):
        me = self.player_id
        opp = self.opp
        pot = gs.pot
        my_bet = gs.bets[me]
        opp_bet = gs.bets[opp]
        my_stack = gs.stacks[me]
        opp_stack = gs.stacks[opp]
        street = self.street
        add = x - my_bet
        all_in = x >= cap
        opp_call = min(x - opp_bet, opp_stack)
        if all_in:
            ctx = "vs_shove"
        elif street == 0:
            if self.raises == 0 and me == self.button:
                ctx = "bb_vs_open"
            else:
                ctx = "vs_3bet"
        else:
            ctx = "post_vs_bet"
        f = self._freqs(ctx)
        pf, pc, pr = f[0], f[1], f[2] + f[3]
        if all_in:
            pc += pr
            pr = 0.0
        # Bigger bets get more folds, smaller fewer, judged by the pot odds
        # the opponent is offered.
        odds = opp_call / max(1.0, pot + add + opp_call)
        if street > 0 or all_in:
            ref = 0.33 if street > 0 else 0.45
            pf = pf * (1 + 1.6 * (odds - ref))
        pf = max(0.01, min(0.95, pf))
        rest = max(1e-9, pc + pr)
        pc, pr = (1 - pf) * pc / rest, (1 - pf) * pr / rest
        ev_fold = pot
        if all_in:
            e_call = self._band(cum, 0.0, pc)
            ev_call = allin_ev(e_call)
            return pf * ev_fold + pc * ev_call
        e_call = self._band(cum, pr, pr + pc)
        final = pot + add + opp_call
        r = self._realize(e_call, ip, street)
        if street == 3:
            r = 1.0
        ev_call = r * e_call * final - add
        ev_raise = -add
        if pr > 0:
            e_r = self._band(cum, 0.0, pr)
            ev_raise = max(-add, allin_ev(e_r))
        return pf * ev_fold + pc * ev_call + pr * ev_raise

    # ----- 3+ players: a plain equity player -----

    def _multiway(self, gs):
        me = self.player_id
        opps = len(gs.players_in_hand) - 1
        board = [card_to_int(c) for c in gs.board]
        hero = self.h
        dead = set(hero) | set(board)
        deck = [c for c in range(52) if c not in dead]
        need = 5 - len(board)
        won = 0.0
        n = 500
        for _ in range(n):
            cards = self.rng.sample(deck, need + 2 * opps)
            full = board + cards[:need]
            mine = eval_cards(hero + full)
            best = 0
            for k in range(opps):
                o = eval_cards(cards[need + 2 * k:need + 2 * k + 2] + full)
                if o > best:
                    best = o
            won += 1.0 if mine > best else (0.5 if mine == best else 0.0)
        eq = won / n
        share = 1.0 / (opps + 1)
        if eq > share * 1.6 and gs.can_raise:
            low, high = gs.raise_range
            total = min(high, max(low, int(gs.pot * 0.75 + gs.current_bet)))
            return PlayerAction.raise_to(me, total)
        if gs.call_amount == 0:
            return PlayerAction.call(me)
        if eq > gs.pot_odds * 1.15:
            return PlayerAction.call(me)
        return PlayerAction.fold(me)

    def end_game(self, actions: list[PlayerAction],
                 game_state: GameStateView):
        try:
            self.hands_seen += 1
            if not self.ready:
                # We never acted: in heads-up only an opponent's
                # small-blind fold ends a hand before our turn.
                if game_state.num_players == 2:
                    opp = 1 - self.player_id
                    team = game_state.teams[opp]
                    if team not in self.stats:
                        self.stats[team] = {
                            k: list(v) for k, v in PRIORS.items()
                        }
                    folded = any(a.kind == "fold" and a.player_id == opp
                                 for a in actions)
                    if folded and opp == game_state.button:
                        self.stats[team]["sb_open"][FOLD] += 1.0
                return
            if self.n != 2:
                return
            for a in actions:
                if a.kind in ("fold", "call", "raise"):
                    self._apply(a)
        except Exception:
            pass
        finally:
            self.w = None
            self.s = None

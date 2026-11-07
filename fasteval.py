"""
Fast 5-7 card hand evaluator on integer cards.

A card is an int 0..51: rank = card >> 2 (0 = deuce .. 12 = ace),
suit = card & 3. eval_cards returns an int; higher is better, equal splits.
Encoding: category << 20 | five 4-bit kickers.
"""

RANKS = "23456789TJQKA"
SUITS = "CDHS"


def card_to_int(card) -> int:
    return RANKS.index(card.rank) * 4 + SUITS.index(card.suit)


def _build_tables():
    straight = [0] * 8192
    top5 = [0] * 8192
    for m in range(8192):
        best = 0
        for hi in range(12, 3, -1):
            need = 0b11111 << (hi - 4)
            if m & need == need:
                best = hi + 1
                break
        if not best and (m & 0b1000000001111) == 0b1000000001111:
            best = 4  # wheel: five high
        straight[m] = best
        code = 0
        n = 0
        for r in range(12, -1, -1):
            if m >> r & 1:
                code = code << 4 | r
                n += 1
                if n == 5:
                    break
        code <<= 4 * (5 - n)
        top5[m] = code
    return straight, top5


STRAIGHT, TOP5 = _build_tables()


def eval_cards(cards) -> int:
    rc = [0] * 13
    s0 = s1 = s2 = s3 = 0
    for c in cards:
        rc[c >> 2] += 1
        s = c & 3
        if s == 0:
            s0 += 1
        elif s == 1:
            s1 += 1
        elif s == 2:
            s2 += 1
        else:
            s3 += 1
    fs = -1
    if s0 >= 5:
        fs = 0
    elif s1 >= 5:
        fs = 1
    elif s2 >= 5:
        fs = 2
    elif s3 >= 5:
        fs = 3
    if fs >= 0:
        m = 0
        for c in cards:
            if c & 3 == fs:
                m |= 1 << (c >> 2)
        st = STRAIGHT[m]
        if st:
            return 8 << 20 | st
        return 5 << 20 | TOP5[m]
    mask = 0
    quad = -1
    trips = []
    pairs = []
    for r in range(12, -1, -1):
        n = rc[r]
        if n:
            mask |= 1 << r
            if n == 2:
                pairs.append(r)
            elif n == 3:
                trips.append(r)
            elif n == 4:
                quad = r
    if quad >= 0:
        rest = mask & ~(1 << quad)
        k = 0
        for r in range(12, -1, -1):
            if rest >> r & 1:
                k = r
                break
        return 7 << 20 | quad << 16 | k << 12
    if trips:
        t = trips[0]
        if len(trips) > 1:
            return 6 << 20 | t << 16 | trips[1] << 12
        if pairs:
            return 6 << 20 | t << 16 | pairs[0] << 12
    st = STRAIGHT[mask]
    if st:
        return 4 << 20 | st
    if trips:
        t = trips[0]
        rest = mask & ~(1 << t)
        return 3 << 20 | t << 16 | (TOP5[rest] >> 12) << 8
    if len(pairs) >= 2:
        p1 = pairs[0]
        p2 = pairs[1]
        rest = mask & ~(1 << p1) & ~(1 << p2)
        return 2 << 20 | p1 << 16 | p2 << 12 | (TOP5[rest] >> 16) << 8
    if pairs:
        p = pairs[0]
        rest = mask & ~(1 << p)
        return 1 << 20 | p << 16 | (TOP5[rest] >> 8) << 4
    return TOP5[mask]


def all_combos():
    """The 1326 two-card combos as (a, b) with a < b."""
    return [(a, b) for a in range(52) for b in range(a + 1, 52)]


def class_index(a: int, b: int) -> int:
    """Index 0..168 of the 169 starting-hand classes: pairs on the
    diagonal, suited above it (hi, lo), offsuit below it (lo, hi)."""
    ra = a >> 2
    rb = b >> 2
    hi = ra if ra > rb else rb
    lo = rb if ra > rb else ra
    if hi == lo:
        return hi * 13 + hi
    if (a & 3) == (b & 3):
        return hi * 13 + lo
    return lo * 13 + hi

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Tuple
import math

from zigzag import Pivot

Direction = Literal["up", "down"]
PatternKind = Literal[
    "impulse","abc","flat","triangle","diagonal",
    "double_top","double_bottom","head_shoulders","inverse_head_shoulders",
    "harmonic_gartley","harmonic_bat","harmonic_butterfly",
    "wxy"
]

@dataclass(frozen=True)
class PatternMatch:
    kind: PatternKind
    direction: Optional[Direction]
    pivots: List[Pivot]
    score: float
    label: str
    details: Dict[str, float | int | str | bool]

# ---------- utility ----------

def _length(p0: Pivot, p1: Pivot) -> float:
    return abs(p1.price - p0.price)

def _dur(p0: Pivot, p1: Pivot) -> int:
    return max(0, p1.i - p0.i)

def _ratio(a: float, b: float) -> float:
    return float('nan') if b == 0 else a / b

def _closest_score(value: float, targets: List[float], tol: float) -> float:
    if value is None or math.isnan(value) or math.isinf(value):
        return 0.0
    d = min(abs(value - t) for t in targets)
    return max(0.0, 1.0 - d / tol)

def _types(seq: List[Pivot]) -> str:
    return ''.join(p.kind for p in seq)  # e.g., "LHLHL"

def _is_up_move(p0: Pivot, p1: Pivot) -> bool:
    return p1.price > p0.price

# ---------- Impulse 1-5 ----------

def detect_impulse(
    pivots: List[Pivot],
    direction: Optional[Direction] = None,
    min_bars_per_wave: int = 3,
    fib_tolerance: float = 0.35,
    overlap_soft_cap: float = 0.7,  # maksymalna kara za overlap
) -> Optional[PatternMatch]:
    if len(pivots) < 6:
        return None
    best: Optional[PatternMatch] = None

    for s in range(0, len(pivots) - 5):
        seq = pivots[s:s+6]  # P0..P5
        dir_guess: Direction = "up" if _is_up_move(seq[0], seq[1]) else "down"
        dir_use = direction or dir_guess
        expected = "LHLHLH" if dir_use == "up" else "HLHLHL"
        if _types(seq) != expected:
            continue

        durs = [_dur(seq[k], seq[k+1]) for k in range(5)]
        if any(d < min_bars_per_wave for d in durs):
            continue

        # Minimalne twarde reguły Elliotta: brak 100% zniesień 1 przez 2 i 3 przez 4
        if dir_use == "up":
            ok = (seq[1].price > seq[0].price and
                  seq[2].price > seq[0].price and
                  seq[3].price > seq[1].price and
                  seq[4].price > seq[2].price and
                  seq[5].price > seq[3].price)
        else:
            ok = (seq[1].price < seq[0].price and
                  seq[2].price < seq[0].price and
                  seq[3].price < seq[1].price and
                  seq[4].price < seq[2].price and
                  seq[5].price < seq[3].price)
        if not ok:
            continue

        L1 = _length(seq[0], seq[1])
        L2 = _length(seq[1], seq[2])
        L3 = _length(seq[2], seq[3])
        L4 = _length(seq[3], seq[4])
        L5 = _length(seq[4], seq[5])
        if min(L1,L2,L3,L4,L5) == 0:
            continue

        r2 = _ratio(L2, L1)
        e3 = _ratio(L3, L1)
        r4 = _ratio(L4, L3)
        e5a = _ratio(L5, L1)
        e5b = _ratio(L5, L3)

        s_r2 = _closest_score(r2, [0.382,0.5,0.618,0.786], fib_tolerance)
        s_e3 = _closest_score(e3, [1.0,1.618,2.0,2.618], fib_tolerance)
        s_r4 = _closest_score(r4, [0.236,0.382,0.5,0.618], fib_tolerance)
        s_e5 = max(
            _closest_score(e5a, [0.618,1.0,1.618,2.618], fib_tolerance),
            _closest_score(e5b, [0.382,0.618,1.0], fib_tolerance),
        )

        # Fala 3 nie najkrótsza
        pen_short3 = 0.4 if L3 < min(L1, L5) else 0.0

        # Płynna kara za overlap 4↔1
        if dir_use == "up":
            overlap_amt = max(0.0, seq[1].price - seq[4].price)
        else:
            overlap_amt = max(0.0, seq[4].price - seq[1].price)
        overlap_frac = overlap_amt / max(1e-9, L3)
        pen_overlap = min(overlap_soft_cap, overlap_frac)

        score01 = max(0.0, (0.27*s_r2 + 0.30*s_e3 + 0.20*s_r4 + 0.23*s_e5) - pen_short3 - pen_overlap)
        score = round(100.0 * score01, 2)

        truncated5 = (seq[5].price <= seq[3].price) if dir_use == "up" else (seq[5].price >= seq[3].price)
        extended3 = e3 >= 1.618 if not math.isnan(e3) else False

        details = {
            "r2": r2, "e3": e3, "r4": r4, "e5a": e5a, "e5b": e5b,
            "pen_short3": pen_short3, "pen_overlap": pen_overlap,
            "truncated5": truncated5, "extended3": extended3
        }

        label = f"Impuls 1–5 ({'wzrost' if dir_use=='up' else 'spadek'})"
        cand = PatternMatch("impulse", dir_use, seq, score, label, details)
        if best is None or cand.score > best.score:
            best = cand

    return best

# ---------- ABC + Flat ----------

def detect_abc(pivots: List[Pivot], min_bars_per_leg: int = 3, fib_tolerance: float = 0.35) -> Optional[PatternMatch]:
    if len(pivots) < 4:
        return None
    best: Optional[PatternMatch] = None
    for s in range(0, len(pivots) - 3):
        seq = pivots[s:s+4]
        for dir_use, expected in (("down", "HLHL"), ("up", "LHLH")):
            if _types(seq) != expected:
                continue
            durs = [_dur(seq[k], seq[k+1]) for k in range(3)]
            if any(d < min_bars_per_leg for d in durs):
                continue
            A = _length(seq[0], seq[1])
            B = _length(seq[1], seq[2])
            C = _length(seq[2], seq[3])
            if min(A,B,C) == 0:
                continue
            rB = _ratio(B, A)
            rC = _ratio(C, A)
            sB = _closest_score(rB, [0.382,0.5,0.618,0.786], fib_tolerance)
            sC = _closest_score(rC, [0.786,1.0,1.272,1.618], fib_tolerance)
            score = round(100.0 * (0.45*sB + 0.55*sC), 2)
            label = f"ABC ({'wzrost' if dir_use=='up' else 'spadek'})"
            cand = PatternMatch("abc", dir_use, seq, score, label, {"rB": rB, "rC": rC})
            if best is None or cand.score > best.score:
                best = cand
    return best

def detect_flat(pivots: List[Pivot], min_bars_per_leg: int = 3, tol: float = 0.2) -> Optional[PatternMatch]:
    """Flat: B ≈ 0.9–1.05 A, C ≈ 0.9–1.1 A (regular) lub B > 1.05 A, C ≈ 1.27–1.618 A (expanded)."""
    if len(pivots) < 4:
        return None
    best: Optional[PatternMatch] = None
    for s in range(0, len(pivots) - 3):
        seq = pivots[s:s+4]
        # Dwa warianty kierunków
        for dir_use, expected in (("down","HLHL"), ("up","LHLH")):
            if _types(seq) != expected:
                continue
            durs = [_dur(seq[k], seq[k+1]) for k in range(3)]
            if any(d < min_bars_per_leg for d in durs):
                continue
            A = _length(seq[0], seq[1])
            B = _length(seq[1], seq[2])
            C = _length(seq[2], seq[3])
            if min(A,B,C) == 0:
                continue
            rB = _ratio(B, A)
            rC = _ratio(C, A)
            # Regular flat
            s_reg = 0.5*_window_score(rB, 0.9, 1.05, tol) + 0.5*_window_score(rC, 0.9, 1.1, tol)
            # Expanded flat
            s_exp = 0.5*_semi_inf_score(rB, 1.05, side="gt", tol=tol) + 0.5*_window_score(rC, 1.27, 1.618, tol)
            score01 = max(s_reg, s_exp)
            kind = "regular" if s_reg >= s_exp else "expanded"
            score = round(100.0 * score01, 2)
            if score <= 0:
                continue
            label = f"Flat {kind} ({'wzrost' if dir_use=='up' else 'spadek'})"
            cand = PatternMatch("flat", dir_use, seq, score, label, {"rB": rB, "rC": rC, "variant": kind})
            if best is None or cand.score > best.score:
                best = cand
    return best

def _window_score(x: float, a: float, b: float, tol: float) -> float:
    """Score 0..1: 1 w środku okna, spada w miarę oddalenia do szerokości tol."""
    if math.isnan(x) or math.isinf(x):
        return 0.0
    if a <= x <= b:
        return 1.0
    # poza oknem – malej liniowo
    if x < a:
        return max(0.0, 1 - (a - x) / tol)
    else:
        return max(0.0, 1 - (x - b) / tol)

def _semi_inf_score(x: float, thr: float, side: str = "gt", tol: float = 0.2) -> float:
    """Score 0..1: (x >= thr) lub (x <= thr) z miękką tolerancją."""
    if math.isnan(x) or math.isinf(x):
        return 0.0
    if side == "gt":
        return 1.0 if x >= thr else max(0.0, 1 - (thr - x)/tol)
    else:
        return 1.0 if x <= thr else max(0.0, 1 - (x - thr)/tol)

# ---------- Triangle A-B-C-D-E ----------

def detect_triangle(pivots: List[Pivot], min_bars_per_leg: int = 2, fib_tolerance: float = 0.35) -> Optional[PatternMatch]:
    """Kontraktujący/rozszerzający trójkąt: 5 pivotów naprzemiennie, legi 0.382–0.786 poprzedniego.
    Prosta heurystyka „kontrakcja/amplituda”: |A-C| > |C-E| dla kontraktującego.
    """
    if len(pivots) < 5:
        return None
    best: Optional[PatternMatch] = None
    for s in range(0, len(pivots) - 4):
        seq = pivots[s:s+5]  # A..E
        if _types(seq) not in ("LHLHL","HLHLH"):
            continue
        durs = [_dur(seq[k], seq[k+1]) for k in range(4)]
        if any(d < min_bars_per_leg for d in durs):
            continue
        L_AB = _length(seq[0], seq[1])
        L_BC = _length(seq[1], seq[2])
        L_CD = _length(seq[2], seq[3])
        L_DE = _length(seq[3], seq[4])
        if min(L_AB,L_BC,L_CD,L_DE) == 0:
            continue
        rB = _ratio(L_BC, L_AB)
        rC = _ratio(L_CD, L_BC)
        rD = _ratio(L_DE, L_CD)
        sB = _closest_score(rB, [0.382,0.5,0.618,0.786], fib_tolerance)
        sC = _closest_score(rC, [0.382,0.5,0.618,0.786], fib_tolerance)
        sD = _closest_score(rD, [0.382,0.5,0.618,0.786], fib_tolerance)
        # Amplituda AC vs CE
        amp_AC = abs(seq[2].price - seq[0].price)
        amp_CE = abs(seq[4].price - seq[2].price)
        contracting = amp_AC > amp_CE
        s_contract = 1.0 if contracting else 0.5  # lekka preferencja kontrakcji
        score = round(100.0 * (0.3*sB + 0.3*sC + 0.3*sD + 0.1*s_contract), 2)
        label = f"Triangle {'contracting' if contracting else 'expanding'}"
        cand = PatternMatch("triangle", None, seq, score, label, {"rB":rB,"rC":rC,"rD":rD,"contracting":contracting})
        if best is None or cand.score > best.score:
            best = cand
    return best

# ---------- Diagonal (leading/ending, klin) ----------

def detect_diagonal(pivots: List[Pivot], min_bars_per_wave: int = 2, fib_tolerance: float = 0.4) -> Optional[PatternMatch]:
    """Prosta heurystyka diagonali: 1-2-3-4-5 z dozwolonym overlap 4↔1 oraz klin (contract/expand)."""
    if len(pivots) < 6:
        return None
    best: Optional[PatternMatch] = None
    for s in range(0, len(pivots) - 5):
        seq = pivots[s:s+6]
        dir_guess: Direction = "up" if seq[1].price > seq[0].price else "down"
        expected = "LHLHLH" if dir_guess == "up" else "HLHLHL"
        if _types(seq) != expected:
            continue
        durs = [_dur(seq[k], seq[k+1]) for k in range(5)]
        if any(d < min_bars_per_wave for d in durs):
            continue
        L1 = _length(seq[0], seq[1]); L2 = _length(seq[1], seq[2]); L3 = _length(seq[2], seq[3])
        L4 = _length(seq[3], seq[4]); L5 = _length(seq[4], seq[5])
        if min(L1,L2,L3,L4,L5) == 0:
            continue
        # Klin: amplituda maleje (contracting) albo rośnie (expanding)
        contracting = (L1 > L3 > L5) if dir_guess=="up" else (L1 > L3 > L5)
        expanding = (L1 < L3 < L5) if dir_guess=="up" else (L1 < L3 < L5)
        wedge_score = 0.7 if contracting else (0.5 if expanding else 0.2)
        # Fibo łagodne
        r2 = _ratio(L2, L1); r4 = _ratio(L4, L3)
        s_r2 = _closest_score(r2, [0.382,0.5,0.618,0.786], fib_tolerance)
        s_r4 = _closest_score(r4, [0.236,0.382,0.5,0.618], fib_tolerance)
        score = round(100.0 * (0.4*wedge_score + 0.3*s_r2 + 0.3*s_r4), 2)
        label = f"Diagonal {'contracting' if contracting else ('expanding' if expanding else 'mixed')} ({'up' if dir_guess=='up' else 'down'})"
        cand = PatternMatch("diagonal", dir_guess, seq, score, label, {"r2":r2,"r4":r4,"contracting":contracting,"expanding":expanding})
        if best is None or cand.score > best.score:
            best = cand
    return best

# ---------- Double Top / Bottom ----------

def detect_double_top_bottom(pivots: List[Pivot], eq_tol: float = 0.01, min_depth_frac: float = 0.02) -> Optional[PatternMatch]:
    """Wykrywa podwójny szczyt/dołek na 3 pivotach typu H-L-H lub L-H-L.
    eq_tol: tolerancja równości (1% = 0.01), min_depth_frac: minimalna głębokość w stosunku do ceny środkowego pivotu.
    """
    if len(pivots) < 3:
        return None
    best: Optional[PatternMatch] = None
    for s in range(0, len(pivots) - 2):
        seq = pivots[s:s+3]
        t = _types(seq)
        if t == "HLH":  # double top
            p1, p2, p3 = seq
            eq = abs(p1.price - p3.price) / max(p1.price, 1e-9)
            depth = (p1.price - p2.price) / max(p1.price, 1e-9)
            if eq <= eq_tol and depth >= min_depth_frac:
                score = round(100.0 * (1 - eq) * min(1.0, depth / (min_depth_frac*2)), 2)
                label = "Double Top"
                cand = PatternMatch("double_top", "down", seq, score, label, {"equal_diff": eq, "depth": depth})
                if best is None or cand.score > best.score:
                    best = cand
        elif t == "LHL":  # double bottom
            p1, p2, p3 = seq
            eq = abs(p1.price - p3.price) / max(p1.price, 1e-9)
            depth = (p2.price - p1.price) / max(p1.price, 1e-9)
            if eq <= eq_tol and depth >= min_depth_frac:
                score = round(100.0 * (1 - eq) * min(1.0, depth / (min_depth_frac*2)), 2)
                label = "Double Bottom"
                cand = PatternMatch("double_bottom", "up", seq, score, label, {"equal_diff": eq, "depth": depth})
                if best is None or cand.score > best.score:
                    best = cand
    return best

# ---------- Head & Shoulders ----------

def detect_head_shoulders(pivots: List[Pivot], shoulder_tol: float = 0.03) -> Optional[PatternMatch]:
    """Top H&S: H-L-H-L-H z H2 (head) > H1 i H3, H1≈H3 (tolerancja), neckline L1-L2.
       Inverse H&S: L-H-L-H-L analo.
    """
    if len(pivots) < 5:
        return None
    best: Optional[PatternMatch] = None
    for s in range(0, len(pivots) - 4):
        seq = pivots[s:s+5]
        t = _types(seq)
        if t == "HLHLH":  # top
            H1,L1,H2,L2,H3 = seq
            if H2.price <= max(H1.price, H3.price):
                continue
            shoulders_eq = abs(H1.price - H3.price) / max(H2.price, 1e-9)
            if shoulders_eq <= shoulder_tol:
                # symetria ramion + podobna wysokość ponad linią szyi
                neckline_slope = (L2.price - L1.price) / max((L2.i - L1.i), 1)
                # score: równość ramion + umiarkowana symetria
                score = round(100.0 * (1 - shoulders_eq), 2)
                details = {"shoulders_eq": shoulders_eq, "neck_slope": neckline_slope}
                cand = PatternMatch("head_shoulders", "down", seq, score, "Head & Shoulders", details)
                if best is None or cand.score > best.score:
                    best = cand
        elif t == "LHLHL":  # inverse
            L1,H1,L2,H2,L3 = seq
            if L2.price >= min(L1.price, L3.price):
                continue
            shoulders_eq = abs(L1.price - L3.price) / max(H1.price, 1e-9)
            if shoulders_eq <= shoulder_tol:
                neckline_slope = (H2.price - H1.price) / max((H2.i - H1.i), 1)
                score = round(100.0 * (1 - shoulders_eq), 2)
                details = {"shoulders_eq": shoulders_eq, "neck_slope": neckline_slope}
                cand = PatternMatch("inverse_head_shoulders", "up", seq, score, "Inverse H&S", details)
                if best is None or cand.score > best.score:
                    best = cand
    return best

# ---------- Harmonics (Gartley, Bat, Butterfly) ----------

def detect_harmonics(pivots: List[Pivot], tol: float = 0.12) -> Optional[PatternMatch]:
    """Wykrywa najlepszy z: Gartley, Bat, Butterfly na bazie X-A-B-C-D (5 pivotów).
    Kierunek up/down wynika z X->A.
    """
    if len(pivots) < 5:
        return None
    best: Optional[PatternMatch] = None

    def score_pattern(X,A,B,C,D, name: str) -> float:
        XA = _length(X,A); AB = _length(A,B); BC = _length(B,C); CD = _length(C,D)
        if min(XA,AB,BC,CD) == 0:
            return 0.0
        rAB = AB/XA
        rBC = BC/AB
        rXD = _length(X,D)/XA  # alternatywnie CD/BC w niektórych regułach
        # Targets per pattern
        if name == "gartley":
            s = 0.35*_window_score(rAB, 0.618-tol, 0.618+tol, tol) + \
                0.25*_window_score(rBC, 0.382, 0.886, tol) + \
                0.40*_window_score(rXD, 0.786-tol, 0.786+tol, tol)
        elif name == "bat":
            s = 0.35*_window_score(rAB, 0.382, 0.5, tol) + \
                0.25*_window_score(rBC, 0.382, 0.886, tol) + \
                0.40*_window_score(rXD, 0.886-tol, 0.886+tol, tol)
        elif name == "butterfly":
            s = 0.35*_window_score(rAB, 0.786-tol, 0.786+tol, tol) + \
                0.25*_window_score(rBC, 0.382, 0.886, tol) + \
                0.40*_window_score(rXD, 1.27-tol, 1.618+tol, tol)
        else:
            s = 0.0
        return s

    for s in range(0, len(pivots) - 4):
        seq = pivots[s:s+5]  # X..D
        t = _types(seq)
        if t not in ("LHLHL","HLHLH"):
            continue
        X,A,B,C,D = seq
        dir_use: Direction = "up" if _is_up_move(X, A) else "down"
        best_name = None
        best_s = 0.0
        for name in ("gartley","bat","butterfly"):
            sc = score_pattern(X,A,B,C,D, name)
            if sc > best_s:
                best_s = sc; best_name = name
        score = round(100.0 * best_s, 2)
        if score <= 0:
            continue
        kind = {"gartley":"harmonic_gartley","bat":"harmonic_bat","butterfly":"harmonic_butterfly"}[best_name]
        label = f"Harmonic {best_name.title()} ({'up' if dir_use=='up' else 'down'})"
        cand = PatternMatch(kind, dir_use, seq, score, label, {})
        if best is None or cand.score > best.score:
            best = cand

    return best

# ---------- W-X-Y (double zigzag) ----------

def detect_wxy(pivots: List[Pivot], min_bars_per_leg: int = 2, fib_tolerance: float = 0.35) -> Optional[PatternMatch]:
    """W-X-Y: ABC – X – ABC. Skanujemy 7 pivotów; X o mniejszej amplitudzie niż A."""
    if len(pivots) < 7:
        return None
    best: Optional[PatternMatch] = None

    def abc_score(seq4: List[Pivot]) -> float:
        A = _length(seq4[0], seq4[1])
        B = _length(seq4[1], seq4[2])
        C = _length(seq4[2], seq4[3])
        if min(A,B,C) == 0:
            return 0.0
        rB = _ratio(B, A); rC = _ratio(C, A)
        sB = _closest_score(rB, [0.382,0.5,0.618,0.786], fib_tolerance)
        sC = _closest_score(rC, [0.786,1.0,1.272,1.618], fib_tolerance)
        return 0.45*sB + 0.55*sC

    for s in range(0, len(pivots) - 6):
        seq = pivots[s:s+7]
        # ABC1 = 0..3, X=3..4 (używamy tylko pivot 4), ABC2 = 4..7 (tu będzie 4 pivoty: 4..7 -> ale mamy 7, więc 4..6?)
        ABC1 = seq[0:4]
        Xp = seq[4]
        ABC2 = seq[3:7]  # nakładamy X w środku – prosta heurystyka
        if _types(ABC1) not in ("LHLH","HLHL"):
            continue
        if _types(ABC2) not in ("LHLH","HLHL"):
            continue
        # min bars per leg
        durs1 = [_dur(ABC1[k], ABC1[k+1]) for k in range(3)]
        durs2 = [_dur(ABC2[k], ABC2[k+1]) for k in range(3)]
        if any(d < min_bars_per_leg for d in durs1+durs2):
            continue
        s1 = abc_score(ABC1)
        s2 = abc_score(ABC2)
        # X powinno być mniejsze niż A (pierwsze A z ABC1)
        A1 = _length(ABC1[0], ABC1[1])
        Xamp = abs(Xp.price - ABC1[3].price)
        pen_X = 0.0 if Xamp < A1 else min(0.5, (Xamp - A1)/max(A1,1e-9))
        score = round(100.0 * max(0.0, (0.5*s1 + 0.5*s2) - pen_X), 2)
        dir_use = "up" if _is_up_move(ABC1[0], ABC1[1]) else "down"
        label = f"W‑X‑Y ({'up' if dir_use=='up' else 'down'})"
        cand = PatternMatch("wxy", dir_use, seq, score, label, {"pen_X": pen_X})
        if best is None or cand.score > best.score:
            best = cand

    return best
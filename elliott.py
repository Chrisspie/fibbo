from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Optional, Dict
from zigzag import Pivot

Direction = Literal["up", "down"]

@dataclass(frozen=True)
class ImpulsePattern:
    direction: Direction
    pivots: List[Pivot]
    score: float
    details: Dict[str, float]

@dataclass(frozen=True)
class AbcPattern:
    direction: Direction
    pivots: List[Pivot]
    score: float
    details: Dict[str, float]

def _closest_fib_score(value: float, targets: List[float], tol: float) -> float:
    if value is None or value != value:
        return 0.0
    dist = min(abs(value - t) for t in targets)
    return max(0.0, 1.0 - dist / tol)

def _safe_ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")

def _length(p0: Pivot, p1: Pivot) -> float:
    return abs(p1.price - p0.price)

def _dur(p0: Pivot, p1: Pivot) -> int:
    return max(0, p1.i - p0.i)

def _types_ok(seq: List[Pivot], expected: List[str]) -> bool:
    return all(seq[k].kind == expected[k] for k in range(len(expected)))

def _basic_monotonicity_ok(seq: List[Pivot], direction: Direction) -> bool:
    """
    Minimalne twarde reguły:
    - up:  P1 > P0, P2 > P0 (2 nie znosi 1 w 100%), P3 > P1, P4 > P2 (4 nie znosi 3 w 100%), P5 > P3
    - down: analogicznie odwrotnie
    """
    if direction == "up":
        return (seq[1].price > seq[0].price and
                seq[2].price > seq[0].price and
                seq[3].price > seq[1].price and
                seq[4].price > seq[2].price and
                seq[5].price > seq[3].price)
    else:
        return (seq[1].price < seq[0].price and
                seq[2].price < seq[0].price and
                seq[3].price < seq[1].price and
                seq[4].price < seq[2].price and
                seq[5].price < seq[3].price)


def detect_impulse(pivots: List[Pivot], direction: Optional[Direction]=None,
                   min_bars_per_wave: int=3, fib_tolerance: float=0.35, overlap_tolerance: float=0.02) -> Optional[ImpulsePattern]:
    if len(pivots) < 6:
        return None
    best = None
    for start in range(0, len(pivots)-5):
        seq = pivots[start:start+6]
        dir_guess = "up" if seq[1].price > seq[0].price else "down"
        dir_use = direction or dir_guess
        expected = ["L","H","L","H","L","H"] if dir_use=="up" else ["H","L","H","L","H","L"]
        if not _types_ok(seq, expected): continue
        durs = [_dur(seq[k], seq[k+1]) for k in range(5)]
        if any(d < min_bars_per_wave for d in durs): continue
        if not _basic_monotonicity_ok(seq, dir_use): continue
        L1,L2,L3,L4,L5 = (_length(seq[i], seq[i+1]) for i in range(5))
        if min(L1,L2,L3,L4,L5) == 0: continue
        r2 = _safe_ratio(L2,L1); e3 = _safe_ratio(L3,L1); r4=_safe_ratio(L4,L3)
        e5a = _safe_ratio(L5,L1); e5b = _safe_ratio(L5,L3); e5 = e5a
        s_r2=_closest_fib_score(r2,[0.382,0.5,0.618,0.786],fib_tolerance)
        s_e3=_closest_fib_score(e3,[1.0,1.618,2.0,2.618],fib_tolerance)
        s_r4=_closest_fib_score(r4,[0.236,0.382,0.5,0.618],fib_tolerance)
        s_e5=max(_closest_fib_score(e5a,[0.618,1.0,1.618,2.618],fib_tolerance),
                 _closest_fib_score(e5b,[0.382,0.618,1.0],fib_tolerance))
        pen_short3 = 0.4 if L3 < min(L1,L5) else 0.0
        overlap_ok = (seq[4].price >= seq[1].price*(1.0-overlap_tolerance)) if dir_use=="up" else (seq[4].price <= seq[1].price*(1.0+overlap_tolerance))
        # Kara za "brak nakładania 4 na 1" – mierzona realną ingerencją w jednostkach ceny
        if dir_use == "up":
            overlap_amt = max(0.0, seq[1].price - seq[4].price)  # ile 4 "wchodzi" poniżej szczytu 1
        else:
            overlap_amt = max(0.0, seq[4].price - seq[1].price)  # ile 4 "wchodzi" powyżej dołka 1

        # Skala – użyjemy długości fali 3, żeby uzyskać bezwymiarową frakcję; bezpiecznik na zero.
        overlap_frac = overlap_amt / max(1e-9, L3)
        # Sufit kary (0.7) – silnie penalizuje duże nakładanie, ale nie zeruje całkiem wyniku
        pen_overlap = min(0.7, overlap_frac)
        score01 = max(0.0, (0.27*s_r2 + 0.30*s_e3 + 0.20*s_r4 + 0.23*s_e5) - pen_short3 - pen_overlap)
        score = round(100.0*score01, 2)
        details = {"r2":r2,"e3":e3,"r4":r4,"e5":e5,"pen_short3":pen_short3,"pen_overlap":pen_overlap}
        cand = ImpulsePattern(direction=dir_use, pivots=seq, score=score, details=details)
        if (best is None) or (cand.score > best.score):
            best = cand
    return best

def detect_abc(pivots: List[Pivot], min_bars_per_leg: int=3, fib_tolerance: float=0.35) -> Optional[AbcPattern]:
    if len(pivots) < 4:
        return None
    best=None
    for start in range(0,len(pivots)-3):
        seq=pivots[start:start+4]
        for dir_use, expected in (("down",["H","L","H","L"]),("up",["L","H","L","H"])):
            if not all(seq[k].kind==expected[k] for k in range(4)): continue
            durs=[max(0,seq[k+1].i-seq[k].i) for k in range(3)]
            if any(d < min_bars_per_leg for d in durs): continue
            A=abs(seq[1].price-seq[0].price); B=abs(seq[2].price-seq[1].price); C=abs(seq[3].price-seq[2].price)
            if min(A,B,C)==0: continue
            rB=B/A; rC=C/A
            sB=_closest_fib_score(rB,[0.382,0.5,0.618,0.786],fib_tolerance)
            sC=_closest_fib_score(rC,[0.786,1.0,1.272,1.618],fib_tolerance)
            score=round(100.0*(0.45*sB+0.55*sC),2)
            cand=AbcPattern(direction=dir_use,pivots=seq,score=score,details={"rB":rB,"rC":rC})
            if (best is None) or (cand.score>best.score): best=cand
    return best
from dataclasses import dataclass
from typing import List, Literal, Optional
import numpy as np
import pandas as pd

Kind = Literal["H", "L"]

@dataclass(frozen=True)
class Pivot:
    i: int
    ts: pd.Timestamp
    price: float
    kind: Kind

def _pct_move(a: float, b: float, use_log: bool) -> float:
    if a <= 0 or b <= 0:
        return (b / a - 1.0) * 100.0 if a != 0 else 0.0
    return (np.log(b) - np.log(a)) * 100.0 if use_log else (b / a - 1.0) * 100.0

def zigzag_from_close(
    df: pd.DataFrame,
    close_col: str = "Close",
    pct_threshold: float = 5.0,
    min_bars_between_pivots: int = 3,
    use_log_change: bool = True,
) -> List[Pivot]:
    if close_col not in df.columns:
        raise ValueError(f"Brak kolumny '{close_col}' w danych. Dostępne kolumny: {list(df.columns)}")
    s = df[close_col].dropna()
    if s.empty:
        return []
    idx = s.index
    prices = s.values.astype(float)
    pivots: List[Pivot] = []
    last_pivot_i = 0
    last_pivot_price = float(prices[0])
    trend: Optional[Literal['up','down']] = None
    extreme_price = last_pivot_price
    extreme_i = 0
    for i in range(1, len(prices)):
        p = float(prices[i])
        if trend is None:
            change = _pct_move(last_pivot_price, p, use_log_change)
            if abs(change) >= pct_threshold and (i - last_pivot_i) >= min_bars_between_pivots:
                trend = "up" if change > 0 else "down"
                extreme_price = p
                extreme_i = i
            else:
                if p > extreme_price:
                    extreme_price = p; extreme_i = i
                elif p < extreme_price:
                    extreme_price = p; extreme_i = i
            continue
        if trend == "up":
            if p >= extreme_price:
                extreme_price = p; extreme_i = i
            reversal = _pct_move(extreme_price, p, use_log_change)
            if reversal <= -pct_threshold and (i - extreme_i) >= min_bars_between_pivots:
                pivots.append(Pivot(i=extreme_i, ts=idx[extreme_i], price=float(extreme_price), kind="H"))
                last_pivot_i = extreme_i; last_pivot_price = extreme_price
                trend = "down"; extreme_price = p; extreme_i = i
        else:
            if p <= extreme_price:
                extreme_price = p; extreme_i = i
            reversal = _pct_move(extreme_price, p, use_log_change)
            if reversal >= pct_threshold and (i - extreme_i) >= min_bars_between_pivots:
                pivots.append(Pivot(i=extreme_i, ts=idx[extreme_i], price=float(extreme_price), kind="L"))
                last_pivot_i = extreme_i; last_pivot_price = extreme_price
                trend = "up"; extreme_price = p; extreme_i = i
    if trend == "up":
        pivots.append(Pivot(i=extreme_i, ts=idx[extreme_i], price=float(extreme_price), kind="H"))
    elif trend == "down":
        pivots.append(Pivot(i=extreme_i, ts=idx[extreme_i], price=float(extreme_price), kind="L"))
    cleaned: List[Pivot] = []
    for pv in pivots:
        if not cleaned:
            cleaned.append(pv); continue
        last = cleaned[-1]
        if pv.kind == last.kind:
            better = pv if (pv.price >= last.price if pv.kind == "H" else pv.price <= last.price) else last
            cleaned[-1] = better
        else:
            cleaned.append(pv)
    if len(cleaned) >= 2 and cleaned[0].kind == cleaned[1].kind:
        cleaned = cleaned[1:]
    return cleaned
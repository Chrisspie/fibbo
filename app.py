# -*- coding: utf-8 -*-
"""
Elliott Waves – Streamlit app v4 (wiele wzorców)
- Impuls 1–5, ABC, Flat (regular/expanded), Triangle A–E, Diagonal (klin),
  Double Top/Bottom, Head & Shoulders, Harmonics (Gartley/Bat/Butterfly), W‑X‑Y
- Higiena danych: MultiIndex z yfinance, tz-naive, sort, dedupe, fallback Adj Close -> Close
"""

from __future__ import annotations
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

from zigzag import zigzag_from_close, Pivot
from elliott import (
    detect_impulse, detect_abc, detect_flat, detect_triangle, detect_diagonal,
    detect_double_top_bottom, detect_head_shoulders, detect_harmonics, detect_wxy,
    PatternMatch
)

# ------------- dane: util -------------

def _flatten_yf_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        try:
            lvl0 = list(df.columns.get_level_values(0))
            lvl1 = list(df.columns.get_level_values(1))
            wanted = {"Open","High","Low","Close","Adj Close","Volume"}
            if (set(lvl0) & wanted) and not (set(lvl1) & wanted):
                df.columns = lvl0
            elif (set(lvl1) & wanted) and not (set(lvl0) & wanted):
                df.columns = lvl1
            else:
                df.columns = [' '.join([str(l) for l in tup if l != '']).strip() for tup in df.columns.values]
        except Exception:
            df.columns = [' '.join([str(l) for l in tup if l != '']).strip() for tup in df.columns.values]
    return df

def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.index = pd.to_datetime(df.index).tz_convert(None)
    except Exception:
        try:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        except Exception:
            df.index = pd.to_datetime(df.index)
    return df

def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["Open","High","Low","Close","Adj Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=False, ttl=600)
def _yf_fetch_debug(ticker: str, period: str) -> Tuple[pd.DataFrame, list]:
    logs = []
    df = pd.DataFrame()

    def log_shape(tag, d):
        try:
            logs.append(f"{tag}: shape={getattr(d, 'shape', None)}, cols={list(getattr(d, 'columns', []))[:8]}")
        except Exception as e:
            logs.append(f"{tag}: err={e}")

    try:
        d1 = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False, group_by='column', threads=False)
        log_shape("download(auto_adjust=False)", d1)
        if d1 is not None and not d1.empty: df = d1
    except Exception as e:
        logs.append(f"download error: {e}")
    if df.empty:
        try:
            d2 = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
            log_shape("history(auto_adjust=False)", d2)
            if d2 is not None and not d2.empty: df = d2
        except Exception as e:
            logs.append(f"history error: {e}")
    if df.empty:
        try:
            d3 = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, group_by='column', threads=False)
            log_shape("download(auto_adjust=True)", d3)
            if d3 is not None and not d3.empty: df = d3
        except Exception as e:
            logs.append(f"download adj error: {e}")
    if df.empty:
        try:
            days = 365
            p = period.strip().lower()
            if p.endswith('y'): days = int(float(p[:-1]) * 365)
            elif p.endswith('mo'): days = int(float(p[:-2]) * 30)
            elif p.endswith('d'): days = int(float(p[:-1]))
            start = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)).date()
            d4 = yf.download(ticker, start=str(start), interval="1d", auto_adjust=True, progress=False, group_by='column', threads=False)
            log_shape(f"download(start={start})", d4)
            if d4 is not None and not d4.empty: df = d4
        except Exception as e:
            logs.append(f"download start error: {e}")

    df = _flatten_yf_cols(df)
    logs.append(f"after flatten: cols={list(df.columns)}")
    df = df[[c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]].copy()
    df = _ensure_numeric(df)
    df = _clean_index(df)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if not df.empty:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    logs.append(f"final: shape={df.shape}, cols={list(df.columns)}")
    return df, logs

def _read_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    date_col = None
    for c in df.columns:
        if str(c).lower() in ("date","data","time","datetime","timestamp"):
            date_col = c; break
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)
    df = df.rename(columns={c: c.title() for c in df.columns})
    df = _ensure_numeric(df)
    df = _clean_index(df)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if "Close" in df.columns:
        df = df[pd.notna(df["Close"])]
    if not df.empty:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df

# ------------- wykres: util -------------

def _candles_figure(df: pd.DataFrame, title: str) -> go.Figure:
    x = pd.to_datetime(df.index)
    try:
        x = x.tz_convert(None)
    except Exception:
        try:
            x = x.tz_localize(None)
        except Exception:
            pass
    fig = go.Figure()
    has_ohlc = {"Open","High","Low","Close"}.issubset(df.columns)
    if "Close" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["Close"], mode="lines", name="Close", line=dict(width=1)))
    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Świece",
            increasing_line_color="green", decreasing_line_color="crimson",
            increasing_fillcolor="green", decreasing_fillcolor="crimson",
            showlegend=True
        ))
    try:
        if has_ohlc:
            lo = float(np.nanmin(df["Low"].values)); hi = float(np.nanmax(df["High"].values))
        else:
            lo = float(np.nanmin(df["Close"].values)); hi = float(np.nanmax(df["Close"].values))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = (hi - lo) * 0.05
            fig.update_yaxes(range=[lo - pad, hi + pad])
    except Exception:
        pass
    fig.update_layout(title=title, xaxis_title="Czas", yaxis_title="Cena",
                      hovermode="x unified", xaxis_rangeslider_visible=False, height=650)
    return fig

def _wave_line(fig: go.Figure, pivots: List[Pivot], name: str) -> None:
    xs = [p.ts for p in pivots]; ys = [p.price for p in pivots]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=name))

def _label_points(fig: go.Figure, seq: List[Pivot], labels: List[str]) -> None:
    for p, lab in zip(seq, labels):
        fig.add_annotation(x=p.ts, y=p.price, text=lab, showarrow=True, arrowhead=1)

def _pivot_table(seq: List[Pivot], labels: List[str]) -> pd.DataFrame:
    return pd.DataFrame([{"Label": lab, "Data": p.ts, "Cena": p.price, "Idx": p.i, "Typ": p.kind} for p,lab in zip(seq, labels)])

# ------------- UI -------------

st.set_page_config(page_title="Fale Elliotta – wykrywanie (pro)", layout="wide")
st.title("🔎 Fale Elliotta – wykrywanie (pro)")
st.caption("Heurystyki: ZigZag + Fibo + geometryczne dopasowania. Edukacyjnie – nie jest to porada inwestycyjna.")

with st.sidebar:
    st.header("Dane")
    source = st.radio("Źródło", ["Yahoo Finance", "CSV"], index=0 if HAS_YF else 1)
    ticker = st.text_input("Ticker (np. AAPL, BTC-USD, PKN.WA)", value="AAPL")
    period = st.selectbox("Okres", ["6mo","1y","2y","5y","10y","max"], index=1)
    st.markdown("---")
    st.header("ZigZag")
    pct_threshold = st.slider("Próg odwrócenia (%)", 1.0, 15.0, 6.0, 0.5)
    min_bars_between = st.slider("Min. świece między pivotami", 1, 30, 3, 1)
    use_log = st.checkbox("Zmiany logarytmiczne", value=True)
    st.markdown("---")
    st.header("Wzorce – wybór")
    patterns = st.multiselect(
        "Co wykrywać?",
        [
            "Impuls 1–5", "ABC", "Flat (regular/expanded)", "Triangle A–E",
            "Diagonal (klin)", "Double Top/Bottom", "Head & Shoulders",
            "Harmonics (Gartley/Bat/Butterfly)", "W‑X‑Y"
        ],
        default=["Impuls 1–5","ABC","Flat (regular/expanded)","Triangle A–E","Diagonal (klin)"]
    )
    st.markdown("---")
    st.header("Parametry dopasowań")
    fib_tol = st.slider("Tolerancja Fibo (±)", 0.05, 0.8, 0.35, 0.05)
    st.markdown("---")
    show_zz = st.checkbox("Pokaż ślad ZigZag", value=True)
    show_tables = st.checkbox("Pokaż tabele dopasowań", value=True)

# ------------- Dane -------------

data_df: Optional[pd.DataFrame] = None
error_load: Optional[str] = None
debug_logs: List[str] = []

if source == "Yahoo Finance":
    if not HAS_YF:
        error_load = "Brak modułu yfinance. Użyj opcji CSV."
    else:
        try:
            data_df, debug_logs = _yf_fetch_debug(ticker, period)
            if data_df is None or data_df.empty:
                error_load = "Brak danych z Yahoo Finance – spróbuj inny ticker/okres."
        except Exception as e:
            error_load = f"Nie udało się pobrać danych: {e}"
else:
    file = st.file_uploader("CSV z Date/Open/High/Low/Close/Volume", type=["csv"])
    if file is not None:
        try:
            data_df = _read_csv(file)
            if data_df.empty:
                error_load = "Plik CSV bez danych."
        except Exception as e:
            error_load = f"Błąd odczytu CSV: {e}"

if error_load:
    st.error(error_load)
    if debug_logs:
        with st.expander("🔧 Debug Yahoo"):
            for line in debug_logs:
                st.text(line)

if data_df is not None and not data_df.empty:
    st.subheader("Wykres")
    title = f"{ticker} – {period}" if source=="Yahoo Finance" else "CSV"
    fig = _candles_figure(data_df, title)
    pivots = zigzag_from_close(data_df, pct_threshold=pct_threshold, min_bars_between_pivots=min_bars_between, use_log_change=use_log)
    if show_zz:
        _wave_line(fig, pivots, f"ZigZag ({pct_threshold:.1f}%)")

    # ------------- Detekcje -------------
    found: List[PatternMatch] = []

    if "Impuls 1–5" in patterns:
        m = detect_impulse(pivots, fib_tolerance=fib_tol)
        if m: 
            found.append(m)
            _wave_line(fig, m.pivots, m.label + f" – score {m.score}")
            _label_points(fig, m.pivots[1:], ["1","2","3","4","5"])

    if "ABC" in patterns:
        m = detect_abc(pivots, fib_tolerance=fib_tol)
        if m:
            found.append(m)
            _wave_line(fig, m.pivots, m.label + f" – score {m.score}")
            _label_points(fig, m.pivots[1:], ["A","B","C"])

    if "Flat (regular/expanded)" in patterns:
        m = detect_flat(pivots)
        if m:
            found.append(m)
            _wave_line(fig, m.pivots, m.label + f" – score {m.score}")
            _label_points(fig, m.pivots[1:], ["A","B","C"])

    if "Triangle A–E" in patterns:
        m = detect_triangle(pivots, fib_tolerance=fib_tol)
        if m:
            found.append(m)
            _wave_line(fig, m.pivots, m.label + f" – score {m.score}")
            _label_points(fig, m.pivots, ["A","B","C","D","E"])

    if "Diagonal (klin)" in patterns:
        m = detect_diagonal(pivots, fib_tolerance=fib_tol)
        if m:
            found.append(m)
            _wave_line(fig, m.pivots, m.label + f" – score {m.score}")
            _label_points(fig, m.pivots[1:], ["1","2","3","4","5"])

    if "Double Top/Bottom" in patterns:
        m = detect_double_top_bottom(pivots)
        if m:
            found.append(m)
            _wave_line(fig, m.pivots, m.label + f" – score {m.score}")
            # Etykiety: L/R lub D1/D2
            if m.kind == "double_top":
                _label_points(fig, m.pivots, ["TopL","Neck","TopR"])
            else:
                _label_points(fig, m.pivots, ["BotL","Neck","BotR"])

    if "Head & Shoulders" in patterns:
        m = detect_head_shoulders(pivots)
        if m:
            found.append(m)
            _wave_line(fig, m.pivots, m.label + f" – score {m.score}")
            if m.kind == "head_shoulders":
                _label_points(fig, m.pivots, ["LS","N1","H","N2","RS"])
            else:
                _label_points(fig, m.pivots, ["LS","N1","H","N2","RS"])

    if "Harmonics (Gartley/Bat/Butterfly)" in patterns:
        m = detect_harmonics(pivots)
        if m:
            found.append(m)
            _wave_line(fig, m.pivots, m.label + f" – score {m.score}")
            _label_points(fig, m.pivots, ["X","A","B","C","D"])

    if "W‑X‑Y" in patterns:
        m = detect_wxy(pivots, fib_tolerance=fib_tol)
        if m:
            found.append(m)
            _wave_line(fig, m.pivots, m.label + f" – score {m.score}")
            _label_points(fig, m.pivots, ["W0","W1","W2","W3","X","Y1","Y2"])

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    if show_tables and found:
        st.subheader("Dopasowania")
        for m in found:
            with st.expander(m.label + f" (score {m.score})"):
                # dynamiczne etykiety
                labs = {
                    "impulse": ["0","1","2","3","4","5"],
                    "abc": ["0","A","B","C"],
                    "flat": ["0","A","B","C"],
                    "triangle": ["A","B","C","D","E"],
                    "diagonal": ["0","1","2","3","4","5"],
                    "double_top": ["L","Neck","R"],
                    "double_bottom": ["L","Neck","R"],
                    "head_shoulders": ["LS","N1","H","N2","RS"],
                    "inverse_head_shoulders": ["LS","N1","H","N2","RS"],
                    "harmonic_gartley": ["X","A","B","C","D"],
                    "harmonic_bat": ["X","A","B","C","D"],
                    "harmonic_butterfly": ["X","A","B","C","D"],
                    "wxy": ["W0","W1","W2","W3","X","Y1","Y2"]
                }.get(m.kind, [str(i) for i in range(len(m.pivots))])
                st.dataframe(_pivot_table(m.pivots, labs), use_container_width=True)
                if m.details:
                    st.json(m.details)

    # Eksport wykresu do HTML
    html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
    st.download_button("📥 Pobierz wykres (HTML)", data=html_bytes, file_name="wykres_elliott.html", mime="text/html")
else:
    st.info("Wczytaj dane z Yahoo Finance lub wgraj plik CSV, aby rozpocząć.")
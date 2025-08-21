# -*- coding: utf-8 -*-
"""
Elliott Waves – Streamlit app (z higieną danych i debug Yahoo)
- Spłaszczanie MultiIndex kolumn z yfinance (odrzucenie poziomu z tickerem)
- Fallback: jeśli brak 'Close', użyj 'Adj Close' jako 'Close'
- Porządkowanie indeksu (tz-naive), sortowanie, deduplikacja
- Panel "🔧 Debug Yahoo Finance" z logami prób pobrania

Uwaga: algorytm rozpoznawania fal opiera się na ZigZag + heurystykach Fibo.
"""

from __future__ import annotations

from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Próba importu yfinance (app wciąż działa z CSV, nawet jeśli yfinance nie ma)
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

from zigzag import zigzag_from_close, Pivot
from elliott import detect_impulse, detect_abc


# ---------------------- UTIL: dane ----------------------
def _flatten_yf_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spłaszcz MultiIndex kolumn zwrócony przez yfinance:
    - jeśli któryś poziom zawiera nazwy OHLCV, wybieramy ten poziom
    - w innym razie łączymy poziomy spacją
    """
    if isinstance(df.columns, pd.MultiIndex):
        try:
            lvl0 = list(df.columns.get_level_values(0))
            lvl1 = list(df.columns.get_level_values(1))
            wanted = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            set0, set1 = set(lvl0), set(lvl1)
            if (set0 & wanted) and not (set1 & wanted):
                df.columns = lvl0
            elif (set1 & wanted) and not (set0 & wanted):
                df.columns = lvl1
            else:
                df.columns = [' '.join([str(l) for l in tup if l != '']).strip()
                              for tup in df.columns.values]
        except Exception:
            df.columns = [' '.join([str(l) for l in tup if l != '']).strip()
                          for tup in df.columns.values]
    return df


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    """Uczyń indeks tz-naive (bez strefy czasowej) i upewnij się, że to datetime."""
    try:
        df.index = pd.to_datetime(df.index).tz_convert(None)
    except Exception:
        try:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        except Exception:
            df.index = pd.to_datetime(df.index)
    return df


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Konwersja kolumn OHLCV do typu numerycznego."""
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False, ttl=600)
def _yf_fetch_debug(ticker: str, period: str) -> Tuple[pd.DataFrame, list]:
    """
    Pobierz dane z Yahoo kilkoma podejściami + zwróć logi do debugowania.
    """
    logs: List[str] = []
    df = pd.DataFrame()

    def log_shape(tag, d):
        try:
            logs.append(f"{tag}: shape={getattr(d, 'shape', None)}, "
                        f"cols={list(getattr(d, 'columns', []))[:8]}")
        except Exception as e:
            logs.append(f"{tag}: err={e}")

    # 1) download (classic)
    try:
        d1 = yf.download(
            ticker, period=period, interval="1d",
            auto_adjust=False, progress=False, group_by='column', threads=False
        )
        log_shape("download(auto_adjust=False)", d1)
        if d1 is not None and not d1.empty:
            df = d1
    except Exception as e:
        logs.append(f"download error: {e}")

    # 2) Ticker().history()
    if df.empty:
        try:
            d2 = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
            log_shape("history(auto_adjust=False)", d2)
            if d2 is not None and not d2.empty:
                df = d2
        except Exception as e:
            logs.append(f"history error: {e}")

    # 3) download(auto_adjust=True)
    if df.empty:
        try:
            d3 = yf.download(
                ticker, period=period, interval="1d",
                auto_adjust=True, progress=False, group_by='column', threads=False
            )
            log_shape("download(auto_adjust=True)", d3)
            if d3 is not None and not d3.empty:
                df = d3
        except Exception as e:
            logs.append(f"download adj error: {e}")

    # 4) download z wyliczonym startem
    if df.empty:
        try:
            days = 365
            p = period.strip().lower()
            if p.endswith('y'):
                days = int(float(p[:-1]) * 365)
            elif p.endswith('mo'):
                days = int(float(p[:-2]) * 30)
            elif p.endswith('d'):
                days = int(float(p[:-1]))
            start = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)).date()
            d4 = yf.download(
                ticker, start=str(start), interval="1d",
                auto_adjust=True, progress=False, group_by='column', threads=False
            )
            log_shape(f"download(start={start})", d4)
            if d4 is not None and not d4.empty:
                df = d4
        except Exception as e:
            logs.append(f"download start error: {e}")

    # Normalizacja kolumn + higiena danych
    df = _flatten_yf_cols(df)
    logs.append(f"after flatten: cols={list(df.columns)}")

    wanted = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[[c for c in wanted if c in df.columns]].copy()
    df = _ensure_numeric(df)
    df = _clean_index(df)

    # Fallback: jeśli nie ma 'Close', użyj 'Adj Close'
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    # Sortuj i deduplikuj indeks
    if not df.empty:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

    logs.append(f"final: shape={df.shape}, cols={list(df.columns)}")
    return df, logs


def _read_csv(file) -> pd.DataFrame:
    """
    Wczytaj CSV (Date/Data/Time/Datetime/Timestamp + OHLCV), uporządkuj typy,
    fallback Close z Adj Close, posprzątaj indeks (sort, dedupe, tz-naive).
    """
    df = pd.read_csv(file)

    # Wykryj kolumnę daty
    date_col = None
    for c in df.columns:
        if str(c).lower() in ("date", "data", "time", "datetime", "timestamp"):
            date_col = c
            break

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)

    # Ujednolicenie nagłówków (np. 'close' -> 'Close')
    rename_map = {c: c.title() for c in df.columns}
    df = df.rename(columns=rename_map)

    # Konwersje i sprzątanie
    df = _ensure_numeric(df)
    df = _clean_index(df)

    # Fallback Close
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    # Usuń wiersze bez ceny
    if "Close" in df.columns:
        df = df[pd.notna(df["Close"])]

    # Sort + dedupe indeksu
    if not df.empty:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

    return df


# ---------------------- UTIL: wykres ----------------------
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
    has_ohlc = {"Open", "High", "Low", "Close"}.issubset(df.columns)

    # Zawsze rysuj linię Close
    if "Close" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["Close"], mode="lines",
                                 name="Close", line=dict(width=1)))

    # Świece – jeśli mamy kompletny OHLC
    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Świece",
            increasing_line_color="green", decreasing_line_color="crimson",
            increasing_fillcolor="green", decreasing_fillcolor="crimson",
            showlegend=True
        ))

    # Sensowny zakres Y z marginesem
    try:
        if has_ohlc:
            lo = float(np.nanmin(df["Low"].values))
            hi = float(np.nanmax(df["High"].values))
        else:
            lo = float(np.nanmin(df["Close"].values))
            hi = float(np.nanmax(df["Close"].values))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = (hi - lo) * 0.05
            fig.update_yaxes(range=[lo - pad, hi + pad])
    except Exception:
        pass

    fig.update_layout(
        title=title, xaxis_title="Czas", yaxis_title="Cena",
        hovermode="x unified", xaxis_rangeslider_visible=False, height=600
    )
    return fig


def _add_zigzag_trace(fig: go.Figure, pivots: List[Pivot], name: str = "ZigZag") -> None:
    if not pivots:
        return
    xs = [p.ts for p in pivots]
    ys = [p.price for p in pivots]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=name))


def _label_points(fig: go.Figure, seq: List[Pivot], labels: List[str]) -> None:
    for p, lab in zip(seq, labels):
        fig.add_annotation(x=p.ts, y=p.price, text=lab, showarrow=True, arrowhead=1)


def _wave_line(fig: go.Figure, seq: List[Pivot], name: str) -> None:
    xs = [p.ts for p in seq]
    ys = [p.price for p in seq]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=name))


def _pattern_table(seq: List[Pivot], labels: List[str]) -> pd.DataFrame:
    return pd.DataFrame([
        {"Fala": lab, "Data": p.ts, "Cena": p.price, "Idx": p.i, "Typ": p.kind}
        for p, lab in zip(seq, labels)
    ])


# ---------------------- UI ----------------------
st.set_page_config(page_title="Fale Elliotta – wykrywanie", layout="wide")
st.title("🔎 Automatyczne wykrywanie fal Elliotta")
st.caption("Heurystyczne rozpoznawanie na podstawie ZigZag i reguł Fibonacciego – tylko do celów edukacyjnych.")

with st.sidebar:
    st.header("Ustawienia danych")
    source = st.radio("Źródło danych", options=["Yahoo Finance", "Plik CSV"], index=0 if HAS_YF else 1)
    ticker = st.text_input("Ticker (np. AAPL, MSFT, BTC-USD, PKN.WA)", value="MSFT")
    period = st.selectbox("Okres (Yahoo Finance)", options=["6mo", "1y", "2y", "5y", "10y", "max"], index=2)
    st.markdown("---")
    st.header("Parametry ZigZag")
    pct_threshold = st.slider("Próg odwrócenia (%)", min_value=1.0, max_value=15.0, value=7.0, step=0.5)
    min_bars_between = st.slider("Min. świece między pivotami", min_value=1, max_value=30, value=3, step=1)
    use_log = st.checkbox("Używaj zmian logarytmicznych", value=True)
    st.markdown("---")
    st.header("Detekcja wzorców")
    choose_pattern = st.multiselect(
        "Szukanie wzorców",
        options=["Impuls 1-2-3-4-5", "Korekta ABC"],
        default=["Impuls 1-2-3-4-5", "Korekta ABC"]
    )
    min_bars_wave = st.slider("Min. świece na falę", min_value=1, max_value=30, value=3, step=1)
    fib_tol = st.slider("Tolerancja Fibo (±)", min_value=0.05, max_value=0.8, value=0.35, step=0.05)
    overlap_tol = st.slider("Tolerancja nakładania 4 na 1", min_value=0.0, max_value=0.1, value=0.02, step=0.005)
    st.markdown("---")
    show_zz = st.checkbox("Pokaż ślad ZigZag", value=True)
    show_table = st.checkbox("Pokaż tabele pivotów/wzorca", value=True)

# ---------------------- Logika ----------------------
data_df: Optional[pd.DataFrame] = None
error_load: Optional[str] = None
debug_logs: List[str] = []

if source == "Yahoo Finance":
    if not HAS_YF:
        error_load = "Moduł yfinance nie jest dostępny w środowisku. Użyj opcji 'Plik CSV'."
    else:
        try:
            data_df, debug_logs = _yf_fetch_debug(ticker, period)
            if data_df is None or data_df.empty:
                error_load = "Brak danych z Yahoo Finance – spróbuj inny ticker/okres."
        except Exception as e:
            error_load = f"Nie udało się pobrać danych: {e}"
else:
    file = st.file_uploader(
        "Wgraj CSV z kolumnami: Date, Open, High, Low, Close, Volume (nagłówki nie są obowiązkowe).",
        type=["csv"]
    )
    if file is not None:
        try:
            data_df = _read_csv(file)
            if data_df is None or data_df.empty:
                error_load = "Plik CSV nie zawiera danych."
        except Exception as e:
            error_load = f"Błąd odczytu CSV: {e}"

if error_load:
    st.error(error_load)
    if debug_logs:
        with st.expander("🔧 Debug Yahoo Finance"):
            for line in debug_logs:
                st.text(line)

if data_df is not None and not data_df.empty:
    st.subheader("Wykres")
    base_title = f"{ticker} – {period}" if source == "Yahoo Finance" else f"Dane z pliku CSV"
    fig = _candles_figure(data_df, title=base_title)

    # ZigZag
    pivots = zigzag_from_close(
        data_df, close_col="Close",
        pct_threshold=pct_threshold,
        min_bars_between_pivots=min_bars_between,
        use_log_change=use_log,
    )
    if show_zz:
        _add_zigzag_trace(fig, pivots, name=f"ZigZag ({pct_threshold:.1f}%)")

    # Detekcja wzorców
    best_impulse = None
    best_abc = None

    if "Impuls 1-2-3-4-5" in choose_pattern:
        best_impulse = detect_impulse(
            pivots,
            direction=None,
            min_bars_per_wave=min_bars_wave,
            fib_tolerance=fib_tol,
            overlap_tolerance=overlap_tol,
        )
        if best_impulse is not None:
            seq = best_impulse.pivots
            name = "Impuls Ⓘ (wzrostowy)" if best_impulse.direction == "up" else "Impuls Ⓘ (spadkowy)"
            _wave_line(fig, seq, name=name + f" – score {best_impulse.score}")
            _label_points(fig, seq[1:], ["1", "2", "3", "4", "5"])

    if "Korekta ABC" in choose_pattern:
        best_abc = detect_abc(pivots, min_bars_per_leg=min_bars_wave, fib_tolerance=fib_tol)
        if best_abc is not None:
            seq = best_abc.pivots
            name = "Korekta ABC (wzrostowa)" if best_abc.direction == "up" else "Korekta ABC (spadkowa)"
            _wave_line(fig, seq, name=name + f" – score {best_abc.score}")
            _label_points(fig, seq[1:], ["A", "B", "C"])

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    # Tabele
    if show_table:
        with st.expander("Punkty ZigZag"):
            st.dataframe(pd.DataFrame(
                [{"Idx": p.i, "Data": p.ts, "Cena": p.price, "Typ": p.kind} for p in pivots]
            ), use_container_width=True)

        if best_impulse is not None:
            with st.expander("Szczegóły dopasowania: Impuls 1-2-3-4-5"):
                st.dataframe(_pattern_table(best_impulse.pivots[1:], ["1", "2", "3", "4", "5"]),
                             use_container_width=True)
                st.json(best_impulse.details)

        if best_abc is not None:
            with st.expander("Szczegóły dopasowania: Korekta ABC"):
                st.dataframe(_pattern_table(best_abc.pivots[1:], ["A", "B", "C"]),
                             use_container_width=True)

    # Eksport wykresu
    try:
        html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button("📥 Pobierz wykres (HTML)", data=html_bytes,
                           file_name="wykres_elliott.html", mime="text/html")
    except Exception:
        pass
else:
    st.info("Wczytaj dane z Yahoo Finance lub wgraj plik CSV, aby rozpocząć.")

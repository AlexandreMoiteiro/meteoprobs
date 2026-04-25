"""
Weather Edge Lab — Streamlit app para comparar previsões meteorológicas

e executar uma análise probabilística de temperatura máxima diária.

Objetivo: apoiar investigação de mercados de previsão meteorológica, sem
automatizar ordens, sem prometer lucro e sem tratar a previsão como certeza.

Como correr:
    pip install streamlit pandas numpy requests plotly python-dateutil
    streamlit run streamlit_weather_polymarket_app.py

Notas:
- Open-Meteo não requer API key para uso não-comercial.
- NWS e MET Norway pedem um User-Agent identificável; edite APP_USER_AGENT.
- O app lê preço/orderbook público do Polymarket, mas não envia ordens.
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dateutil.parser import isoparse


APP_USER_AGENT = "WeatherEdgeLab/0.1 contact: you@example.com"
REQUEST_TIMEOUT = 20


# -----------------------------------------------------------------------------
# Configuração de fontes
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class WeatherSource:
    key: str
    label: str
    url: str
    default_weight: float
    notes: str
    forecast_days_hint: int


OPEN_METEO_SOURCES = [
    WeatherSource(
        key="best_match",
        label="Open-Meteo Best Match",
        url="https://api.open-meteo.com/v1/forecast",
        default_weight=1.05,
        notes="Mistura automática de modelos conforme localização.",
        forecast_days_hint=16,
    ),
    WeatherSource(
        key="ecmwf",
        label="ECMWF IFS/AIFS via Open-Meteo",
        url="https://api.open-meteo.com/v1/ecmwf",
        default_weight=1.25,
        notes="Modelo global europeu; horizonte tipicamente até 15 dias.",
        forecast_days_hint=15,
    ),
    WeatherSource(
        key="gfs",
        label="NOAA GFS/HRRR/NBM via Open-Meteo",
        url="https://api.open-meteo.com/v1/gfs",
        default_weight=1.00,
        notes="GFS global; HRRR/NBM disponíveis sobretudo nos EUA.",
        forecast_days_hint=16,
    ),
    WeatherSource(
        key="dwd_icon",
        label="DWD ICON via Open-Meteo",
        url="https://api.open-meteo.com/v1/dwd-icon",
        default_weight=1.10,
        notes="ICON global/europeu/alemão conforme domínio disponível.",
        forecast_days_hint=10,
    ),
    WeatherSource(
        key="meteofrance",
        label="Météo-France ARPEGE/AROME via Open-Meteo",
        url="https://api.open-meteo.com/v1/meteofrance",
        default_weight=1.10,
        notes="ARPEGE global/europeu; AROME de alta resolução em França e arredores.",
        forecast_days_hint=4,
    ),
    WeatherSource(
        key="gem",
        label="Canadian GEM via Open-Meteo",
        url="https://api.open-meteo.com/v1/gem",
        default_weight=1.00,
        notes="Modelo global/regional do serviço meteorológico canadiano.",
        forecast_days_hint=10,
    ),
    WeatherSource(
        key="ukmo",
        label="UK Met Office UKMO via Open-Meteo",
        url="https://api.open-meteo.com/v1/ukmo",
        default_weight=1.05,
        notes="UKMO global e UKV para Reino Unido/Irlanda.",
        forecast_days_hint=7,
    ),
    WeatherSource(
        key="jma",
        label="JMA GSM/MSM via Open-Meteo",
        url="https://api.open-meteo.com/v1/jma",
        default_weight=0.90,
        notes="JMA global e regional Japão/Coreia.",
        forecast_days_hint=11,
    ),
    WeatherSource(
        key="bom",
        label="BOM ACCESS-G via Open-Meteo",
        url="https://api.open-meteo.com/v1/bom",
        default_weight=0.90,
        notes="Modelo australiano ACCESS-G.",
        forecast_days_hint=10,
    ),
    WeatherSource(
        key="meteoswiss",
        label="MeteoSwiss ICON CH via Open-Meteo",
        url="https://api.open-meteo.com/v1/meteoswiss",
        default_weight=1.15,
        notes="Alta resolução para Suíça/Centro da Europa; curto horizonte.",
        forecast_days_hint=5,
    ),
    WeatherSource(
        key="metno_openmeteo",
        label="MET Norway Nordic via Open-Meteo",
        url="https://api.open-meteo.com/v1/metno",
        default_weight=1.00,
        notes="Modelo nórdico; útil sobretudo na Escandinávia.",
        forecast_days_hint=3,
    ),
]


# -----------------------------------------------------------------------------
# Matemática / probabilidade
# -----------------------------------------------------------------------------

def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    if sigma <= 0 or not np.isfinite(sigma):
        return float("nan")
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def prob_event(mu: float, sigma: float, threshold: float, operator: str) -> float:
    """Probabilidade aproximada do evento de temperatura máxima."""
    if operator in (">", ">="):
        return 1.0 - normal_cdf(threshold, mu, sigma)
    if operator in ("<", "<="):
        return normal_cdf(threshold, mu, sigma)
    raise ValueError(f"Operador inválido: {operator}")


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def weighted_std(values: np.ndarray, weights: np.ndarray, center: Optional[float] = None) -> float:
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() <= 1:
        return 0.0
    values = values[mask]
    weights = weights[mask]
    if center is None:
        center = float(np.average(values, weights=weights))
    variance = np.average((values - center) ** 2, weights=weights)
    return float(math.sqrt(max(variance, 0.0)))


def effective_n(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights > 0)]
    if len(weights) == 0:
        return 0.0
    return float((weights.sum() ** 2) / np.sum(weights ** 2))


def ensemble_analysis(
    df: pd.DataFrame,
    target_date: dt.date,
    sigma_floor: float,
    calibration_rmse: float,
    horizon_sigma_per_day: float,
    correlation_inflation: float,
    threshold: float,
    operator: str,
) -> dict[str, Any]:
    x = df["tmax_c"].to_numpy(dtype=float)
    w = df["weight"].to_numpy(dtype=float)
    mu = weighted_mean(x, w)
    spread = weighted_std(x, w, mu)

    today = dt.date.today()
    horizon_days = max((target_date - today).days, 0)
    horizon_sigma = horizon_sigma_per_day * math.sqrt(max(horizon_days, 0))

    # A dispersão dos modelos não captura todo o erro. O sigma final soma:
    # - discordância entre modelos;
    # - erro mínimo calibrável;
    # - degradação com horizonte;
    # - inflação por correlação, já que muitos modelos partilham observações iniciais.
    base_sigma = max(spread, sigma_floor)
    sigma = math.sqrt(
        (base_sigma * correlation_inflation) ** 2
        + calibration_rmse**2
        + horizon_sigma**2
    )

    p = prob_event(mu, sigma, threshold, operator)
    n_eff = effective_n(w)
    median = float(np.nanmedian(x))

    # Score operacional: não é probabilidade; penaliza poucos modelos, dispersão alta
    # e evento muito próximo do limite.
    distance_z = abs(mu - threshold) / sigma if sigma > 0 else 0.0
    model_count_factor = min(1.0, len(df) / 6.0)
    neff_factor = min(1.0, n_eff / 5.0)
    spread_factor = max(0.15, min(1.0, 1.0 - spread / 6.0))
    distance_factor = max(0.10, min(1.0, distance_z / 2.0))
    consensus_factor = 2.0 * abs(p - 0.5)
    confidence_score = 100.0 * model_count_factor * neff_factor * spread_factor * distance_factor * consensus_factor

    return {
        "mu": mu,
        "median": median,
        "spread": spread,
        "sigma": sigma,
        "p_event": p,
        "n_models": int(len(df)),
        "n_eff": n_eff,
        "horizon_days": horizon_days,
        "horizon_sigma": horizon_sigma,
        "distance_z": distance_z,
        "confidence_score": max(0.0, min(100.0, confidence_score)),
        "ci80": (mu - 1.2816 * sigma, mu + 1.2816 * sigma),
        "ci90": (mu - 1.6449 * sigma, mu + 1.6449 * sigma),
        "ci95": (mu - 1.9600 * sigma, mu + 1.9600 * sigma),
    }


def risk_flags(result: dict[str, Any], df: pd.DataFrame, threshold: float) -> list[str]:
    flags: list[str] = []
    if result["n_models"] < 4:
        flags.append("Poucos modelos disponíveis para a cidade/data escolhida.")
    if result["n_eff"] < 3:
        flags.append("Número efetivo de modelos baixo; as fontes podem estar demasiado correlacionadas.")
    if result["spread"] >= 2.5:
        flags.append("Dispersão entre modelos acima de 2,5 °C.")
    if result["distance_z"] < 1.0:
        flags.append("Previsão central está a menos de 1 sigma do limite; zona de alta ambiguidade.")
    if df["tmax_c"].max() - df["tmax_c"].min() >= 4.0:
        flags.append("Amplitude entre modelo mais quente e mais frio é ≥ 4 °C.")
    if result["horizon_days"] > 7:
        flags.append("Horizonte superior a 7 dias; erro de temperatura máxima cresce bastante.")
    if abs(result["mu"] - threshold) < 0.75:
        flags.append("Limite está a menos de 0,75 °C da previsão central; risco de arredondamento/resolução.")
    return flags


# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------

def http_get_json(url: str, params: Optional[dict[str, Any]] = None, headers: Optional[dict[str, str]] = None) -> Any:
    hdrs = {"User-Agent": APP_USER_AGENT, "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    r = requests.get(url, params=params, headers=hdrs, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_city(query: str) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    data = http_get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": query, "count": 10, "language": "pt", "format": "json"},
    )
    return data.get("results", []) or []


def location_label(item: dict[str, Any]) -> str:
    parts = [item.get("name"), item.get("admin1"), item.get("country")]
    bits = [str(p) for p in parts if p]
    lat = item.get("latitude")
    lon = item.get("longitude")
    if lat is not None and lon is not None:
        bits.append(f"{lat:.4f}, {lon:.4f}")
    return " — ".join(bits)


# -----------------------------------------------------------------------------
# Previsão Open-Meteo e fontes oficiais diretas
# -----------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_open_meteo_source(source_key: str, lat: float, lon: float, target_date: str, timezone: str) -> dict[str, Any]:
    src = next(s for s in OPEN_METEO_SOURCES if s.key == source_key)
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "hourly": "temperature_2m",
        "temperature_unit": "celsius",
        "timezone": timezone or "auto",
        "start_date": target_date,
        "end_date": target_date,
        "forecast_days": min(src.forecast_days_hint, 16),
    }
    try:
        data = http_get_json(src.url, params=params)
    except Exception as exc:
        return {
            "source": src.label,
            "key": src.key,
            "status": "erro",
            "error": str(exc),
            "tmax_c": np.nan,
            "weight": src.default_weight,
            "notes": src.notes,
        }

    tmax = np.nan
    method = ""

    # Caminho preferido: agregação diária pronta.
    try:
        daily = data.get("daily", {})
        times = daily.get("time", [])
        vals = daily.get("temperature_2m_max", [])
        if target_date in times:
            idx = times.index(target_date)
            tmax = float(vals[idx])
            method = "daily.temperature_2m_max"
    except Exception:
        pass

    # Fallback: calcular máximo das horas do dia.
    if not np.isfinite(tmax):
        try:
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            vals = hourly.get("temperature_2m", [])
            hdf = pd.DataFrame({"time": pd.to_datetime(times), "temperature_2m": vals})
            hdf["date"] = hdf["time"].dt.date.astype(str)
            day = hdf[hdf["date"] == target_date]
            if not day.empty:
                tmax = float(pd.to_numeric(day["temperature_2m"], errors="coerce").max())
                method = "max(hourly.temperature_2m)"
        except Exception:
            pass

    if np.isfinite(tmax):
        status = "ok"
        err = ""
    else:
        status = "sem dados"
        err = "Fonte não devolveu temperatura máxima para esta data/local."

    return {
        "source": src.label,
        "key": src.key,
        "status": status,
        "error": err,
        "tmax_c": tmax,
        "weight": src.default_weight,
        "notes": src.notes,
        "method": method,
        "url": src.url,
    }


@st.cache_data(ttl=900, show_spinner=False)
def fetch_nws_direct(lat: float, lon: float, target_date: str) -> dict[str, Any]:
    """NWS direto: só funciona para EUA/territórios suportados."""
    try:
        point = http_get_json(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}")
        grid_url = point["properties"].get("forecastGridData")
        if not grid_url:
            raise RuntimeError("forecastGridData ausente na resposta NWS.")
        grid = http_get_json(grid_url)
        values = grid.get("properties", {}).get("maxTemperature", {}).get("values", [])
        rows = []
        for item in values:
            valid_time = item.get("validTime", "")
            value = item.get("value")
            if value is None:
                continue
            start = valid_time.split("/")[0]
            # A data do grid NWS é suficiente aqui para uma primeira aproximação.
            if start[:10] == target_date:
                rows.append(float(value))
        if not rows:
            raise RuntimeError("NWS não devolveu maxTemperature para a data escolhida.")
        return {
            "source": "NOAA/NWS direto",
            "key": "nws_direct",
            "status": "ok",
            "error": "",
            "tmax_c": float(max(rows)),
            "weight": 1.20,
            "notes": "API oficial NWS; apenas EUA/territórios suportados.",
            "method": "gridData.maxTemperature",
            "url": grid_url,
        }
    except Exception as exc:
        return {
            "source": "NOAA/NWS direto",
            "key": "nws_direct",
            "status": "erro/indisponível",
            "error": str(exc),
            "tmax_c": np.nan,
            "weight": 1.20,
            "notes": "API oficial NWS; apenas EUA/territórios suportados.",
            "method": "",
            "url": "https://api.weather.gov",
        }


@st.cache_data(ttl=900, show_spinner=False)
def fetch_metno_direct(lat: float, lon: float, target_date: str, timezone: str) -> dict[str, Any]:
    """MET Norway locationforecast direto; útil como fonte pública oficial adicional."""
    try:
        data = http_get_json(
            "https://api.met.no/weatherapi/locationforecast/2.0/compact",
            params={"lat": lat, "lon": lon},
        )
        tz = ZoneInfo(timezone) if timezone else ZoneInfo("UTC")
        rows = []
        for item in data.get("properties", {}).get("timeseries", []):
            t_utc = isoparse(item["time"])
            local_date = t_utc.astimezone(tz).date().isoformat()
            if local_date != target_date:
                continue
            details = item.get("data", {}).get("instant", {}).get("details", {})
            temp = details.get("air_temperature")
            if temp is not None:
                rows.append(float(temp))
        if not rows:
            raise RuntimeError("MET Norway não devolveu horas para a data escolhida.")
        return {
            "source": "MET Norway direto",
            "key": "metno_direct",
            "status": "ok",
            "error": "",
            "tmax_c": float(max(rows)),
            "weight": 1.00,
            "notes": "API pública oficial MET Norway locationforecast.",
            "method": "max(timeseries.air_temperature)",
            "url": "https://api.met.no/weatherapi/locationforecast/2.0/compact",
        }
    except Exception as exc:
        return {
            "source": "MET Norway direto",
            "key": "metno_direct",
            "status": "erro/indisponível",
            "error": str(exc),
            "tmax_c": np.nan,
            "weight": 1.00,
            "notes": "API pública oficial MET Norway locationforecast.",
            "method": "",
            "url": "https://api.met.no/weatherapi/locationforecast/2.0/compact",
        }


def collect_forecasts(
    lat: float,
    lon: float,
    target_date: dt.date,
    timezone: str,
    selected_keys: list[str],
    include_nws: bool,
    include_metno_direct: bool,
) -> pd.DataFrame:
    tdate = target_date.isoformat()
    rows: list[dict[str, Any]] = []
    for key in selected_keys:
        rows.append(fetch_open_meteo_source(key, lat, lon, tdate, timezone))
    if include_nws:
        rows.append(fetch_nws_direct(lat, lon, tdate))
    if include_metno_direct:
        rows.append(fetch_metno_direct(lat, lon, tdate, timezone))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["tmax_c"] = pd.to_numeric(df["tmax_c"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
    return df


# -----------------------------------------------------------------------------
# Polymarket: leitura pública apenas
# -----------------------------------------------------------------------------

def parse_maybe_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            try:
                return ast.literal_eval(value)
            except Exception:
                return value
    return value


@st.cache_data(ttl=60, show_spinner=False)
def fetch_polymarket_market_by_slug(slug: str) -> dict[str, Any]:
    slug = slug.strip().strip("/")
    if not slug:
        raise ValueError("slug vazio")
    return http_get_json(f"https://gamma-api.polymarket.com/markets/slug/{slug}")


@st.cache_data(ttl=30, show_spinner=False)
def fetch_polymarket_orderbook(token_id: str) -> dict[str, Any]:
    token_id = token_id.strip()
    if not token_id:
        raise ValueError("token_id vazio")
    return http_get_json("https://clob.polymarket.com/book", params={"token_id": token_id})


def best_bid_ask(book: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    bids = book.get("bids", []) or []
    asks = book.get("asks", []) or []
    best_bid = None
    best_ask = None
    try:
        if bids:
            best_bid = max(float(x["price"]) for x in bids if x.get("price") is not None)
    except Exception:
        best_bid = None
    try:
        if asks:
            best_ask = min(float(x["price"]) for x in asks if x.get("price") is not None)
    except Exception:
        best_ask = None
    return best_bid, best_ask


def extract_threshold_from_question(question: str) -> Optional[float]:
    """Heurística simples para perguntas como 'above 75°F' ou 'over 30°C'."""
    q = question.lower()
    # Celsius explícito.
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*c\b", q)
    if m:
        return float(m.group(1))
    # Fahrenheit explícito; converter para Celsius.
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*f\b", q)
    if m:
        f = float(m.group(1))
        return (f - 32.0) * 5.0 / 9.0
    return None


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Weather Edge Lab", page_icon="🌡️", layout="wide")

st.title("🌡️ Weather Edge Lab")
st.caption("Comparação multi-modelo de temperatura máxima diária + análise probabilística para mercados de previsão.")

with st.expander("Aviso importante", expanded=True):
    st.markdown(
        """
        Esta app **não dá garantia de lucro nem aposta automaticamente**. A temperatura máxima observada pode depender da estação usada,
        arredondamentos, regras de resolução do mercado, microclima urbano, ilhas de calor, falhas de dados e revisões posteriores.
        Use isto como ferramenta de investigação probabilística. O resultado principal é uma estimativa de probabilidade e de incerteza,
        não uma recomendação personalizada de aposta.
        """
    )

# Sidebar — localização e parâmetros
with st.sidebar:
    st.header("1) Local e data")
    city_query = st.text_input("Cidade", value="Lisbon")

    locations = geocode_city(city_query) if city_query else []
    if locations:
        selected_label = st.selectbox("Escolhe a localização", [location_label(x) for x in locations])
        loc = locations[[location_label(x) for x in locations].index(selected_label)]
    else:
        loc = None
        st.info("Escreve uma cidade para geocodificar.")

    default_date = dt.date.today() + dt.timedelta(days=2)
    target_date = st.date_input("Dia da temperatura máxima", value=default_date)

    st.header("2) Fontes")
    default_keys = ["best_match", "ecmwf", "gfs", "dwd_icon", "meteofrance", "gem", "ukmo"]
    labels_by_key = {s.key: s.label for s in OPEN_METEO_SOURCES}
    selected_labels = st.multiselect(
        "Modelos via Open-Meteo",
        options=[s.label for s in OPEN_METEO_SOURCES],
        default=[labels_by_key[k] for k in default_keys if k in labels_by_key],
    )
    selected_keys = [s.key for s in OPEN_METEO_SOURCES if s.label in selected_labels]

    include_nws = st.checkbox("Adicionar NOAA/NWS direto quando for EUA", value=True)
    include_metno_direct = st.checkbox("Adicionar MET Norway direto", value=False)

    st.header("3) Mercado / limite")
    operator = st.selectbox("Evento", options=[">=", ">", "<=", "<"], index=0)
    threshold_c = st.number_input("Limite de temperatura em °C", value=30.0, step=0.5)

    st.header("4) Incerteza")
    sigma_floor = st.slider("Piso de sigma por incerteza local (°C)", 0.2, 4.0, 0.8, 0.1)
    calibration_rmse = st.slider("Erro mínimo de calibração/RMSE assumido (°C)", 0.5, 5.0, 1.4, 0.1)
    horizon_sigma_per_day = st.slider("Penalização por horizonte, °C × √dias", 0.0, 1.0, 0.15, 0.05)
    correlation_inflation = st.slider("Inflação por correlação entre modelos", 1.0, 2.5, 1.25, 0.05)

    st.header("5) Polymarket opcional")
    market_slug = st.text_input("Market slug Polymarket opcional", placeholder="ex: will-nyc-high-temperature-be-above-...")
    token_id_manual = st.text_input("Ou token_id/outcome asset id opcional", placeholder="asset/token id do outcome")
    manual_price = st.number_input("Preço manual do outcome, se não houver orderbook", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    run = st.button("Analisar", type="primary", use_container_width=True)


if not loc:
    st.stop()

lat = float(loc["latitude"])
lon = float(loc["longitude"])
timezone = loc.get("timezone") or "auto"

st.subheader("Localização")
cols = st.columns(4)
cols[0].metric("Cidade", loc.get("name", "—"))
cols[1].metric("País", loc.get("country", "—"))
cols[2].metric("Latitude", f"{lat:.4f}")
cols[3].metric("Longitude", f"{lon:.4f}")
st.caption(f"Timezone: {timezone}")

if not run:
    st.info("Configura a cidade, data e limite na barra lateral e clica em **Analisar**.")
    st.stop()

if not selected_keys and not include_nws and not include_metno_direct:
    st.error("Seleciona pelo menos uma fonte meteorológica.")
    st.stop()

with st.spinner("A recolher previsões dos modelos..."):
    raw_df = collect_forecasts(lat, lon, target_date, timezone, selected_keys, include_nws, include_metno_direct)

if raw_df.empty:
    st.error("Não consegui recolher dados de previsão.")
    st.stop()

ok_df = raw_df[pd.to_numeric(raw_df["tmax_c"], errors="coerce").notna()].copy()

st.subheader("Previsões recolhidas")
show_cols = ["source", "tmax_c", "weight", "status", "method", "notes", "error"]
st.dataframe(
    raw_df[show_cols].rename(
        columns={
            "source": "Fonte",
            "tmax_c": "Tmax °C",
            "weight": "Peso",
            "status": "Estado",
            "method": "Método",
            "notes": "Notas",
            "error": "Erro",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

if len(ok_df) < 2:
    st.error("Há menos de 2 fontes com dados válidos. A análise estatística seria demasiado fraca.")
    st.stop()

fig = px.bar(
    ok_df.sort_values("tmax_c"),
    x="tmax_c",
    y="source",
    orientation="h",
    text=ok_df.sort_values("tmax_c")["tmax_c"].map(lambda x: f"{x:.1f} °C"),
    labels={"tmax_c": "Temperatura máxima prevista (°C)", "source": "Fonte"},
    title="Temperatura máxima por modelo/fonte",
)
fig.update_layout(height=max(350, 35 * len(ok_df)))
st.plotly_chart(fig, use_container_width=True)

result = ensemble_analysis(
    ok_df,
    target_date,
    sigma_floor=sigma_floor,
    calibration_rmse=calibration_rmse,
    horizon_sigma_per_day=horizon_sigma_per_day,
    correlation_inflation=correlation_inflation,
    threshold=threshold_c,
    operator=operator,
)

p = result["p_event"]

st.subheader("Análise matemática")
metrics = st.columns(5)
metrics[0].metric("Tmax estimada", f"{result['mu']:.2f} °C")
metrics[1].metric("Sigma final", f"{result['sigma']:.2f} °C")
metrics[2].metric("Prob. evento", f"{100*p:.1f}%")
metrics[3].metric("Modelos válidos", f"{result['n_models']}")
metrics[4].metric("Confiança operacional", f"{result['confidence_score']:.0f}/100")

c1, c2, c3 = st.columns(3)
c1.info(f"Intervalo 80%: **{result['ci80'][0]:.1f} a {result['ci80'][1]:.1f} °C**")
c2.info(f"Intervalo 90%: **{result['ci90'][0]:.1f} a {result['ci90'][1]:.1f} °C**")
c3.info(f"Intervalo 95%: **{result['ci95'][0]:.1f} a {result['ci95'][1]:.1f} °C**")

st.markdown(
    f"""
    **Evento testado:** temperatura máxima em **{target_date.isoformat()}** {operator} **{threshold_c:.1f} °C**.  
    **Interpretação:** assumindo distribuição aproximadamente normal em torno do ensemble ponderado, a probabilidade estimada é **{100*p:.1f}%**.
    """
)

# Simulação Monte Carlo para visualização
rng = np.random.default_rng(42)
samples = rng.normal(result["mu"], result["sigma"], size=50_000)
sim_df = pd.DataFrame({"Tmax simulada °C": samples})
fig_hist = px.histogram(sim_df, x="Tmax simulada °C", nbins=60, title="Distribuição simulada da temperatura máxima")
fig_hist.add_vline(x=threshold_c, line_dash="dash", annotation_text="Limite")
st.plotly_chart(fig_hist, use_container_width=True)

flags = risk_flags(result, ok_df, threshold_c)
if flags:
    st.warning("Sinais de risco / razões para não entrar automaticamente:")
    for f in flags:
        st.markdown(f"- {f}")
else:
    st.success("Não apareceram flags fortes, mas isto continua a ser probabilístico e dependente das regras de resolução.")

with st.expander("Como a conta é feita"):
    st.latex(r"\mu = \frac{\sum_i w_i x_i}{\sum_i w_i}")
    st.latex(r"\sigma = \sqrt{(\max(s_{modelos}, \sigma_{piso}) \cdot k_{corr})^2 + RMSE_{calib}^2 + \sigma_{horizonte}^2}")
    st.latex(r"P(T_{max} \ge L) = 1 - \Phi\left(\frac{L - \mu}{\sigma}\right)")
    st.write(
        {
            "media_ponderada_mu": result["mu"],
            "dispersao_modelos": result["spread"],
            "sigma_final": result["sigma"],
            "n_efetivo": result["n_eff"],
            "horizonte_dias": result["horizon_days"],
            "sigma_horizonte": result["horizon_sigma"],
            "distancia_ao_limite_em_sigmas": result["distance_z"],
        }
    )

# -----------------------------------------------------------------------------
# Polymarket opcional
# -----------------------------------------------------------------------------

st.subheader("Polymarket — comparação com preço de mercado")
st.caption("Leitura pública de mercado/orderbook. A app não envia ordens.")

market_info = None
selected_token = token_id_manual.strip() or None
market_price = manual_price if manual_price > 0 else None

if market_slug.strip():
    try:
        market_info = fetch_polymarket_market_by_slug(market_slug.strip())
        question = market_info.get("question", "")
        st.markdown(f"**Mercado:** {question}")
        st.write(f"Slug: `{market_info.get('slug', market_slug.strip())}`")
        st.write(f"Ativo: `{market_info.get('active')}`, fechado: `{market_info.get('closed')}`")

        auto_threshold = extract_threshold_from_question(question)
        if auto_threshold is not None:
            st.info(f"Heurística encontrou possível limite na pergunta: **{auto_threshold:.2f} °C**. Confirma manualmente na barra lateral.")

        outcomes = parse_maybe_json(market_info.get("outcomes")) or []
        prices = parse_maybe_json(market_info.get("outcomePrices")) or []
        token_ids = parse_maybe_json(market_info.get("clobTokenIds")) or parse_maybe_json(market_info.get("clobTokenIDs")) or []

        outcome_rows = []
        for i, outcome in enumerate(outcomes):
            outcome_rows.append(
                {
                    "outcome": outcome,
                    "price": prices[i] if i < len(prices) else None,
                    "token_id": token_ids[i] if i < len(token_ids) else None,
                }
            )
        if outcome_rows:
            st.dataframe(pd.DataFrame(outcome_rows), hide_index=True, use_container_width=True)

        # Se houver token manual, respeita. Caso contrário, tenta escolher outcome Yes.
        if selected_token is None and outcome_rows:
            yes_rows = [r for r in outcome_rows if str(r.get("outcome", "")).lower() in ("yes", "sim")]
            if yes_rows and yes_rows[0].get("token_id"):
                selected_token = str(yes_rows[0]["token_id"])
                st.caption("Token selecionado automaticamente: outcome Yes/Sim.")
            elif outcome_rows[0].get("token_id"):
                selected_token = str(outcome_rows[0]["token_id"])
                st.caption("Token selecionado automaticamente: primeiro outcome.")
    except Exception as exc:
        st.error(f"Não consegui ler o mercado Polymarket pelo slug: {exc}")

if selected_token:
    try:
        book = fetch_polymarket_orderbook(selected_token)
        best_bid, best_ask = best_bid_ask(book)
        last_trade = book.get("last_trade_price")
        cols = st.columns(3)
        cols[0].metric("Best bid", "—" if best_bid is None else f"{best_bid:.3f}")
        cols[1].metric("Best ask", "—" if best_ask is None else f"{best_ask:.3f}")
        cols[2].metric("Último trade", "—" if last_trade is None else str(last_trade))
        if best_ask is not None:
            market_price = best_ask
    except Exception as exc:
        st.error(f"Não consegui ler orderbook Polymarket: {exc}")

if market_price is not None and 0 < market_price < 1:
    edge = p - market_price
    fair = p
    st.markdown("### Leitura estatística vs preço")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prob. modelo", f"{100*p:.1f}%")
    c2.metric("Preço mercado/outcome", f"{100*market_price:.1f}%")
    c3.metric("Diferença p − preço", f"{100*edge:.1f} pp")

    st.write(f"Preço estatisticamente neutro estimado: **{fair:.3f}**. Acima disso, o modelo não vê margem positiva.")

    if edge > 0.10 and result["confidence_score"] >= 55 and not flags:
        st.success("Modelo mostra margem estatística relevante, mas valida regras de resolução, liquidez, spread e tamanho antes de qualquer decisão.")
    elif edge > 0.03:
        st.info("Há pequena margem estatística, mas as flags/uncerteza podem anulá-la.")
    else:
        st.warning("Pelo modelo atual, não há margem estatística clara a este preço.")
else:
    st.info("Para comparar com mercado, insere um preço manual, um token_id ou um slug de mercado.")

# Exportação
st.subheader("Exportar")
export = ok_df.copy()
export["target_date"] = target_date.isoformat()
export["threshold_c"] = threshold_c
export["operator"] = operator
export["mu_c"] = result["mu"]
export["sigma_c"] = result["sigma"]
export["prob_event"] = p
st.download_button(
    "Download CSV da análise",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name=f"weather_edge_{loc.get('name','city')}_{target_date.isoformat()}.csv",
    mime="text/csv",
)

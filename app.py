import math
from datetime import date, timedelta
from statistics import NormalDist
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# WEATHER EDGE — STREAMLIT APP
# Sem sidebar. Sem API key. Open-Meteo apenas.
# ============================================================

st.set_page_config(
    page_title="Weather Edge",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
USER_AGENT = "weather-edge-polymarket/2.0"

# Modelos fortes/precisos. A app tenta cada um e ignora automaticamente
# os que não estiverem disponíveis para a localização/data.
PRECISE_MODELS = [
    {
        "name": "Open-Meteo Best Match",
        "code": None,
        "provider": "Open-Meteo multi-model",
        "base_weight": 1.20,
        "note": "Combina automaticamente os melhores modelos para o local.",
    },
    {
        "name": "ECMWF IFS 0.25°",
        "code": "ecmwf_ifs025",
        "provider": "ECMWF",
        "base_weight": 1.35,
        "note": "Modelo global europeu, normalmente muito forte em médio prazo.",
    },
    {
        "name": "ECMWF AIFS 0.25°",
        "code": "ecmwf_aifs025_single",
        "provider": "ECMWF AI",
        "base_weight": 1.20,
        "note": "Modelo AI do ECMWF; útil como visão independente.",
    },
    {
        "name": "UK Met Office Seamless",
        "code": "ukmo_seamless",
        "provider": "UKMO",
        "base_weight": 1.10,
        "note": "Modelo global/regional do Met Office.",
    },
    {
        "name": "DWD ICON Seamless",
        "code": "icon_seamless",
        "provider": "DWD",
        "base_weight": 1.08,
        "note": "Forte na Europa; global/regional seamless.",
    },
    {
        "name": "Météo-France Seamless",
        "code": "meteofrance_seamless",
        "provider": "Météo-France",
        "base_weight": 1.08,
        "note": "Muito útil para Europa ocidental e curto prazo.",
    },
    {
        "name": "NOAA GFS Seamless",
        "code": "gfs_seamless",
        "provider": "NOAA/NCEP",
        "base_weight": 1.00,
        "note": "Global, bom para comparação e divergência.",
    },
]

MODEL_LABELS = [m["name"] for m in PRECISE_MODELS]

BET_EXACT = "Temperatura máxima exata"
BET_OVER = "Temperatura máxima maior do que"

UNIT_OPTIONS = {
    "Celsius °C": {"api": "celsius", "symbol": "°C", "factor_from_c": 1.0},
    "Fahrenheit °F": {"api": "fahrenheit", "symbol": "°F", "factor_from_c": 1.8},
}

DECISION_GREEN = "🟢 VERDE"
DECISION_YELLOW = "🟡 AMARELO"
DECISION_RED = "🔴 VERMELHO"


# ============================================================
# CSS / UI
# ============================================================

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {display: none;}
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }
    .hero {
        padding: 1.35rem 1.45rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(127,127,127,0.06));
        border: 1px solid rgba(127,127,127,0.18);
        margin-bottom: 1.1rem;
    }
    .hero h1 {
        margin-bottom: 0.25rem;
        font-size: 2.15rem;
        line-height: 1.15;
    }
    .subtle {
        color: #888;
        font-size: 0.96rem;
    }
    .card {
        border: 1px solid rgba(127,127,127,0.18);
        border-radius: 1rem;
        padding: 1rem;
        background: rgba(127,127,127,0.045);
    }
    .decision-green {
        border-radius: 1.2rem;
        padding: 1.2rem;
        border: 1px solid rgba(0, 180, 90, 0.35);
        background: rgba(0, 180, 90, 0.10);
    }
    .decision-yellow {
        border-radius: 1.2rem;
        padding: 1.2rem;
        border: 1px solid rgba(220, 170, 0, 0.35);
        background: rgba(220, 170, 0, 0.10);
    }
    .decision-red {
        border-radius: 1.2rem;
        padding: 1.2rem;
        border: 1px solid rgba(220, 60, 60, 0.35);
        background: rgba(220, 60, 60, 0.10);
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(127,127,127,0.16);
        border-radius: 1rem;
        padding: 0.85rem;
        background: rgba(127,127,127,0.035);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🌡️ Weather Edge</h1>
        <div class="subtle">
            Análise probabilística da temperatura máxima para mercados tipo Polymarket:
            temperatura exata ou maior do que uma linha. Interface limpa, sem sidebar.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API HELPERS
# ============================================================

def http_get_json(url: str, params: Optional[dict] = None, timeout: int = 25) -> dict:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:350]}")
    return r.json()


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def geocode_city(city: str) -> List[dict]:
    data = http_get_json(
        OPEN_METEO_GEOCODING,
        params={"name": city, "count": 8, "language": "pt", "format": "json"},
    )
    return data.get("results", []) or []


def place_label(place: dict) -> str:
    bits = [place.get("name"), place.get("admin1"), place.get("country")]
    label = ", ".join([str(x) for x in bits if x])
    return f"{label} · {place.get('latitude'):.4f}, {place.get('longitude'):.4f}"


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


@st.cache_data(ttl=18 * 60, show_spinner=False)
def fetch_model_forecast(
    lat: float,
    lon: float,
    target_day: str,
    unit_api: str,
    model_name: str,
    model_code: Optional[str],
    provider: str,
    base_weight: float,
    note: str,
) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": target_day,
        "end_date": target_day,
        "daily": "temperature_2m_max,temperature_2m_min,apparent_temperature_max,precipitation_sum,wind_speed_10m_max",
        "timezone": "auto",
        "temperature_unit": unit_api,
        "cell_selection": "land",
    }
    if model_code:
        params["models"] = model_code

    data = http_get_json(OPEN_METEO_FORECAST, params=params)
    daily = data.get("daily", {})
    times = daily.get("time", [])

    if target_day not in times:
        raise RuntimeError("A data não veio na resposta do modelo.")

    idx = times.index(target_day)

    tmax = safe_float((daily.get("temperature_2m_max") or [None])[idx])
    tmin = safe_float((daily.get("temperature_2m_min") or [None])[idx])
    apparent = safe_float((daily.get("apparent_temperature_max") or [None])[idx])
    rain = safe_float((daily.get("precipitation_sum") or [None])[idx])
    wind = safe_float((daily.get("wind_speed_10m_max") or [None])[idx])

    if tmax is None:
        raise RuntimeError("Sem temperature_2m_max.")

    return {
        "modelo": model_name,
        "codigo": model_code or "best_match",
        "fornecedor": provider,
        "tmax": tmax,
        "tmin": tmin,
        "sensacao_max": apparent,
        "precipitacao": rain,
        "vento_max": wind,
        "peso_base": base_weight,
        "nota": note,
    }


def circular_doy_distance(a: int, b: int, days: int = 366) -> int:
    diff = abs(a - b)
    return min(diff, days - diff)


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def fetch_climatology(
    lat: float,
    lon: float,
    target_day: str,
    unit_api: str,
    years_back: int,
    window_days: int,
) -> pd.DataFrame:
    target = date.fromisoformat(target_day)
    end_year = min(date.today().year - 1, target.year - 1)
    start_year = max(1940, end_year - years_back + 1)

    if end_year < start_year:
        return pd.DataFrame(columns=["date", "tmax"])

    data = http_get_json(
        OPEN_METEO_ARCHIVE,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
            "daily": "temperature_2m_max",
            "temperature_unit": unit_api,
            "timezone": "auto",
        },
        timeout=35,
    )

    daily = data.get("daily", {})
    dates = daily.get("time", []) or []
    temps = daily.get("temperature_2m_max", []) or []
    target_doy = target.timetuple().tm_yday

    rows = []
    for d_str, temp in zip(dates, temps):
        value = safe_float(temp)
        if value is None:
            continue
        d = date.fromisoformat(d_str)
        if circular_doy_distance(d.timetuple().tm_yday, target_doy) <= window_days:
            rows.append({"date": d, "tmax": value})

    return pd.DataFrame(rows)


# ============================================================
# MATHEMATICAL MODEL
# ============================================================

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    weights = np.where(weights <= 0, 1.0, weights)
    return float(np.average(values, weights=weights))


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    mu = weighted_mean(values, weights)
    var = float(np.average((values - mu) ** 2, weights=weights))
    return math.sqrt(max(0.0, var))


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    return NormalDist(mu=mu, sigma=max(sigma, 1e-6)).cdf(x)


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    return NormalDist(mu=mu, sigma=max(sigma, 1e-6)).pdf(x)


def horizon_base_sigma(horizon_days: int, unit_factor: float, conservative: float) -> float:
    # Piso deliberadamente conservador para Tmax diária.
    # Em °C: ~0.85 no dia 0, ~1.5 dia 4, ~2.3 dia 8, ~3.2 dia 14.
    h = max(0, horizon_days)
    sigma_c = 0.85 + 0.16 * h + 0.025 * (h ** 1.25)
    return sigma_c * unit_factor * conservative


def make_mixture_distribution(
    forecasts: pd.DataFrame,
    clim: pd.DataFrame,
    target_day: date,
    unit_factor: float,
    conservative: float,
) -> Dict[str, Any]:
    horizon = max(0, (target_day - date.today()).days)

    df = forecasts.copy()
    x = df["tmax"].to_numpy(dtype=float)
    base_weights = df["peso_base"].to_numpy(dtype=float)

    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median))) if len(x) else 0.0
    robust_scale = max(1.0 * unit_factor, 1.4826 * mad)

    # Penaliza outliers sem os remover.
    outlier_distance = np.abs(x - median) / robust_scale
    outlier_penalty = np.where(outlier_distance > 2.5, 0.45, 1.0)
    weights = base_weights * outlier_penalty

    model_spread = weighted_std(x, weights)
    model_range = float(np.max(x) - np.min(x)) if len(x) else math.nan
    model_mean = weighted_mean(x, weights)

    base_sigma = horizon_base_sigma(horizon, unit_factor, conservative)

    # Cada modelo vira um componente normal.
    # Modelos penalizados recebem sigma maior.
    component_sigmas = []
    for penalty in outlier_penalty:
        s = base_sigma * (1.25 if penalty < 1 else 1.0)
        component_sigmas.append(max(0.45 * unit_factor, s))

    components = []
    for i, row in df.iterrows():
        components.append(
            {
                "label": row["modelo"],
                "mu": float(row["tmax"]),
                "sigma": float(component_sigmas[i]),
                "weight": float(weights[i]),
                "type": "modelo",
            }
        )

    clim_mean = None
    clim_std = None
    prior_weight = 0.0

    if not clim.empty and len(clim) >= 20:
        clim_mean = float(clim["tmax"].mean())
        clim_std = float(clim["tmax"].std(ddof=1))
        if math.isnan(clim_std) or clim_std <= 0:
            clim_std = 3.0 * unit_factor

        # Quanto mais longe o horizonte, maior a força do prior histórico.
        prior_weight = min(0.34, 0.05 + horizon / 55.0)
        components.append(
            {
                "label": "Climatologia sazonal",
                "mu": clim_mean,
                "sigma": max(clim_std * 0.70, base_sigma * 1.15),
                "weight": prior_weight * float(np.sum(weights)),
                "type": "prior",
            }
        )

    raw_w = np.array([c["weight"] for c in components], dtype=float)
    raw_w = np.where(raw_w <= 0, 1.0, raw_w)
    norm_w = raw_w / raw_w.sum()

    for c, w in zip(components, norm_w):
        c["weight_norm"] = float(w)

    mus = np.array([c["mu"] for c in components], dtype=float)
    sigmas = np.array([c["sigma"] for c in components], dtype=float)

    mix_mean = float(np.sum(norm_w * mus))
    mix_second = float(np.sum(norm_w * (sigmas**2 + mus**2)))
    mix_var = max(0.0, mix_second - mix_mean**2)
    mix_sigma = math.sqrt(mix_var)

    # Score técnico: não é "probabilidade de estar certo"; é qualidade do sinal.
    dispersion_penalty = min(28.0, 12.0 * (model_spread / max(unit_factor, 1e-6)))
    range_penalty = min(22.0, 4.0 * (model_range / max(unit_factor, 1e-6)))
    horizon_penalty = min(28.0, 2.0 * horizon)
    n_penalty = 18.0 if len(df) < 3 else 8.0 if len(df) < 5 else 0.0
    sigma_penalty = min(22.0, 5.0 * (mix_sigma / max(unit_factor, 1e-6)))

    confidence = 100.0 - dispersion_penalty - range_penalty - horizon_penalty - n_penalty - sigma_penalty
    confidence = float(max(0.0, min(100.0, confidence)))

    if model_spread <= 0.75 * unit_factor and model_range <= 2.0 * unit_factor and len(df) >= 5:
        consensus = "Forte"
    elif model_spread <= 1.35 * unit_factor and model_range <= 3.5 * unit_factor and len(df) >= 4:
        consensus = "Médio"
    else:
        consensus = "Fraco"

    return {
        "components": components,
        "mean": mix_mean,
        "sigma": mix_sigma,
        "model_mean": model_mean,
        "model_spread": model_spread,
        "model_range": model_range,
        "horizon": horizon,
        "base_sigma": base_sigma,
        "clim_mean": clim_mean,
        "clim_std": clim_std,
        "prior_weight": prior_weight,
        "confidence": confidence,
        "consensus": consensus,
        "outlier_penalty": outlier_penalty,
    }


def mixture_probability_interval(components: List[dict], lower: float, upper: float) -> float:
    total = 0.0
    for c in components:
        w = c["weight_norm"]
        total += w * (normal_cdf(upper, c["mu"], c["sigma"]) - normal_cdf(lower, c["mu"], c["sigma"]))
    return float(max(0.0, min(1.0, total)))


def mixture_probability_over(components: List[dict], threshold: float) -> float:
    total = 0.0
    for c in components:
        w = c["weight_norm"]
        total += w * (1.0 - normal_cdf(threshold, c["mu"], c["sigma"]))
    return float(max(0.0, min(1.0, total)))


def mixture_quantile(components: List[dict], q: float, lo: float, hi: float) -> float:
    for _ in range(80):
        mid = (lo + hi) / 2
        cdf = 0.0
        for c in components:
            cdf += c["weight_norm"] * normal_cdf(mid, c["mu"], c["sigma"])
        if cdf < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def expected_value(prob: float, price_yes: float) -> Dict[str, float]:
    price_yes = max(0.001, min(0.999, price_yes))
    price_no = 1.0 - price_yes

    edge_yes = prob - price_yes
    edge_no = (1.0 - prob) - price_no

    roi_yes = edge_yes / price_yes
    roi_no = edge_no / max(price_no, 0.001)

    return {
        "edge_yes": edge_yes,
        "edge_no": edge_no,
        "roi_yes": roi_yes,
        "roi_no": roi_no,
        "best_side": "YES" if edge_yes >= edge_no else "NO",
        "best_edge": max(edge_yes, edge_no),
        "best_roi": roi_yes if edge_yes >= edge_no else roi_no,
    }


def decide_signal(
    prob: float,
    ev: Dict[str, float],
    confidence: float,
    consensus: str,
    sigma: float,
    unit_factor: float,
    bet_type: str,
) -> Dict[str, str]:
    best_edge = ev["best_edge"]
    best_side = ev["best_side"]

    risk_flags = []
    if confidence < 55:
        risk_flags.append("confiança técnica baixa")
    if consensus == "Fraco":
        risk_flags.append("modelos discordam")
    if sigma > 3.0 * unit_factor:
        risk_flags.append("distribuição muito larga")
    if bet_type == BET_EXACT and prob < 0.18 and best_side == "YES":
        risk_flags.append("temperatura exata é naturalmente difícil")

    if best_edge >= 0.08 and confidence >= 68 and consensus != "Fraco" and sigma <= 3.2 * unit_factor:
        color = DECISION_GREEN
        css = "decision-green"
        action = f"Sinal favorável para {best_side}"
        explanation = "Há margem estatística clara contra o preço do mercado e a qualidade do sinal é aceitável."
    elif best_edge >= 0.035 and confidence >= 50 and len(risk_flags) <= 1:
        color = DECISION_YELLOW
        css = "decision-yellow"
        action = f"Sinal moderado para {best_side}, melhor esperar preço melhor ou reduzir tamanho"
        explanation = "Existe edge, mas a incerteza ainda é relevante."
    else:
        color = DECISION_RED
        css = "decision-red"
        action = "Não apostar / sem edge suficiente"
        explanation = "O preço não compensa a incerteza estimada ou os modelos não estão alinhados."

    return {
        "color": color,
        "css": css,
        "action": action,
        "explanation": explanation,
        "risk_flags": ", ".join(risk_flags) if risk_flags else "sem alertas fortes",
    }


def probability_text(p: float) -> str:
    if p >= 0.80:
        return "muito alta"
    if p >= 0.65:
        return "alta"
    if p >= 0.52:
        return "ligeiramente favorável"
    if p >= 0.40:
        return "incerta"
    if p >= 0.20:
        return "baixa"
    return "muito baixa"


def fmt_optional(value: Optional[float], symbol: str) -> str:
    if value is None:
        return "não disponível"
    return f"{value:.1f}{symbol}"


# ============================================================
# CHARTS
# ============================================================

def chart_models(df: pd.DataFrame, target_value: float, final_mean: float, symbol: str, bet_type: str) -> go.Figure:
    data = df.sort_values("tmax")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["tmax"],
            y=data["modelo"],
            mode="markers",
            marker=dict(size=14),
            text=data["fornecedor"],
            hovertemplate="<b>%{y}</b><br>%{text}<br>Tmax=%{x:.1f}" + symbol + "<extra></extra>",
            name="Modelos",
        )
    )
    fig.add_vline(x=final_mean, line_dash="dash", annotation_text="estimativa final")
    if bet_type == BET_OVER:
        fig.add_vline(x=target_value, line_dash="dot", annotation_text="linha do mercado")
    else:
        fig.add_vrect(
            x0=target_value - 0.5,
            x1=target_value + 0.5,
            opacity=0.12,
            line_width=0,
            annotation_text="zona exata",
        )
        fig.add_vline(x=target_value, line_dash="dot", annotation_text="alvo exato")
    fig.update_layout(
        title="Previsões dos modelos premium",
        xaxis_title=f"Temperatura máxima ({symbol})",
        yaxis_title="",
        height=max(360, 55 * len(data)),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def chart_distribution(components: List[dict], target_value: float, symbol: str, bet_type: str) -> go.Figure:
    mus = np.array([c["mu"] for c in components], dtype=float)
    sigmas = np.array([c["sigma"] for c in components], dtype=float)

    lo = float(np.min(mus - 4 * sigmas))
    hi = float(np.max(mus + 4 * sigmas))
    xs = np.linspace(lo, hi, 600)

    ys = []
    for x in xs:
        y = 0.0
        for c in components:
            y += c["weight_norm"] * normal_pdf(float(x), c["mu"], c["sigma"])
        ys.append(y)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Probabilidade estimada"))

    if bet_type == BET_EXACT:
        fig.add_vrect(
            x0=target_value - 0.5,
            x1=target_value + 0.5,
            opacity=0.18,
            line_width=0,
            annotation_text=f"evento exato: {target_value:g}{symbol}",
        )
    else:
        fig.add_vrect(
            x0=target_value,
            x1=hi,
            opacity=0.13,
            line_width=0,
            annotation_text=f"evento: > {target_value:g}{symbol}",
        )

    fig.add_vline(x=target_value, line_dash="dot")
    fig.update_layout(
        title="Distribuição probabilística final",
        xaxis_title=f"Temperatura máxima ({symbol})",
        yaxis_title="Densidade",
        height=380,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def chart_climatology(clim: pd.DataFrame, final_mean: float, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=clim["tmax"],
            nbinsx=32,
            name="Histórico",
            opacity=0.75,
        )
    )
    fig.add_vline(x=final_mean, line_dash="dash", annotation_text="estimativa atual")
    fig.update_layout(
        title="Histórico da mesma época do ano",
        xaxis_title=f"Tmax histórica ({symbol})",
        yaxis_title="Frequência",
        height=340,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# ============================================================
# INPUTS — SEM SIDEBAR
# ============================================================

with st.container():
    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        city = st.text_input("Cidade", value="Lisboa", placeholder="Ex.: Lisboa, New York, Madrid")
    with c2:
        unit_label = st.selectbox("Unidade do mercado", list(UNIT_OPTIONS.keys()), index=0)
    with c3:
        max_date = date.today() + timedelta(days=15)
        target_day = st.date_input(
            "Data da aposta",
            value=date.today() + timedelta(days=3),
            min_value=date.today(),
            max_value=max_date,
        )

unit = UNIT_OPTIONS[unit_label]
unit_api = unit["api"]
symbol = unit["symbol"]
unit_factor = unit["factor_from_c"]

matches: List[dict] = []
selected_place = None

if city.strip():
    try:
        matches = geocode_city(city.strip())
    except Exception as e:
        st.error(f"Erro ao procurar cidade: {e}")

if matches:
    selected_place = st.selectbox("Local encontrado", matches, format_func=place_label)
else:
    st.warning("Escreve uma cidade válida para procurar a localização.")

st.markdown("### Tipo de mercado")
b1, b2, b3 = st.columns([1.15, 1, 1])

with b1:
    bet_type = st.radio(
        "Aposta",
        [BET_EXACT, BET_OVER],
        horizontal=False,
        help="Exata = probabilidade da Tmax arredondar para esse valor. Maior do que = probabilidade de superar a linha.",
    )

with b2:
    target_value = st.number_input(
        f"Valor alvo ({symbol})",
        value=25.0 if symbol == "°C" else 77.0,
        step=1.0 if bet_type == BET_EXACT else 0.5,
        format="%.1f",
        help="Para temperatura exata, assume arredondamento ao inteiro mais próximo: alvo ± 0.5.",
    )

with b3:
    market_price = st.number_input(
        "Preço YES no mercado",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
        help="Ex.: preço 0.37 significa que o mercado implica ~37% antes de taxas/spread.",
    )

with st.expander("Configuração avançada", expanded=False):
    a1, a2, a3 = st.columns(3)
    with a1:
        years_back = st.slider("Anos de histórico", 8, 30, 20, 1)
    with a2:
        window_days = st.slider("Janela sazonal ± dias", 5, 25, 12, 1)
    with a3:
        conservative = st.slider("Conservadorismo da incerteza", 1.00, 1.80, 1.25, 0.05)

    chosen_labels = st.multiselect(
        "Modelos usados",
        MODEL_LABELS,
        default=MODEL_LABELS,
        help="Estão pré-filtrados para modelos fortes. Se muitos falharem para uma região/data, a app continua com os restantes.",
    )

run = st.button("Analisar mercado", type="primary", use_container_width=True)

if not run:
    st.info("Preenche cidade, data, tipo de mercado e preço YES. Depois clica em **Analisar mercado**.")
    st.stop()

if selected_place is None:
    st.error("Não foi possível escolher uma localização.")
    st.stop()

selected_models = [m for m in PRECISE_MODELS if m["name"] in chosen_labels]
if not selected_models:
    st.error("Seleciona pelo menos um modelo.")
    st.stop()

lat = float(selected_place["latitude"])
lon = float(selected_place["longitude"])
target_day_str = target_day.isoformat()

st.markdown(
    f"""
    <div class="card">
    <b>Local:</b> {place_label(selected_place)}<br>
    <b>Evento:</b> {bet_type} · alvo <b>{target_value:g}{symbol}</b> · preço YES <b>{market_price:.2f}</b>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FETCH DATA
# ============================================================

records = []
errors = []

progress = st.progress(0, text="A consultar modelos meteorológicos premium...")
for i, model in enumerate(selected_models, start=1):
    try:
        rec = fetch_model_forecast(
            lat=lat,
            lon=lon,
            target_day=target_day_str,
            unit_api=unit_api,
            model_name=model["name"],
            model_code=model["code"],
            provider=model["provider"],
            base_weight=model["base_weight"],
            note=model["note"],
        )
        records.append(rec)
    except Exception as e:
        errors.append(f"{model['name']}: {e}")
    progress.progress(i / len(selected_models), text="A consultar modelos meteorológicos premium...")

progress.empty()

if len(records) < 2:
    st.error("Poucos modelos responderam. Tenta outra data ou cidade. A análise precisa de pelo menos 2 fontes.")
    with st.expander("Erros das fontes"):
        for err in errors:
            st.write(f"- {err}")
    st.stop()

forecasts = pd.DataFrame(records)

with st.spinner("A calcular climatologia histórica..."):
    try:
        clim = fetch_climatology(lat, lon, target_day_str, unit_api, years_back, window_days)
    except Exception as e:
        clim = pd.DataFrame(columns=["date", "tmax"])
        errors.append(f"Climatologia: {e}")

mix = make_mixture_distribution(forecasts, clim, target_day, unit_factor, conservative)
components = mix["components"]

if bet_type == BET_EXACT:
    lower = float(target_value) - 0.5
    upper = float(target_value) + 0.5
    event_prob = mixture_probability_interval(components, lower, upper)
    event_desc = f"Tmax arredondada = {target_value:g}{symbol}"
else:
    event_prob = mixture_probability_over(components, float(target_value))
    event_desc = f"Tmax > {target_value:g}{symbol}"

ev = expected_value(event_prob, market_price)
decision = decide_signal(
    prob=event_prob,
    ev=ev,
    confidence=mix["confidence"],
    consensus=mix["consensus"],
    sigma=mix["sigma"],
    unit_factor=unit_factor,
    bet_type=bet_type,
)

q05 = mixture_quantile(components, 0.05, mix["mean"] - 8 * mix["sigma"], mix["mean"] + 8 * mix["sigma"])
q10 = mixture_quantile(components, 0.10, mix["mean"] - 8 * mix["sigma"], mix["mean"] + 8 * mix["sigma"])
q50 = mixture_quantile(components, 0.50, mix["mean"] - 8 * mix["sigma"], mix["mean"] + 8 * mix["sigma"])
q90 = mixture_quantile(components, 0.90, mix["mean"] - 8 * mix["sigma"], mix["mean"] + 8 * mix["sigma"])
q95 = mixture_quantile(components, 0.95, mix["mean"] - 8 * mix["sigma"], mix["mean"] + 8 * mix["sigma"])


# ============================================================
# OUTPUT
# ============================================================

st.markdown("## Decisão")

st.markdown(
    f"""
    <div class="{decision['css']}">
        <h2 style="margin-top:0">{decision['color']} · {decision['action']}</h2>
        <p style="font-size:1.05rem">{decision['explanation']}</p>
        <p><b>Evento analisado:</b> {event_desc}</p>
        <p><b>Alertas:</b> {decision['risk_flags']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Probabilidade estimada", f"{100 * event_prob:.1f}%")
m2.metric("Preço justo YES", f"{event_prob:.3f}", help="Probabilidade estimada convertida para preço justo binário.")
m3.metric("Edge melhor lado", f"{ev['best_edge']:+.3f}", help="Diferença entre probabilidade estimada e preço de mercado.")
m4.metric("Melhor lado", ev["best_side"])

m5, m6, m7, m8 = st.columns(4)
m5.metric("Tmax central", f"{mix['mean']:.1f}{symbol}")
m6.metric("Intervalo 80%", f"{q10:.1f}–{q90:.1f}{symbol}")
m7.metric("Consenso modelos", mix["consensus"])
m8.metric("Score técnico", f"{mix['confidence']:.0f}/100")

st.markdown("### Interpretação")
if ev["best_side"] == "YES":
    st.write(
        f"O mercado cobra **{market_price:.2f}** pelo YES. O modelo estima **{event_prob:.3f}**. "
        f"Edge YES = **{ev['edge_yes']:+.3f}** e ROI teórico por unidade arriscada ≈ **{100 * ev['roi_yes']:+.1f}%**."
    )
else:
    st.write(
        f"O mercado cobra **{market_price:.2f}** pelo YES, logo o NO custa aproximadamente **{1-market_price:.2f}**. "
        f"O modelo estima probabilidade de NO de **{1-event_prob:.3f}**. "
        f"Edge NO = **{ev['edge_no']:+.3f}** e ROI teórico por unidade arriscada ≈ **{100 * ev['roi_no']:+.1f}%**."
    )

st.write(
    f"A probabilidade do evento é **{probability_text(event_prob)}**. "
    f"A temperatura máxima central estimada é **{mix['mean']:.1f}{symbol}**, "
    f"com intervalo de 90% aproximado entre **{q05:.1f}{symbol}** e **{q95:.1f}{symbol}**."
)

tab1, tab2, tab3, tab4 = st.tabs(["Gráficos", "Modelos", "Matemática", "Riscos"])

with tab1:
    st.plotly_chart(chart_models(forecasts, target_value, mix["mean"], symbol, bet_type), use_container_width=True)
    st.plotly_chart(chart_distribution(components, target_value, symbol, bet_type), use_container_width=True)
    if not clim.empty:
        st.plotly_chart(chart_climatology(clim, mix["mean"], symbol), use_container_width=True)
    else:
        st.info("Sem climatologia histórica suficiente para este local.")

with tab2:
    table = forecasts.copy()
    numeric_cols = ["tmax", "tmin", "sensacao_max", "precipitacao", "vento_max", "peso_base"]
    for col in numeric_cols:
        table[col] = pd.to_numeric(table[col], errors="coerce").round(2)

    st.dataframe(
        table[
            [
                "modelo",
                "fornecedor",
                "codigo",
                "tmax",
                "tmin",
                "sensacao_max",
                "precipitacao",
                "vento_max",
                "peso_base",
                "nota",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    if errors:
        with st.expander("Fontes ignoradas / erros"):
            for err in errors:
                st.write(f"- {err}")

with tab3:
    st.markdown(
        f"""
        A análise usa uma **mistura de distribuições normais**, não apenas uma média simples.

        **1. Cada modelo é um componente probabilístico**

        Para cada modelo `i`:

        `Tmax_i ~ Normal(mu_i, sigma_i)`

        onde `mu_i` é a previsão do modelo e `sigma_i` aumenta com:
        - distância temporal da data;
        - conservadorismo escolhido;
        - possível comportamento de outlier.

        **2. Pesos**

        Os pesos base favorecem modelos premium como ECMWF/Best Match,
        mas previsões muito afastadas da mediana são penalizadas em vez de removidas.

        **3. Climatologia como prior**

        Se houver histórico suficiente, a app adiciona um componente de climatologia sazonal.
        O peso deste prior cresce com o horizonte temporal. Nesta análise:
        - peso do prior histórico: **{100 * mix['prior_weight']:.1f}%**
        - média climática: **{fmt_optional(mix['clim_mean'], symbol)}**
        - desvio climático: **{fmt_optional(mix['clim_std'], symbol)}**

        **4. Temperatura exata**

        Para aposta exata, a app calcula:

        `P(alvo - 0.5 <= Tmax < alvo + 0.5)`

        Isto corresponde a uma regra de arredondamento ao inteiro mais próximo.

        **5. Maior do que**

        Para aposta de linha, calcula:

        `P(Tmax > linha)`

        **6. Decisão verde/amarelo/vermelho**

        A cor combina:
        - edge contra o preço do mercado;
        - dispersão entre modelos;
        - largura da distribuição;
        - número de modelos válidos;
        - horizonte temporal;
        - consenso dos modelos.
        """
    )

    detail = pd.DataFrame(
        [
            {"métrica": "Média ponderada dos modelos", "valor": f"{mix['model_mean']:.2f}{symbol}"},
            {"métrica": "Dispersão ponderada entre modelos", "valor": f"{mix['model_spread']:.2f}{symbol}"},
            {"métrica": "Amplitude máx-min modelos", "valor": f"{mix['model_range']:.2f}{symbol}"},
            {"métrica": "Sigma base por horizonte", "valor": f"{mix['base_sigma']:.2f}{symbol}"},
            {"métrica": "Sigma final da mistura", "valor": f"{mix['sigma']:.2f}{symbol}"},
            {"métrica": "Horizonte", "valor": f"{mix['horizon']} dias"},
            {"métrica": "Probabilidade YES", "valor": f"{event_prob:.4f}"},
            {"métrica": "Probabilidade NO", "valor": f"{1-event_prob:.4f}"},
            {"métrica": "Edge YES", "valor": f"{ev['edge_yes']:+.4f}"},
            {"métrica": "Edge NO", "valor": f"{ev['edge_no']:+.4f}"},
        ]
    )
    st.dataframe(detail, hide_index=True, use_container_width=True)

with tab4:
    st.warning(
        """
        Antes de apostar, verifica sempre:
        1. qual é a estação meteorológica oficial usada na resolução;
        2. se a temperatura é em °C ou °F;
        3. se a regra é arredondamento, truncagem ou valor observado bruto;
        4. hora local do dia de resolução;
        5. liquidez, spread e possibilidade de mudança na fonte de resolução.
        """
    )
    st.write(
        "Mesmo um sinal verde não significa risco zero. Temperaturas máximas locais podem falhar por microclima, "
        "brisa marítima, altitude, nebulosidade, precipitação convectiva ou diferença entre grelha do modelo e estação oficial."
    )

st.caption(
    "Sem API key. Usa Open-Meteo para previsão, geocoding e histórico. Não executa trades; calcula sinal estatístico."
)


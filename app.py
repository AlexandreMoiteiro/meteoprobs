import math
from datetime import date, datetime, timedelta
from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Weather Edge",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}

    .main .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }

    .hero {
        padding: 1.35rem 1.5rem;
        border: 1px solid rgba(120,120,120,.22);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(255,255,255,.06), rgba(120,120,120,.06));
        margin-bottom: 1rem;
    }

    .small-muted {
        color: #777;
        font-size: .92rem;
    }

    .signal-card {
        padding: 1.15rem 1.25rem;
        border-radius: 22px;
        border: 1px solid rgba(120,120,120,.22);
        margin: .5rem 0 1rem 0;
    }

    .green {
        background: rgba(20, 150, 80, .13);
        border-color: rgba(20, 150, 80, .35);
    }

    .yellow {
        background: rgba(230, 170, 25, .15);
        border-color: rgba(230, 170, 25, .38);
    }

    .red {
        background: rgba(210, 60, 70, .13);
        border-color: rgba(210, 60, 70, .35);
    }

    .big-signal {
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: .2rem;
    }

    .metric-card {
        border: 1px solid rgba(120,120,120,.18);
        border-radius: 18px;
        padding: 1rem 1rem;
        background: rgba(120,120,120,.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

APP_USER_AGENT = "weather-edge-streamlit/2.0"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Só modelos fortes / institucionais. A app tenta cada um isoladamente;
# se algum não estiver disponível para a cidade/data, é ignorado com aviso.
PRECISE_MODELS = {
    "ECMWF IFS 0.25°": {
        "code": "ecmwf_ifs025",
        "weight": 1.35,
        "tier": "elite",
    },
    "ECMWF AIFS 0.25°": {
        "code": "ecmwf_aifs025_single",
        "weight": 1.20,
        "tier": "elite",
    },
    "DWD ICON Seamless": {
        "code": "icon_seamless",
        "weight": 1.10,
        "tier": "strong",
    },
    "Météo‑France Seamless": {
        "code": "meteofrance_seamless",
        "weight": 1.10,
        "tier": "strong",
    },
    "UK Met Office Seamless": {
        "code": "ukmo_seamless",
        "weight": 1.08,
        "tier": "strong",
    },
    "Open‑Meteo Best Match": {
        "code": "best_match",
        "weight": 1.05,
        "tier": "blend",
    },
    # GFS fica como fallback útil para cobertura global, mas com menor peso.
    "NOAA GFS Seamless": {
        "code": "gfs_seamless",
        "weight": 0.88,
        "tier": "fallback",
    },
}

DEFAULT_MODELS = [
    "ECMWF IFS 0.25°",
    "ECMWF AIFS 0.25°",
    "DWD ICON Seamless",
    "Météo‑France Seamless",
    "UK Met Office Seamless",
    "Open‑Meteo Best Match",
]

RISK_PROFILES = {
    "Normal": 1.00,
    "Conservador": 1.20,
    "Muito conservador": 1.45,
}


# ============================================================
# DATA FUNCTIONS
# ============================================================

def today_local() -> date:
    return date.today()


def safe_float(value):
    try:
        if value is None:
            return None
        value = float(value)
        if math.isnan(value):
            return None
        return value
    except Exception:
        return None


@st.cache_data(ttl=30 * 60, show_spinner=False)
def get_json(url: str, params: dict | None = None, timeout: int = 25) -> dict:
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": APP_USER_AGENT},
        timeout=timeout,
    )
    if response.status_code != 200:
        body = response.text[:450] if response.text else ""
        raise RuntimeError(f"HTTP {response.status_code}: {body}")
    return response.json()


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def geocode_city(city: str) -> list[dict]:
    data = get_json(
        OPEN_METEO_GEOCODING_URL,
        params={"name": city, "count": 8, "language": "pt", "format": "json"},
    )
    return data.get("results", []) or []


def format_place(item: dict) -> str:
    parts = [item.get("name"), item.get("admin1"), item.get("country")]
    text = ", ".join(str(x) for x in parts if x)
    lat = item.get("latitude")
    lon = item.get("longitude")
    return f"{text}  ·  {lat:.4f}, {lon:.4f}"


def parse_open_meteo_tmax(data: dict, target_day: date) -> float | None:
    daily = data.get("daily", {})
    times = daily.get("time", [])
    target = target_day.isoformat()
    if target not in times:
        return None

    idx = times.index(target)
    keys = [key for key in daily.keys() if key.startswith("temperature_2m_max")]
    for key in keys:
        values = daily.get(key, [])
        if isinstance(values, list) and idx < len(values):
            value = safe_float(values[idx])
            if value is not None:
                return value
    return None


@st.cache_data(ttl=20 * 60, show_spinner=False)
def fetch_model(lat: float, lon: float, target_day_str: str, model_name: str) -> dict:
    target_day = date.fromisoformat(target_day_str)
    meta = PRECISE_MODELS[model_name]
    code = meta["code"]

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "timezone": "auto",
        "temperature_unit": "celsius",
        "start_date": target_day_str,
        "end_date": target_day_str,
        "cell_selection": "land",
    }

    if code != "best_match":
        params["models"] = code

    data = get_json(OPEN_METEO_FORECAST_URL, params=params)
    tmax = parse_open_meteo_tmax(data, target_day)
    if tmax is None:
        raise RuntimeError("sem Tmax para esta data/local")

    return {
        "source": model_name,
        "model_code": code,
        "tier": meta["tier"],
        "tmax_c": float(tmax),
        "weight": float(meta["weight"]),
    }


def circular_doy_distance(a: int, b: int, days: int = 366) -> int:
    diff = abs(a - b)
    return min(diff, days - diff)


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def fetch_climatology(
    lat: float,
    lon: float,
    target_day_str: str,
    years_back: int,
    window_days: int,
) -> pd.DataFrame:
    target_day = date.fromisoformat(target_day_str)
    end_year = min(today_local().year - 1, target_day.year - 1)
    start_year = max(1940, end_year - years_back + 1)

    if end_year < start_year:
        return pd.DataFrame(columns=["date", "tmax_c"])

    data = get_json(
        OPEN_METEO_ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
            "daily": "temperature_2m_max",
            "temperature_unit": "celsius",
            "timezone": "auto",
        },
        timeout=35,
    )

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_max", [])
    target_doy = target_day.timetuple().tm_yday

    rows = []
    for d_str, temp in zip(dates, temps):
        value = safe_float(temp)
        if value is None:
            continue
        d = date.fromisoformat(d_str)
        if circular_doy_distance(d.timetuple().tm_yday, target_doy) <= window_days:
            rows.append({"date": d, "tmax_c": value})

    return pd.DataFrame(rows)


# ============================================================
# MATH FUNCTIONS
# ============================================================

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.where(weights <= 0, 1.0, weights)
    return float(np.average(values, weights=weights))


def weighted_std(values: np.ndarray, weights: np.ndarray, mean: float | None = None) -> float:
    if len(values) <= 1:
        return 0.0
    weights = np.where(weights <= 0, 1.0, weights)
    if mean is None:
        mean = weighted_mean(values, weights)
    var = float(np.average((values - mean) ** 2, weights=weights))
    return math.sqrt(max(var, 0.0))


def robust_location(values: np.ndarray, weights: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    w_mean = weighted_mean(values, weights)
    median = float(np.median(values))

    if len(values) >= 5:
        ordered = np.sort(values)
        trim_n = max(1, int(len(ordered) * 0.15))
        trimmed = ordered[trim_n:-trim_n] if len(ordered[trim_n:-trim_n]) else ordered
        trimmed_mean = float(np.mean(trimmed))
    else:
        trimmed_mean = w_mean

    # Mistura robusta: ECMWF/modelos bons via média ponderada, mas com travão por mediana.
    robust = 0.58 * w_mean + 0.27 * median + 0.15 * trimmed_mean

    return {
        "weighted_mean": w_mean,
        "median": median,
        "trimmed_mean": trimmed_mean,
        "robust": robust,
    }


def leave_one_out_sensitivity(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) <= 2:
        return 0.0
    base = weighted_mean(values, weights)
    deviations = []
    for i in range(len(values)):
        mask = np.ones(len(values), dtype=bool)
        mask[i] = False
        loo = weighted_mean(values[mask], weights[mask])
        deviations.append(abs(loo - base))
    return float(max(deviations)) if deviations else 0.0


def forecast_error_floor(horizon_days: int, n_models: int) -> float:
    # Piso conservador. Não é RMSE oficial; evita dar 99% de certeza só porque os modelos concordam.
    h = max(0, horizon_days)
    base = 0.90 + 0.16 * h
    if h >= 5:
        base += 0.20
    if h >= 10:
        base += 0.30
    if n_models < 4:
        base += 0.35
    return base


def shrink_to_climatology(forecast_mean: float, clim_mean: float | None, horizon_days: int) -> tuple[float, float]:
    if clim_mean is None or math.isnan(clim_mean):
        return forecast_mean, 1.0

    # Para 0-3 dias quase não mexe. Para 10-16 dias, puxa mais para o histórico.
    forecast_weight = 1.0 / (1.0 + (max(0, horizon_days) / 15.0) ** 2)
    posterior_mean = forecast_weight * forecast_mean + (1.0 - forecast_weight) * clim_mean
    return float(posterior_mean), float(forecast_weight)


def probability_for_market(dist: NormalDist, market_type: str, target_temp: float, exact_step: float) -> dict:
    if market_type == "Temperatura exata":
        half = exact_step / 2.0
        low = target_temp - half
        high = target_temp + half
        p_yes = dist.cdf(high) - dist.cdf(low)
        readable = f"Tmax entre {low:.2f} °C e {high:.2f} °C"
        boundary = (low, high)
    else:
        p_yes = 1.0 - dist.cdf(target_temp)
        readable = f"Tmax maior do que {target_temp:.2f} °C"
        boundary = (target_temp, None)

    return {
        "p_yes": max(0.0, min(1.0, float(p_yes))),
        "p_no": 1.0 - max(0.0, min(1.0, float(p_yes))),
        "readable": readable,
        "boundary": boundary,
    }


def score_signal(
    p_yes: float,
    market_price_yes: float,
    sigma: float,
    spread: float,
    horizon_days: int,
    n_models: int,
    loo: float,
    market_type: str,
) -> dict:
    edge = p_yes - market_price_yes
    fair_price = p_yes

    # Edge mínimo maior para temperatura exata, porque a resolução do mercado e estação oficial importam mais.
    if market_type == "Temperatura exata":
        green_edge = 0.10
        yellow_edge = 0.04
        max_sigma_green = 2.25
    else:
        green_edge = 0.07
        yellow_edge = 0.03
        max_sigma_green = 2.75

    data_quality = 100.0
    data_quality -= min(30, max(0, 4 - n_models) * 10)
    data_quality -= min(22, spread * 8)
    data_quality -= min(16, loo * 12)
    data_quality -= min(18, max(0, horizon_days - 4) * 2.2)
    data_quality -= min(18, max(0, sigma - 1.2) * 6)
    data_quality = max(0.0, min(100.0, data_quality))

    # Kelly teórico para contrato binário: b = payout odds netas.
    # Para YES com preço c e payout 1, lucro líquido por unidade arriscada = (1-c)/c.
    if 0 < market_price_yes < 1:
        b = (1.0 - market_price_yes) / market_price_yes
        kelly = (b * p_yes - (1 - p_yes)) / b
        kelly = max(0.0, min(float(kelly), 0.25))
    else:
        kelly = 0.0

    if (
        edge >= green_edge
        and data_quality >= 68
        and sigma <= max_sigma_green
        and spread <= 1.35
        and loo <= 0.55
        and n_models >= 4
    ):
        color = "green"
        label = "🟢 VERDE"
        decision = "Sinal estatístico favorável. Só faz sentido se as regras do mercado baterem certo com a fonte oficial de resolução."
    elif edge >= yellow_edge and data_quality >= 52 and n_models >= 3:
        color = "yellow"
        label = "🟡 AMARELO"
        decision = "Há algum edge, mas não é limpo. Reduz tamanho, espera melhor preço ou evita se a liquidez/spread for fraco."
    else:
        color = "red"
        label = "🔴 VERMELHO"
        decision = "Não há margem estatística suficiente para justificar a aposta com baixo risco."

    return {
        "edge": edge,
        "fair_price": fair_price,
        "data_quality": data_quality,
        "kelly": kelly,
        "color": color,
        "label": label,
        "decision": decision,
    }


def analyse(
    forecasts: pd.DataFrame,
    clim: pd.DataFrame,
    target_day: date,
    market_type: str,
    target_temp: float,
    exact_step: float,
    market_price_yes: float,
    risk_profile: str,
) -> dict:
    values = forecasts["tmax_c"].to_numpy(dtype=float)
    weights = forecasts["weight"].to_numpy(dtype=float)

    loc = robust_location(values, weights)
    spread = weighted_std(values, weights, loc["weighted_mean"])
    loo = leave_one_out_sensitivity(values, weights)
    horizon_days = max(0, (target_day - today_local()).days)

    clim_mean = None
    clim_std = None
    if not clim.empty:
        clim_mean = float(clim["tmax_c"].mean())
        clim_std = float(clim["tmax_c"].std(ddof=1)) if len(clim) > 1 else None

    posterior_mean, forecast_weight = shrink_to_climatology(loc["robust"], clim_mean, horizon_days)

    error_floor = forecast_error_floor(horizon_days, len(forecasts))
    seasonal_floor = 0.0
    if clim_std is not None and not math.isnan(clim_std):
        seasonal_floor = min(1.15, 0.16 * clim_std)

    risk_multiplier = RISK_PROFILES.get(risk_profile, 1.20)

    # Combina incertezas aproximadamente independentes.
    sigma = math.sqrt(
        spread**2
        + error_floor**2
        + seasonal_floor**2
        + min(1.0, loo * 1.15) ** 2
    ) * risk_multiplier
    sigma = max(0.70, float(sigma))

    dist = NormalDist(mu=posterior_mean, sigma=sigma)
    market = probability_for_market(dist, market_type, target_temp, exact_step)
    signal = score_signal(
        p_yes=market["p_yes"],
        market_price_yes=market_price_yes,
        sigma=sigma,
        spread=spread,
        horizon_days=horizon_days,
        n_models=len(forecasts),
        loo=loo,
        market_type=market_type,
    )

    # Massa por bins para mercado de temperatura exata.
    exact_grid = []
    step = exact_step
    center_min = math.floor((posterior_mean - 4 * sigma) / step) * step
    center_max = math.ceil((posterior_mean + 4 * sigma) / step) * step
    centers = np.arange(center_min, center_max + step, step)
    for c in centers:
        low = c - step / 2
        high = c + step / 2
        p = dist.cdf(high) - dist.cdf(low)
        exact_grid.append({"temp_c": round(float(c), 3), "prob": float(max(0, p))})
    exact_df = pd.DataFrame(exact_grid).sort_values("prob", ascending=False)

    return {
        "locations": loc,
        "posterior_mean": posterior_mean,
        "forecast_weight": forecast_weight,
        "spread": spread,
        "loo": loo,
        "horizon_days": horizon_days,
        "clim_mean": clim_mean,
        "clim_std": clim_std,
        "error_floor": error_floor,
        "seasonal_floor": seasonal_floor,
        "sigma": sigma,
        "dist": dist,
        "market": market,
        "signal": signal,
        "ci80": (dist.inv_cdf(0.10), dist.inv_cdf(0.90)),
        "ci90": (dist.inv_cdf(0.05), dist.inv_cdf(0.95)),
        "ci95": (dist.inv_cdf(0.025), dist.inv_cdf(0.975)),
        "exact_df": exact_df,
    }


# ============================================================
# CHARTS
# ============================================================

def source_chart(forecasts: pd.DataFrame, posterior_mean: float) -> go.Figure:
    df = forecasts.sort_values("tmax_c")
    fig = px.bar(
        df,
        x="tmax_c",
        y="source",
        orientation="h",
        color="tier",
        text=df["tmax_c"].map(lambda x: f"{x:.1f}°"),
        labels={"tmax_c": "Tmax prevista (°C)", "source": "Modelo", "tier": "Classe"},
        title="Previsões dos modelos selecionados",
    )
    fig.add_vline(x=posterior_mean, line_dash="dash", annotation_text="estimativa final")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=max(390, 48 * len(df)), margin=dict(l=10, r=10, t=60, b=10))
    return fig


def distribution_chart(analysis: dict, target_temp: float, market_type: str) -> go.Figure:
    mu = analysis["posterior_mean"]
    sigma = analysis["sigma"]
    dist = analysis["dist"]
    xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 380)
    ys = [dist.pdf(float(x)) for x in xs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Distribuição estimada"))
    fig.add_vline(x=mu, line_dash="dash", annotation_text="estimativa central")

    if market_type == "Temperatura exata":
        low, high = analysis["market"]["boundary"]
        fig.add_vrect(x0=low, x1=high, opacity=0.18, line_width=0, annotation_text="zona YES")
    else:
        fig.add_vline(x=target_temp, line_dash="dot", annotation_text="linha do mercado")

    fig.update_layout(
        title="Distribuição probabilística da Tmax",
        xaxis_title="Temperatura máxima diária (°C)",
        yaxis_title="Densidade",
        height=390,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def exact_probs_chart(exact_df: pd.DataFrame, target_temp: float) -> go.Figure:
    top = exact_df.head(12).sort_values("temp_c")
    fig = px.bar(
        top,
        x="temp_c",
        y="prob",
        text=top["prob"].map(lambda p: f"{100*p:.1f}%"),
        labels={"temp_c": "Temperatura exata / bucket (°C)", "prob": "Probabilidade"},
        title="Temperaturas exatas mais prováveis",
    )
    fig.add_vline(x=target_temp, line_dash="dot", annotation_text="alvo")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=360, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=60, b=10))
    return fig


def climatology_chart(clim: pd.DataFrame, posterior_mean: float) -> go.Figure:
    fig = px.histogram(
        clim,
        x="tmax_c",
        nbins=34,
        labels={"tmax_c": "Tmax histórica na mesma época do ano (°C)"},
        title="Contexto histórico sazonal",
    )
    fig.add_vline(x=posterior_mean, line_dash="dash", annotation_text="estimativa atual")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=60, b=10))
    return fig


# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:.2rem">🌡️ Weather Edge</h1>
        <div class="small-muted">
            Análise probabilística da temperatura máxima diária para mercados tipo Polymarket.
            Usa modelos meteorológicos institucionais, climatologia e incerteza conservadora.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.subheader("1) Mercado")
    r1c1, r1c2, r1c3 = st.columns([1.3, 1, 1])

    with r1c1:
        city = st.text_input("Cidade", value="Lisboa, Portugal", placeholder="Ex.: Lisboa, Portugal")

    with r1c2:
        target_day = st.date_input(
            "Data",
            value=today_local() + timedelta(days=3),
            min_value=today_local(),
            max_value=today_local() + timedelta(days=16),
        )

    with r1c3:
        market_type = st.radio(
            "Tipo",
            ["Temperatura exata", "Maior do que"],
            horizontal=True,
            index=0,
        )

    r2c1, r2c2, r2c3 = st.columns([1, 1, 1])

    with r2c1:
        target_temp = st.number_input(
            "Temperatura-alvo / linha (°C)",
            value=25.0,
            step=0.5,
            format="%.1f",
        )

    with r2c2:
        market_price_yes = st.number_input(
            "Preço YES no mercado",
            min_value=0.01,
            max_value=0.99,
            value=0.20 if market_type == "Temperatura exata" else 0.50,
            step=0.01,
            format="%.2f",
            help="Ex.: 0.32 significa que o mercado paga como se a probabilidade fosse 32% antes de fees/spread.",
        )

    with r2c3:
        exact_step = st.selectbox(
            "Resolução para temperatura exata",
            [1.0, 0.5, 0.1],
            index=0,
            format_func=lambda x: f"bucket de ±{x/2:g} °C",
            disabled=(market_type != "Temperatura exata"),
            help="Se o mercado resolve por graus inteiros, usa 1.0. Se resolve por décimos, usa 0.1.",
        )

with st.expander("Configuração avançada", expanded=False):
    a1, a2, a3 = st.columns([1.2, 1, 1])

    with a1:
        selected_models = st.multiselect(
            "Modelos usados",
            options=list(PRECISE_MODELS.keys()),
            default=DEFAULT_MODELS,
        )

    with a2:
        risk_profile = st.selectbox(
            "Perfil de incerteza",
            options=list(RISK_PROFILES.keys()),
            index=1,
        )

    with a3:
        years_back = st.slider("Anos históricos", 8, 30, 18, 1)
        window_days = st.slider("Janela sazonal ± dias", 4, 21, 10, 1)

run = st.button("Analisar mercado", type="primary", use_container_width=True)

st.caption(
    "Nota: a app dá um sinal estatístico, não uma garantia de lucro. Confirma sempre a fonte de resolução oficial do mercado, fuso horário, spread e liquidez."
)

if not run:
    st.stop()

if not city.strip():
    st.error("Escreve uma cidade.")
    st.stop()

if len(selected_models) < 3:
    st.error("Usa pelo menos 3 modelos para uma análise minimamente robusta.")
    st.stop()

# ============================================================
# RUN PIPELINE
# ============================================================

try:
    matches = geocode_city(city.strip())
except Exception as exc:
    st.error(f"Erro ao procurar cidade: {exc}")
    st.stop()

if not matches:
    st.error("Não encontrei a cidade. Tenta incluir país/região, por exemplo: 'Porto, Portugal'.")
    st.stop()

if len(matches) == 1:
    place = matches[0]
else:
    place = st.selectbox("Escolhe a localização correta", matches, format_func=format_place)

lat = float(place["latitude"])
lon = float(place["longitude"])
timezone_name = place.get("timezone") or "UTC"

warnings = []
records = []

with st.status("A calcular previsões e incerteza...", expanded=False) as status:
    st.write("A consultar modelos meteorológicos...")
    for model_name in selected_models:
        try:
            records.append(fetch_model(lat, lon, target_day.isoformat(), model_name))
        except Exception as exc:
            warnings.append(f"{model_name}: {exc}")

    if len(records) < 3:
        st.error("Poucos modelos devolveram dados. Tenta outra data/cidade ou ativa mais modelos.")
        if warnings:
            with st.expander("Avisos técnicos"):
                for w in warnings:
                    st.write(f"- {w}")
        st.stop()

    st.write("A obter climatologia histórica...")
    try:
        clim = fetch_climatology(lat, lon, target_day.isoformat(), years_back, window_days)
    except Exception as exc:
        clim = pd.DataFrame(columns=["date", "tmax_c"])
        warnings.append(f"Climatologia: {exc}")

    forecasts = pd.DataFrame(records)
    analysis = analyse(
        forecasts=forecasts,
        clim=clim,
        target_day=target_day,
        market_type=market_type,
        target_temp=float(target_temp),
        exact_step=float(exact_step),
        market_price_yes=float(market_price_yes),
        risk_profile=risk_profile,
    )
    status.update(label="Análise concluída", state="complete")

# ============================================================
# OUTPUT
# ============================================================

st.markdown("---")
st.subheader("2) Decisão")

signal = analysis["signal"]
market = analysis["market"]

st.markdown(
    f"""
    <div class="signal-card {signal['color']}">
        <div class="big-signal">{signal['label']}</div>
        <div style="font-size:1.08rem; margin-bottom:.35rem">{signal['decision']}</div>
        <div class="small-muted">Mercado analisado: <b>{market['readable']}</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Probabilidade YES", f"{100 * market['p_yes']:.1f}%")
m2.metric("Preço justo YES", f"{signal['fair_price']:.3f}")
m3.metric("Edge vs mercado", f"{signal['edge']:+.3f}")
m4.metric("Qualidade técnica", f"{signal['data_quality']:.0f}/100")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Tmax central", f"{analysis['posterior_mean']:.1f} °C")
m6.metric("Intervalo 90%", f"{analysis['ci90'][0]:.1f}–{analysis['ci90'][1]:.1f} °C")
m7.metric("Incerteza σ", f"{analysis['sigma']:.2f} °C")
m8.metric("Kelly teórico capado", f"{100 * signal['kelly']:.1f}%")

if signal["kelly"] > 0:
    st.info(
        "Kelly é apenas uma referência matemática para tamanho máximo teórico. Em mercados ilíquidos, usa uma fração muito menor ou zero."
    )

st.subheader("3) Análise matemática")

math_c1, math_c2 = st.columns([1, 1])
with math_c1:
    st.markdown(
        f"""
        **Modelo probabilístico usado**

        - Média ponderada dos modelos: **{analysis['locations']['weighted_mean']:.2f} °C**
        - Mediana dos modelos: **{analysis['locations']['median']:.2f} °C**
        - Média robusta/trimmed: **{analysis['locations']['trimmed_mean']:.2f} °C**
        - Estimativa robusta antes da climatologia: **{analysis['locations']['robust']:.2f} °C**
        - Peso dado à previsão vs climatologia: **{100 * analysis['forecast_weight']:.1f}%**
        - Dispersão ponderada entre modelos: **{analysis['spread']:.2f} °C**
        - Sensibilidade leave‑one‑out: **{analysis['loo']:.2f} °C**
        - Piso de erro por horizonte: **{analysis['error_floor']:.2f} °C**
        - Piso sazonal histórico: **{analysis['seasonal_floor']:.2f} °C**
        """
    )

with math_c2:
    if analysis["clim_mean"] is not None:
        st.markdown(
            f"""
            **Climatologia da mesma época**

            - Observações históricas usadas: **{len(clim)}**
            - Média histórica sazonal: **{analysis['clim_mean']:.2f} °C**
            - Desvio-padrão histórico sazonal: **{analysis['clim_std']:.2f} °C**
            - Horizonte da aposta: **{analysis['horizon_days']} dias**
            - Perfil de incerteza: **{risk_profile}**

            A distribuição final é aproximada como normal, com média **{analysis['posterior_mean']:.2f} °C** e σ **{analysis['sigma']:.2f} °C**.
            """
        )
    else:
        st.markdown(
            f"""
            **Climatologia indisponível**

            A análise usou apenas o ensemble de modelos e pisos conservadores de erro.

            - Horizonte da aposta: **{analysis['horizon_days']} dias**
            - Perfil de incerteza: **{risk_profile}**
            - Distribuição final: média **{analysis['posterior_mean']:.2f} °C**, σ **{analysis['sigma']:.2f} °C**
            """
        )

with st.expander("Interpretação do verde/amarelo/vermelho"):
    st.markdown(
        """
        **Verde** exige edge positivo claro, boa qualidade técnica, vários modelos disponíveis, baixa dispersão e baixa sensibilidade a remover um modelo.

        **Amarelo** significa que há algum edge, mas a margem pode desaparecer com spread, mudança de previsão, fonte oficial diferente ou baixa liquidez.

        **Vermelho** significa que a app não encontrou edge suficiente para considerar a aposta de baixo risco.

        Para mercados de **temperatura exata**, a app é mais exigente, porque pequenos erros de arredondamento, estação oficial e fuso horário podem mudar totalmente a resolução.
        """
    )

st.subheader("4) Gráficos")

g1, g2 = st.columns([1, 1])
with g1:
    st.plotly_chart(source_chart(forecasts, analysis["posterior_mean"]), use_container_width=True)
with g2:
    st.plotly_chart(distribution_chart(analysis, float(target_temp), market_type), use_container_width=True)

if market_type == "Temperatura exata":
    st.plotly_chart(exact_probs_chart(analysis["exact_df"], float(target_temp)), use_container_width=True)

if not clim.empty:
    st.plotly_chart(climatology_chart(clim, analysis["posterior_mean"]), use_container_width=True)

st.subheader("5) Dados")

show_df = forecasts.copy().sort_values("tmax_c", ascending=False)
show_df["tmax_c"] = show_df["tmax_c"].round(2)
show_df["weight"] = show_df["weight"].round(2)
st.dataframe(
    show_df[["source", "tier", "model_code", "tmax_c", "weight"]],
    hide_index=True,
    use_container_width=True,
)

if market_type == "Temperatura exata":
    top_exact = analysis["exact_df"].head(8).copy()
    top_exact["prob_%"] = (100 * top_exact["prob"]).round(2)
    top_exact = top_exact[["temp_c", "prob_%"]]
    st.markdown("**Top temperaturas exatas estimadas**")
    st.dataframe(top_exact, hide_index=True, use_container_width=True)

if warnings:
    with st.expander("Avisos técnicos das fontes"):
        for w in warnings:
            st.warning(w)

st.caption(
    "Versão simples sem sidebar e sem secrets. Usa Open‑Meteo para previsões e histórico. Não faz trading automático nem substitui a validação das regras oficiais do mercado."
)

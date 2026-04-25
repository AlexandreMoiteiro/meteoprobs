import math
from datetime import date, timedelta
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Weather Edge Pro",
    page_icon="🌡️",
    layout="wide",
)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HEADERS = {"User-Agent": "weather-edge-pro-streamlit/2.0"}

MODELS = {
    "Best Match Open-Meteo": "best_match",
    "ECMWF IFS 0.25°": "ecmwf_ifs025",
    "ECMWF AIFS": "ecmwf_aifs025_single",
    "NOAA GFS": "gfs_seamless",
    "DWD ICON": "icon_seamless",
    "Météo-France": "meteofrance_seamless",
    "UK Met Office": "ukmo_seamless",
    "GEM Canadá": "gem_seamless",
    "BOM ACCESS Global": "bom_access_global",
}

MODEL_WEIGHTS = {
    "best_match": 1.20,
    "ecmwf_ifs025": 1.25,
    "ecmwf_aifs025_single": 1.10,
    "gfs_seamless": 1.00,
    "icon_seamless": 1.05,
    "meteofrance_seamless": 1.05,
    "ukmo_seamless": 1.05,
    "gem_seamless": 0.95,
    "bom_access_global": 0.95,
}

DEFAULT_MODELS = [
    "Best Match Open-Meteo",
    "ECMWF IFS 0.25°",
    "ECMWF AIFS",
    "NOAA GFS",
    "DWD ICON",
    "Météo-France",
    "UK Met Office",
]

RISK_MULTIPLIERS = {
    "Normal": 1.00,
    "Conservador": 1.25,
    "Muito conservador": 1.50,
}

LIGHT_COLORS = {
    "VERDE": "#16a34a",
    "AMARELO": "#ca8a04",
    "VERMELHO": "#dc2626",
}

LIGHT_BG = {
    "VERDE": "rgba(22, 163, 74, 0.12)",
    "AMARELO": "rgba(202, 138, 4, 0.13)",
    "VERMELHO": "rgba(220, 38, 38, 0.12)",
}


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1180px;
        }
        .hero {
            padding: 1.35rem 1.5rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(59,130,246,0.13), rgba(16,185,129,0.10));
            margin-bottom: 1.1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.25rem;
            line-height: 1.1;
        }
        .muted {
            color: #64748b;
            font-size: 0.96rem;
        }
        .result-card {
            padding: 1.25rem 1.35rem;
            border-radius: 24px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }
        .result-title {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .result-subtitle {
            font-size: 1.05rem;
            color: #334155;
            margin-bottom: 0.15rem;
        }
        .small-pill {
            display: inline-block;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            background: rgba(100,116,139,0.12);
            color: #334155;
            font-size: 0.85rem;
            margin-right: 0.3rem;
            margin-top: 0.25rem;
        }
        .formula-box {
            padding: 0.9rem 1rem;
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: rgba(248, 250, 252, 0.72);
            font-size: 0.96rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API HELPERS
# ============================================================

@st.cache_data(ttl=30 * 60, show_spinner=False)
def get_json(url, params=None):
    response = requests.get(url, params=params, headers=HEADERS, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def geocode_city(city):
    data = get_json(
        OPEN_METEO_GEOCODING_URL,
        params={
            "name": city,
            "count": 8,
            "language": "pt",
            "format": "json",
        },
    )
    return data.get("results", []) or []


def format_place(place):
    parts = [place.get("name"), place.get("admin1"), place.get("country")]
    return ", ".join([str(p) for p in parts if p])


def safe_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except Exception:
        return None


def parse_tmax(data, target_day):
    daily = data.get("daily", {})
    days = daily.get("time", [])
    target = target_day.isoformat()

    if target not in days:
        return None

    idx = days.index(target)

    for key, values in daily.items():
        if key.startswith("temperature_2m_max") and isinstance(values, list):
            if idx < len(values):
                return safe_float(values[idx])

    return None


@st.cache_data(ttl=20 * 60, show_spinner=False)
def fetch_model_forecast(lat, lon, target_day_str, model_name, model_code):
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

    if model_code != "best_match":
        params["models"] = model_code

    data = get_json(OPEN_METEO_FORECAST_URL, params=params)
    tmax = parse_tmax(data, date.fromisoformat(target_day_str))

    if tmax is None:
        raise RuntimeError("Sem temperature_2m_max para esta data/modelo.")

    return {
        "modelo": model_name,
        "codigo": model_code,
        "tmax": tmax,
        "peso": MODEL_WEIGHTS.get(model_code, 1.0),
    }


def doy_distance(a, b, days=366):
    diff = abs(a - b)
    return min(diff, days - diff)


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def fetch_climatology(lat, lon, target_day_str, years_back=20, window_days=10):
    target_day = date.fromisoformat(target_day_str)
    end_year = min(date.today().year - 1, target_day.year - 1)
    start_year = max(1940, end_year - years_back + 1)

    if end_year < start_year:
        return pd.DataFrame(columns=["data", "tmax"])

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
        if doy_distance(d.timetuple().tm_yday, target_doy) <= window_days:
            rows.append({"data": d, "tmax": value})

    return pd.DataFrame(rows)


# ============================================================
# MATH
# ============================================================

def weighted_mean_std(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = np.where(weights <= 0, 1.0, weights)

    mean = float(np.average(values, weights=weights))

    if len(values) <= 1:
        return mean, 0.0

    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, math.sqrt(max(variance, 0.0))


def weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum = np.cumsum(weights) / np.sum(weights)
    return float(values[np.searchsorted(cum, 0.5)])


def trimmed_mean(values, trim=0.15):
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    if n <= 3:
        return float(np.mean(values))
    k = int(math.floor(n * trim))
    if 2 * k >= n:
        return float(np.mean(values))
    return float(np.mean(values[k:n-k]))


def robust_mad(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def base_forecast_error(horizon_days):
    """
    Piso conservador de erro para Tmax diária.
    Não é validação oficial; é uma proteção contra excesso de confiança.
    """
    h = max(0, int(horizon_days))
    return 0.95 + 0.17 * h + (0.20 if h >= 5 else 0.0) + (0.25 if h >= 10 else 0.0)


def scale_0_100(x, low, high):
    if high == low:
        return 0.0
    return float(np.clip(100 * (x - low) / (high - low), 0, 100))


def probability_from_normal(mu, sigma, line_c):
    dist = NormalDist(mu=mu, sigma=max(0.01, sigma))
    return 1.0 - dist.cdf(line_c)


def simulate_temperature_distribution(
    forecasts,
    clim_mean,
    forecast_weight,
    error_floor,
    seasonal_error,
    risk_multiplier,
    n_samples=40000,
):
    rng = np.random.default_rng(42)

    values = forecasts["tmax"].to_numpy(dtype=float)
    weights = forecasts["peso"].to_numpy(dtype=float)
    probs = weights / weights.sum()

    chosen = rng.choice(values, size=n_samples, replace=True, p=probs)

    if clim_mean is not None and not math.isnan(clim_mean):
        centers = forecast_weight * chosen + (1.0 - forecast_weight) * clim_mean
    else:
        centers = chosen

    normal_noise = rng.normal(
        loc=0.0,
        scale=max(0.4, math.sqrt(error_floor**2 + seasonal_error**2)) * risk_multiplier,
        size=n_samples,
    )

    # Cauda pesada: alguns dias têm erro sistemático local, estação diferente ou evento convectivo.
    tail_event = rng.random(n_samples) < 0.08
    tail_noise = rng.normal(
        loc=0.0,
        scale=max(0.8, 1.75 * error_floor) * risk_multiplier,
        size=n_samples,
    )

    return centers + normal_noise + tail_event * tail_noise


def analyse(forecasts, climatology, target_day, line_c, yes_price, risk_profile, resolution_risk):
    values = forecasts["tmax"].to_numpy(dtype=float)
    weights = forecasts["peso"].to_numpy(dtype=float)
    n_models = len(values)

    weighted_mean, model_spread = weighted_mean_std(values, weights)
    w_median = weighted_median(values, weights)
    t_mean = trimmed_mean(values)
    mad = robust_mad(values)
    robust_sigma = 1.4826 * mad

    ensemble_center = 0.55 * weighted_mean + 0.30 * w_median + 0.15 * t_mean
    horizon = max(0, (target_day - date.today()).days)

    clim_mean = None
    clim_std = None
    clim_percentile = None

    if not climatology.empty:
        clim_mean = float(climatology["tmax"].mean())
        clim_std = float(climatology["tmax"].std(ddof=1))
        clim_percentile = float((climatology["tmax"] <= ensemble_center).mean())

    # Quanto mais longe, mais a previsão é puxada para a climatologia histórica.
    if clim_mean is not None and not math.isnan(clim_mean):
        forecast_weight = 1.0 / (1.0 + (horizon / 16.0) ** 2)
        final_mean = forecast_weight * ensemble_center + (1.0 - forecast_weight) * clim_mean
    else:
        forecast_weight = 1.0
        final_mean = ensemble_center

    error_floor = base_forecast_error(horizon)
    seasonal_error = 0.14 * clim_std if clim_std is not None and not math.isnan(clim_std) else 0.0
    small_sample_penalty = 0.65 if n_models < 4 else 0.35 if n_models < 6 else 0.0
    disagreement = max(model_spread, 0.75 * robust_sigma)
    risk_multiplier = RISK_MULTIPLIERS.get(risk_profile, 1.25)

    sigma = math.sqrt(
        error_floor**2
        + disagreement**2
        + seasonal_error**2
        + small_sample_penalty**2
    ) * risk_multiplier
    sigma = max(0.85, sigma)

    p_normal_yes = probability_from_normal(final_mean, sigma, line_c)

    simulated = simulate_temperature_distribution(
        forecasts=forecasts,
        clim_mean=clim_mean,
        forecast_weight=forecast_weight,
        error_floor=error_floor,
        seasonal_error=seasonal_error,
        risk_multiplier=risk_multiplier,
    )
    p_mc_yes = float(np.mean(simulated >= line_c))

    # Probabilidade usada no semáforo: mistura e depois aplica haircut conservador.
    p_blend_yes = 0.55 * p_normal_yes + 0.45 * p_mc_yes
    uncertainty_haircut = 0.015 + 0.006 * horizon + 0.015 * max(0, 5 - n_models)
    uncertainty_haircut += 0.020 if model_spread > 1.5 else 0.0
    uncertainty_haircut += 0.025 if model_spread > 2.2 else 0.0
    uncertainty_haircut += 0.010 * resolution_risk

    p_cons_yes = float(np.clip(p_blend_yes - uncertainty_haircut, 0.0, 1.0))
    p_cons_no = float(np.clip(1.0 - p_blend_yes - uncertainty_haircut, 0.0, 1.0))

    yes_edge = p_cons_yes - yes_price
    no_price = 1.0 - yes_price
    no_edge = p_cons_no - no_price

    if yes_edge >= no_edge:
        side = "YES"
        chosen_prob = p_cons_yes
        chosen_price = yes_price
        chosen_edge = yes_edge
        margin_c = final_mean - line_c
    else:
        side = "NO"
        chosen_prob = p_cons_no
        chosen_price = no_price
        chosen_edge = no_edge
        margin_c = line_c - final_mean

    margin_sigma = margin_c / sigma

    prob_score = scale_0_100(chosen_prob, 0.52, 0.90)
    edge_score = scale_0_100(chosen_edge, 0.00, 0.12)
    margin_score = scale_0_100(margin_sigma, 0.10, 1.35)
    consensus_score = 100 - scale_0_100(model_spread, 0.60, 3.00)
    horizon_score = 100 - scale_0_100(horizon, 2, 16)
    data_score = scale_0_100(n_models, 3, 7)
    resolution_score = 100 - 10 * resolution_risk

    technical_score = (
        0.24 * prob_score
        + 0.24 * edge_score
        + 0.17 * margin_score
        + 0.14 * consensus_score
        + 0.09 * horizon_score
        + 0.07 * data_score
        + 0.05 * resolution_score
    )
    technical_score = float(np.clip(technical_score, 0, 100))

    hard_red_reasons = []
    if n_models < 3:
        hard_red_reasons.append("poucos modelos disponíveis")
    if chosen_edge <= 0.015:
        hard_red_reasons.append("edge insuficiente")
    if chosen_prob < 0.58:
        hard_red_reasons.append("probabilidade conservadora baixa")
    if abs(margin_sigma) < 0.25:
        hard_red_reasons.append("linha demasiado perto da estimativa")

    if hard_red_reasons:
        light = "VERMELHO"
    elif (
        chosen_prob >= 0.82
        and chosen_edge >= 0.07
        and margin_sigma >= 0.75
        and model_spread <= 1.70
        and sigma <= 3.90
        and n_models >= 5
        and technical_score >= 74
        and resolution_risk <= 5
    ):
        light = "VERDE"
    elif chosen_prob >= 0.64 and chosen_edge >= 0.03 and margin_sigma >= 0.35 and technical_score >= 52:
        light = "AMARELO"
    else:
        light = "VERMELHO"

    if light == "VERDE":
        advice = "Setup favorável: apostar só faz sentido no lado indicado e com gestão de banca."
    elif light == "AMARELO":
        advice = "Zona intermédia: só consideraria se o preço melhorar, com stake pequena, ou esperaria nova atualização dos modelos."
    else:
        advice = "Não aconselho apostar: a margem estatística não compensa o risco estimado."

    normal_dist = NormalDist(mu=final_mean, sigma=sigma)

    stress_rows = []
    for shock in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        p_yes_shock = probability_from_normal(final_mean + shock, sigma, line_c)
        p_side = p_yes_shock if side == "YES" else 1.0 - p_yes_shock
        stress_rows.append(
            {
                "Cenário": f"Erro sistemático {shock:+.1f} °C",
                "Prob. lado escolhido": p_side,
                "Edge estimado": p_side - chosen_price,
            }
        )

    return {
        "weighted_mean": weighted_mean,
        "weighted_median": w_median,
        "trimmed_mean": t_mean,
        "ensemble_center": ensemble_center,
        "model_spread": model_spread,
        "mad": mad,
        "robust_sigma": robust_sigma,
        "final_mean": final_mean,
        "sigma": sigma,
        "error_floor": error_floor,
        "seasonal_error": seasonal_error,
        "small_sample_penalty": small_sample_penalty,
        "forecast_weight": forecast_weight,
        "horizon": horizon,
        "clim_mean": clim_mean,
        "clim_std": clim_std,
        "clim_percentile": clim_percentile,
        "p_normal_yes": p_normal_yes,
        "p_mc_yes": p_mc_yes,
        "p_blend_yes": p_blend_yes,
        "p_cons_yes": p_cons_yes,
        "p_cons_no": p_cons_no,
        "yes_edge": yes_edge,
        "no_edge": no_edge,
        "side": side,
        "chosen_prob": chosen_prob,
        "chosen_price": chosen_price,
        "chosen_edge": chosen_edge,
        "margin_c": margin_c,
        "margin_sigma": margin_sigma,
        "technical_score": technical_score,
        "light": light,
        "advice": advice,
        "hard_red_reasons": hard_red_reasons,
        "scores": {
            "Probabilidade": prob_score,
            "Edge": edge_score,
            "Margem vs linha": margin_score,
            "Consenso modelos": consensus_score,
            "Horizonte": horizon_score,
            "Quantidade de dados": data_score,
            "Risco de resolução": resolution_score,
        },
        "interval_80": (normal_dist.inv_cdf(0.10), normal_dist.inv_cdf(0.90)),
        "interval_90": (normal_dist.inv_cdf(0.05), normal_dist.inv_cdf(0.95)),
        "interval_95": (normal_dist.inv_cdf(0.025), normal_dist.inv_cdf(0.975)),
        "simulated": simulated,
        "stress": pd.DataFrame(stress_rows),
    }


def build_leave_one_out(forecasts, base_result, line_c, yes_price):
    rows = []
    if len(forecasts) < 4:
        return pd.DataFrame(rows)

    for idx in forecasts.index:
        reduced = forecasts.drop(index=idx).reset_index(drop=True)
        vals = reduced["tmax"].to_numpy(dtype=float)
        w = reduced["peso"].to_numpy(dtype=float)
        mean, spread = weighted_mean_std(vals, w)
        med = weighted_median(vals, w)
        tm = trimmed_mean(vals)
        center = 0.55 * mean + 0.30 * med + 0.15 * tm
        p_yes = probability_from_normal(center, base_result["sigma"], line_c)
        side_prob = p_yes if base_result["side"] == "YES" else 1.0 - p_yes
        price = yes_price if base_result["side"] == "YES" else 1.0 - yes_price
        rows.append(
            {
                "Sem o modelo": forecasts.loc[idx, "modelo"],
                "Tmax centro": center,
                "Dispersão": spread,
                "Prob. lado escolhido": side_prob,
                "Edge": side_prob - price,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# CHARTS
# ============================================================

def source_chart(forecasts, final_mean, line_c):
    df = forecasts.sort_values("tmax")
    fig = px.bar(
        df,
        x="tmax",
        y="modelo",
        orientation="h",
        text=df["tmax"].map(lambda x: f"{x:.1f}°C"),
        labels={"tmax": "Tmax prevista", "modelo": "Modelo"},
    )
    fig.add_vline(x=final_mean, line_dash="dash", annotation_text="estimativa final")
    fig.add_vline(x=line_c, line_dash="dot", annotation_text="linha mercado")
    fig.update_layout(height=max(380, 48 * len(df)), margin=dict(l=10, r=10, t=25, b=10), showlegend=False)
    return fig


def probability_chart(mu, sigma, line_c, side):
    dist = NormalDist(mu=mu, sigma=sigma)
    xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 350)
    ys = np.array([dist.pdf(float(x)) for x in xs])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Distribuição"))
    fig.add_vline(x=mu, line_dash="dash", annotation_text="estimativa")
    fig.add_vline(x=line_c, line_dash="dot", annotation_text="linha")

    if side == "YES":
        fill_x = xs[xs >= line_c]
    else:
        fill_x = xs[xs <= line_c]
    fill_y = np.array([dist.pdf(float(x)) for x in fill_x])
    fig.add_trace(
        go.Scatter(
            x=fill_x,
            y=fill_y,
            fill="tozeroy",
            mode="none",
            name=f"Zona {side}",
        )
    )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Temperatura máxima diária (°C)",
        yaxis_title="Densidade",
        showlegend=False,
    )
    return fig


def score_chart(scores):
    df = pd.DataFrame(
        {"Fator": list(scores.keys()), "Score": [round(v, 1) for v in scores.values()]}
    ).sort_values("Score")
    fig = px.bar(df, x="Score", y="Fator", orientation="h", range_x=[0, 100])
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=25, b=10), showlegend=False)
    return fig


def simulation_chart(simulated, line_c, final_mean):
    fig = px.histogram(
        x=simulated,
        nbins=60,
        labels={"x": "Tmax simulada (°C)", "y": "Frequência"},
    )
    fig.add_vline(x=final_mean, line_dash="dash", annotation_text="estimativa")
    fig.add_vline(x=line_c, line_dash="dot", annotation_text="linha")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=25, b=10), showlegend=False)
    return fig


def stress_chart(stress):
    df = stress.copy()
    df["Prob. lado escolhido"] = 100 * df["Prob. lado escolhido"]
    fig = px.line(
        df,
        x="Cenário",
        y="Prob. lado escolhido",
        markers=True,
        labels={"Prob. lado escolhido": "Probabilidade (%)"},
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=25, b=10), yaxis_range=[0, 100])
    return fig


# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🌡️ Weather Edge Pro</h1>
        <div class="muted">
            Semáforo estatístico para mercados de temperatura máxima: modelos meteorológicos, climatologia,
            incerteza, stress tests e edge vs preço.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    top1, top2, top3, top4 = st.columns([1.7, 1.05, 1.05, 0.9])

    with top1:
        city = st.text_input("Cidade", value="Lisboa", placeholder="Ex.: Lisboa, Portugal")

    with top2:
        target_day = st.date_input(
            "Dia",
            value=date.today() + timedelta(days=3),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=16),
        )

    with top3:
        line_c = st.number_input(
            "Linha: Tmax ≥ X °C",
            value=25.0,
            step=0.5,
            format="%.1f",
        )

    with top4:
        yes_price = st.number_input(
            "Preço YES",
            min_value=0.01,
            max_value=0.99,
            value=0.50,
            step=0.01,
        )

    with st.expander("Opções avançadas", expanded=False):
        adv1, adv2, adv3 = st.columns([1.2, 1.0, 1.0])

        with adv1:
            selected_model_names = st.multiselect(
                "Modelos meteorológicos",
                options=list(MODELS.keys()),
                default=DEFAULT_MODELS,
            )

        with adv2:
            risk_profile = st.selectbox(
                "Perfil matemático",
                options=list(RISK_MULTIPLIERS.keys()),
                index=1,
                help="Conservador aumenta a incerteza usada no semáforo.",
            )
            years_back = st.slider("Anos históricos", 8, 30, 20, 1)

        with adv3:
            resolution_risk = st.slider(
                "Risco de resolução/regras",
                0,
                10,
                3,
                help="Aumenta se o mercado usa uma estação específica, regra ambígua, hora estranha ou fonte difícil de replicar.",
            )
            window_days = st.slider("Janela histórica ± dias", 5, 21, 10, 1)

    calculate = st.button("Analisar mercado", type="primary", use_container_width=True)

if not calculate:
    st.info("Preenche cidade, dia, linha e preço YES. Depois clica em **Analisar mercado**.")
    st.stop()

if not city.strip():
    st.error("Escreve uma cidade.")
    st.stop()

if not selected_model_names:
    st.error("Seleciona pelo menos um modelo.")
    st.stop()

with st.spinner("A encontrar a cidade..."):
    places = geocode_city(city.strip())

if not places:
    st.error("Não encontrei essa cidade. Experimenta escrever também o país, por exemplo: 'Porto, Portugal'.")
    st.stop()

place = places[0]
lat = float(place["latitude"])
lon = float(place["longitude"])
place_name = format_place(place)

records = []
errors = []

with st.spinner("A comparar modelos meteorológicos oficiais/agregados..."):
    for model_name in selected_model_names:
        code = MODELS[model_name]
        try:
            records.append(fetch_model_forecast(lat, lon, target_day.isoformat(), model_name, code))
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

if not records:
    st.error("Nenhum modelo devolveu previsão utilizável para esta cidade/data.")
    with st.expander("Erros"):
        for error in errors:
            st.write("-", error)
    st.stop()

forecasts = pd.DataFrame(records)

with st.spinner("A carregar climatologia histórica da mesma época do ano..."):
    climatology = fetch_climatology(
        lat=lat,
        lon=lon,
        target_day_str=target_day.isoformat(),
        years_back=years_back,
        window_days=window_days,
    )

result = analyse(
    forecasts=forecasts,
    climatology=climatology,
    target_day=target_day,
    line_c=line_c,
    yes_price=yes_price,
    risk_profile=risk_profile,
    resolution_risk=resolution_risk,
)

light = result["light"]
side = result["side"]
color = LIGHT_COLORS[light]
bg = LIGHT_BG[light]
emoji = "🟢" if light == "VERDE" else "🟡" if light == "AMARELO" else "🔴"

st.caption(f"Local usado: **{place_name}** · Coordenadas: {lat:.4f}, {lon:.4f}")

st.markdown(
    f"""
    <div class="result-card" style="background:{bg}; border-color:{color}55;">
        <div class="result-title" style="color:{color};">{emoji} {light} — {side}</div>
        <div class="result-subtitle">{result['advice']}</div>
        <span class="small-pill">Score técnico: {result['technical_score']:.0f}/100</span>
        <span class="small-pill">Prob. conservadora {side}: {100 * result['chosen_prob']:.1f}%</span>
        <span class="small-pill">Edge: {result['chosen_edge']:+.3f}</span>
        <span class="small-pill">Margem: {result['margin_c']:+.1f} °C</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if result["hard_red_reasons"]:
    st.warning("Motivos principais: " + ", ".join(result["hard_red_reasons"]) + ".")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Tmax estimada", f"{result['final_mean']:.1f} °C")
m2.metric("Linha", f"{line_c:.1f} °C")
m3.metric("Prob. YES", f"{100 * result['p_cons_yes']:.1f}%")
m4.metric("Prob. NO", f"{100 * result['p_cons_no']:.1f}%")
m5.metric("Sigma", f"{result['sigma']:.2f} °C")

st.divider()

summary_tab, math_tab, charts_tab, stress_tab, models_tab, method_tab = st.tabs(
    ["Resumo", "Matemática", "Gráficos", "Stress tests", "Modelos", "Metodologia"]
)

with summary_tab:
    left, right = st.columns([1.1, 1.0])

    with left:
        st.markdown("### Decisão")
        if light == "VERDE":
            st.success(
                f"O lado estatisticamente favorecido é **{side}**. O semáforo ficou verde porque probabilidade, edge, consenso e margem passam os filtros."
            )
        elif light == "AMARELO":
            st.warning(
                f"O lado matematicamente melhor é **{side}**, mas ainda não é setup limpo. Espera melhor preço ou nova previsão."
            )
        else:
            st.error(
                "O semáforo está vermelho: não há vantagem suficientemente robusta para justificar aposta de baixo risco."
            )

        st.markdown(
            f"""
            - **Preço YES informado:** {yes_price:.3f}
            - **Preço justo conservador YES:** {result['p_cons_yes']:.3f}
            - **Preço justo conservador NO:** {result['p_cons_no']:.3f}
            - **Edge YES:** {result['yes_edge']:+.3f}
            - **Edge NO:** {result['no_edge']:+.3f}
            """
        )

    with right:
        st.markdown("### Fatores do semáforo")
        st.plotly_chart(score_chart(result["scores"]), use_container_width=True)

with math_tab:
    st.markdown("### Análise matemática")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Média ponderada", f"{result['weighted_mean']:.2f} °C")
    a2.metric("Mediana ponderada", f"{result['weighted_median']:.2f} °C")
    a3.metric("Média aparada", f"{result['trimmed_mean']:.2f} °C")
    a4.metric("Centro ensemble", f"{result['ensemble_center']:.2f} °C")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Dispersão modelos", f"{result['model_spread']:.2f} °C")
    b2.metric("MAD", f"{result['mad']:.2f} °C")
    b3.metric("Erro base", f"{result['error_floor']:.2f} °C")
    b4.metric("Erro sazonal", f"{result['seasonal_error']:.2f} °C")

    st.markdown(
        f"""
        <div class="formula-box">
        <b>Estimativa final:</b><br>
        centro = 55% média ponderada + 30% mediana ponderada + 15% média aparada<br>
        previsão final = peso_previsão × centro + (1 - peso_previsão) × climatologia<br><br>
        <b>Incerteza:</b><br>
        sigma = sqrt(erro_base² + discordância_modelos² + erro_sazonal² + penalização_amostra²) × multiplicador_risco<br><br>
        <b>Probabilidade:</b><br>
        P(YES) = P(Tmax ≥ linha). O semáforo usa uma mistura Normal + Monte Carlo com desconto conservador.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("P(YES) Normal", f"{100 * result['p_normal_yes']:.1f}%")
    c2.metric("P(YES) Monte Carlo", f"{100 * result['p_mc_yes']:.1f}%")
    c3.metric("P(YES) conservadora", f"{100 * result['p_cons_yes']:.1f}%")

    st.markdown("### Intervalos")
    st.write(
        f"**80%:** {result['interval_80'][0]:.1f} °C a {result['interval_80'][1]:.1f} °C"
    )
    st.write(
        f"**90%:** {result['interval_90'][0]:.1f} °C a {result['interval_90'][1]:.1f} °C"
    )
    st.write(
        f"**95%:** {result['interval_95'][0]:.1f} °C a {result['interval_95'][1]:.1f} °C"
    )

    if result["clim_mean"] is not None:
        st.markdown("### Climatologia")
        st.write(
            f"Média histórica da época: **{result['clim_mean']:.1f} °C** · "
            f"desvio-padrão histórico: **{result['clim_std']:.1f} °C** · "
            f"percentil climático da previsão: **{100 * result['clim_percentile']:.0f}%**."
        )
        st.write(f"Peso dado aos modelos atuais: **{100 * result['forecast_weight']:.0f}%**.")

with charts_tab:
    col_a, col_b = st.columns([1.05, 1.0])
    with col_a:
        st.markdown("### Modelos vs linha")
        st.plotly_chart(source_chart(forecasts, result["final_mean"], line_c), use_container_width=True)
    with col_b:
        st.markdown("### Distribuição estimada")
        st.plotly_chart(
            probability_chart(result["final_mean"], result["sigma"], line_c, side),
            use_container_width=True,
        )

    st.markdown("### Simulação Monte Carlo")
    st.plotly_chart(simulation_chart(result["simulated"], line_c, result["final_mean"]), use_container_width=True)

with stress_tab:
    st.markdown("### Stress tests")
    st.write(
        "Aqui a app força erros sistemáticos na temperatura prevista. Se o edge desaparece com -0.5 °C ou +0.5 °C, o setup não é robusto."
    )

    stress_df = result["stress"].copy()
    display_stress = stress_df.copy()
    display_stress["Prob. lado escolhido"] = (100 * display_stress["Prob. lado escolhido"]).round(1).astype(str) + "%"
    display_stress["Edge estimado"] = display_stress["Edge estimado"].map(lambda x: f"{x:+.3f}")

    st.plotly_chart(stress_chart(stress_df), use_container_width=True)
    st.dataframe(display_stress, hide_index=True, use_container_width=True)

    loo = build_leave_one_out(forecasts, result, line_c, yes_price)
    if not loo.empty:
        st.markdown("### Sensibilidade leave-one-out")
        st.write("Remove um modelo de cada vez para ver se uma única fonte está a dominar a conclusão.")
        loo_display = loo.copy()
        loo_display["Tmax centro"] = loo_display["Tmax centro"].round(2)
        loo_display["Dispersão"] = loo_display["Dispersão"].round(2)
        loo_display["Prob. lado escolhido"] = (100 * loo_display["Prob. lado escolhido"]).round(1).astype(str) + "%"
        loo_display["Edge"] = loo_display["Edge"].map(lambda x: f"{x:+.3f}")
        st.dataframe(loo_display, hide_index=True, use_container_width=True)

with models_tab:
    st.markdown("### Previsões por modelo")
    table = forecasts.copy()
    table["tmax"] = table["tmax"].round(2)
    table["peso"] = table["peso"].round(2)
    st.dataframe(
        table.rename(columns={"modelo": "Modelo", "tmax": "Tmax °C", "peso": "Peso"})[["Modelo", "Tmax °C", "Peso"]],
        hide_index=True,
        use_container_width=True,
    )

    if errors:
        with st.expander("Modelos que falharam"):
            for error in errors:
                st.write("-", error)

with method_tab:
    st.markdown("### Como ler o semáforo")
    st.markdown(
        """
        **🟢 Verde** — só aparece quando há probabilidade conservadora alta, edge relevante, modelos relativamente alinhados,
        margem clara face à linha e risco de resolução baixo/moderado.

        **🟡 Amarelo** — existe algum edge, mas não é suficientemente robusto. Normalmente significa esperar melhor preço,
        reduzir stake ou aguardar nova atualização dos modelos.

        **🔴 Vermelho** — não aconselha aposta. Pode ser por edge pequeno, linha demasiado próxima da previsão,
        poucos modelos, muita dispersão ou risco de resolução.
        """
    )

    st.markdown("### Regras práticas usadas")
    st.markdown(
        """
        - O preço justo do **YES** é a probabilidade conservadora de `Tmax ≥ linha`.
        - O preço justo do **NO** é a probabilidade conservadora de `Tmax < linha`.
        - `edge = preço_justo - preço_de_mercado`.
        - O modelo desconta a probabilidade quando há poucos modelos, data distante, dispersão elevada ou risco de resolução.
        - O Monte Carlo adiciona ruído e caudas pesadas para evitar falsa precisão.
        """
    )

    st.warning(
        "Isto não é garantia de lucro nem recomendação financeira personalizada. Meteorologia tem erro local, e mercados podem resolver por fonte/estação diferente da previsão usada."
    )

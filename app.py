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

HEADERS = {"User-Agent": "weather-edge-pro-streamlit/4.0"}

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

MARKET_EXACT = "Temperatura específica"
MARKET_OVER = "Maior que / acima da linha"


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


def format_place_with_coords(place):
    name = format_place(place)
    lat = place.get("latitude")
    lon = place.get("longitude")
    elevation = place.get("elevation")

    coord_text = ""
    if lat is not None and lon is not None:
        coord_text = f" · {float(lat):.4f}, {float(lon):.4f}"

    elev_text = ""
    if elevation is not None:
        elev_text = f" · {float(elevation):.0f} m"

    return f"{name}{coord_text}{elev_text}"


def safe_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except Exception:
        return None


def parse_hourly_tmax(data, target_day):
    """
    Calcula a Tmax manualmente a partir da série horária.

    Isto é mais seguro para mercados/apostas do que pedir diretamente
    daily=temperature_2m_max, porque garante que todos os modelos são
    agregados com a mesma regra: máximo das horas locais do dia alvo.
    """
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    target = target_day.isoformat()

    if not times:
        return None, None

    temp_keys = [
        key for key, values in hourly.items()
        if key.startswith("temperature_2m") and isinstance(values, list)
    ]

    if not temp_keys:
        return None, None

    best_tmax = None
    best_time = None

    for key in temp_keys:
        values = hourly.get(key, [])
        for t, value in zip(times, values):
            if not str(t).startswith(target):
                continue
            temp = safe_float(value)
            if temp is None:
                continue
            if best_tmax is None or temp > best_tmax:
                best_tmax = temp
                best_time = str(t)

    return best_tmax, best_time


@st.cache_data(ttl=20 * 60, show_spinner=False)
def fetch_model_forecast(lat, lon, target_day_str, model_name, model_code):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "timezone": "auto",
        "temperature_unit": "celsius",
        "start_date": target_day_str,
        "end_date": target_day_str,
        "cell_selection": "land",
    }

    if model_code != "best_match":
        params["models"] = model_code

    data = get_json(OPEN_METEO_FORECAST_URL, params=params)
    tmax, max_hour = parse_hourly_tmax(data, date.fromisoformat(target_day_str))

    if tmax is None:
        raise RuntimeError("Sem temperatura horária disponível para esta data/modelo.")

    return {
        "modelo": model_name,
        "codigo": model_code,
        "tmax": tmax,
        "hora_max": max_hour or "",
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
# MATH HELPERS
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
    Não é validação oficial; é proteção contra excesso de confiança.
    """
    h = max(0, int(horizon_days))
    return 0.95 + 0.17 * h + (0.20 if h >= 5 else 0.0) + (0.25 if h >= 10 else 0.0)


def scale_0_100(x, low, high):
    if high == low:
        return 0.0
    return float(np.clip(100 * (x - low) / (high - low), 0, 100))


def prob_over_normal(mu, sigma, line_c):
    dist = NormalDist(mu=mu, sigma=max(0.01, sigma))
    return float(1.0 - dist.cdf(line_c))


def prob_exact_normal(mu, sigma, lower, upper):
    dist = NormalDist(mu=mu, sigma=max(0.01, sigma))
    return float(max(0.0, min(1.0, dist.cdf(upper) - dist.cdf(lower))))


def simulate_temperature_distribution(
    forecasts,
    clim_mean,
    forecast_weight,
    error_floor,
    seasonal_error,
    risk_multiplier,
    n_samples=50000,
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

    # Cauda pesada: erro de estação, microclima, regra de resolução ou atualização de modelo.
    tail_event = rng.random(n_samples) < 0.08
    tail_noise = rng.normal(
        loc=0.0,
        scale=max(0.8, 1.75 * error_floor) * risk_multiplier,
        size=n_samples,
    )

    return centers + normal_noise + tail_event * tail_noise


def build_core_forecast(forecasts, climatology, target_day, risk_profile):
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

    simulated = simulate_temperature_distribution(
        forecasts=forecasts,
        clim_mean=clim_mean,
        forecast_weight=forecast_weight,
        error_floor=error_floor,
        seasonal_error=seasonal_error,
        risk_multiplier=risk_multiplier,
    )

    normal_dist = NormalDist(mu=final_mean, sigma=sigma)

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
        "simulated": simulated,
        "interval_80": (normal_dist.inv_cdf(0.10), normal_dist.inv_cdf(0.90)),
        "interval_90": (normal_dist.inv_cdf(0.05), normal_dist.inv_cdf(0.95)),
        "interval_95": (normal_dist.inv_cdf(0.025), normal_dist.inv_cdf(0.975)),
    }


def build_exact_buckets_from_distribution(simulated, half_width, target_temp_c):
    step = 2 * half_width
    if step <= 0:
        step = 1.0

    decimals = 0 if step >= 0.999 else 1 if step >= 0.099 else 2
    span = max(6 * half_width, 6)
    min_center = target_temp_c - span
    max_center = target_temp_c + span

    centers = np.arange(min_center, max_center + step / 2, step)
    rows = []

    for center in centers:
        lower = center - half_width
        upper = center + half_width
        p = float(np.mean((simulated >= lower) & (simulated < upper)))
        rows.append(
            {
                "Temperatura": round(float(center), decimals),
                "Intervalo": f"[{lower:.2f}, {upper:.2f}) °C",
                "Probabilidade": p,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("Probabilidade", ascending=False).reset_index(drop=True)
    df["Ranking"] = np.arange(1, len(df) + 1)
    return df


def finish_decision(
    market_mode,
    core,
    p_normal_yes,
    p_mc_yes,
    p_cons_yes,
    p_cons_no,
    yes_price,
    resolution_risk,
    exact_meta=None,
    over_meta=None,
):
    n_models = exact_meta.get("n_models") if exact_meta else over_meta.get("n_models")
    model_spread = core["model_spread"]
    horizon = core["horizon"]
    sigma = core["sigma"]

    no_price = 1.0 - yes_price
    yes_edge = p_cons_yes - yes_price
    no_edge = p_cons_no - no_price

    if yes_edge >= no_edge:
        side = "YES"
        chosen_prob = p_cons_yes
        chosen_price = yes_price
        chosen_edge = yes_edge
    else:
        side = "NO"
        chosen_prob = p_cons_no
        chosen_price = no_price
        chosen_edge = no_edge

    if market_mode == MARKET_EXACT:
        target_rank = exact_meta["target_rank"]
        target_prob_ratio = exact_meta["target_prob_ratio"]
        distance_sigma = exact_meta["distance_sigma"]

        if side == "YES":
            prob_score = scale_0_100(chosen_prob, 0.06, 0.28)
            edge_score = scale_0_100(chosen_edge, 0.00, 0.10)
            rank_score = 100 if target_rank == 1 else 80 if target_rank == 2 else 60 if target_rank == 3 else 35 if target_rank and target_rank <= 5 else 10
            distance_score = 100 - scale_0_100(distance_sigma, 0.00, 1.75)
        else:
            prob_score = scale_0_100(chosen_prob, 0.70, 0.95)
            edge_score = scale_0_100(chosen_edge, 0.00, 0.10)
            rank_score = 100 if target_rank and target_rank >= 4 else 70 if target_rank == 3 else 45 if target_rank == 2 else 20
            distance_score = scale_0_100(distance_sigma, 0.15, 1.50)

        exactness_weight = 0.16
        hard_red_reasons = []
        if n_models < 3:
            hard_red_reasons.append("poucos modelos disponíveis")
        if chosen_edge <= 0.015:
            hard_red_reasons.append("edge insuficiente")
        if side == "YES" and target_rank is not None and target_rank > 4:
            hard_red_reasons.append("temperatura alvo longe dos buckets mais prováveis")
        if side == "YES" and p_cons_yes < max(0.035, yes_price * 0.85):
            hard_red_reasons.append("probabilidade do YES demasiado baixa para o preço")
        if side == "NO" and p_cons_no < 0.72:
            hard_red_reasons.append("NO sem probabilidade conservadora suficiente")

    else:
        margin_sigma = over_meta["margin_sigma"]
        prob_score = scale_0_100(chosen_prob, 0.55, 0.90)
        edge_score = scale_0_100(chosen_edge, 0.00, 0.12)
        rank_score = scale_0_100(abs(margin_sigma), 0.20, 1.40)
        distance_score = rank_score
        target_prob_ratio = None

        exactness_weight = 0.00
        hard_red_reasons = []
        if n_models < 3:
            hard_red_reasons.append("poucos modelos disponíveis")
        if chosen_edge <= 0.015:
            hard_red_reasons.append("edge insuficiente")
        if chosen_prob < 0.58:
            hard_red_reasons.append("probabilidade conservadora baixa")
        if abs(margin_sigma) < 0.25:
            hard_red_reasons.append("linha demasiado perto da estimativa")

    consensus_score = 100 - scale_0_100(model_spread, 0.60, 3.00)
    horizon_score = 100 - scale_0_100(horizon, 2, 16)
    data_score = scale_0_100(n_models, 3, 7)
    resolution_score = 100 - 10 * resolution_risk

    if market_mode == MARKET_EXACT:
        scores = {
            "Edge": edge_score,
            "Probabilidade": prob_score,
            "Ranking do alvo": rank_score,
            "Distância ao centro": distance_score,
            "Consenso modelos": consensus_score,
            "Horizonte": horizon_score,
            "Quantidade de dados": data_score,
            "Risco de resolução": resolution_score,
        }
        technical_score = (
            0.24 * edge_score
            + 0.18 * prob_score
            + exactness_weight * rank_score
            + 0.13 * distance_score
            + 0.12 * consensus_score
            + 0.07 * horizon_score
            + 0.06 * data_score
            + 0.04 * resolution_score
        )
    else:
        scores = {
            "Probabilidade": prob_score,
            "Edge": edge_score,
            "Margem vs linha": rank_score,
            "Consenso modelos": consensus_score,
            "Horizonte": horizon_score,
            "Quantidade de dados": data_score,
            "Risco de resolução": resolution_score,
        }
        technical_score = (
            0.24 * prob_score
            + 0.24 * edge_score
            + 0.17 * rank_score
            + 0.14 * consensus_score
            + 0.09 * horizon_score
            + 0.07 * data_score
            + 0.05 * resolution_score
        )

    technical_score = float(np.clip(technical_score, 0, 100))

    if hard_red_reasons:
        light = "VERMELHO"
    elif market_mode == MARKET_EXACT and side == "YES" and (
        chosen_edge >= 0.055
        and p_cons_yes >= 0.075
        and exact_meta["target_rank"] is not None
        and exact_meta["target_rank"] <= 3
        and target_prob_ratio >= 0.70
        and model_spread <= 1.80
        and technical_score >= 70
        and resolution_risk <= 5
    ):
        light = "VERDE"
    elif market_mode == MARKET_EXACT and side == "NO" and (
        chosen_edge >= 0.045
        and p_cons_no >= 0.84
        and technical_score >= 68
        and resolution_risk <= 6
    ):
        light = "VERDE"
    elif market_mode == MARKET_OVER and (
        chosen_prob >= 0.82
        and chosen_edge >= 0.07
        and abs(over_meta["margin_sigma"]) >= 0.75
        and model_spread <= 1.70
        and sigma <= 3.90
        and n_models >= 5
        and technical_score >= 74
        and resolution_risk <= 5
    ):
        light = "VERDE"
    elif chosen_edge >= 0.025 and technical_score >= 48:
        light = "AMARELO"
    else:
        light = "VERMELHO"

    if market_mode == MARKET_EXACT:
        if light == "VERDE" and side == "YES":
            advice = "A temperatura específica parece subvalorizada pelo mercado. Mesmo assim, é uma aposta de bucket exato, nunca de risco baixo absoluto."
        elif light == "VERDE" and side == "NO":
            advice = "O mercado parece pagar demasiado pelo YES; matematicamente o lado favorecido é NO."
        elif light == "AMARELO":
            advice = "Existe algum edge, mas a robustez ainda não chega para sinal verde. Esperaria melhor preço ou nova atualização dos modelos."
        else:
            advice = "Não aconselho apostar: para temperatura específica, a margem estatística não compensa o risco estimado."
    else:
        if light == "VERDE":
            advice = "Setup favorável no modo maior que/acima da linha. Ainda assim, confirma regra, fonte de resolução, liquidez e spread."
        elif light == "AMARELO":
            advice = "Existe algum edge, mas não é setup limpo. Consideraria esperar melhor preço ou nova atualização dos modelos."
        else:
            advice = "Não aconselho apostar: a margem estatística não compensa o risco estimado."

    return {
        "p_normal_yes": p_normal_yes,
        "p_mc_yes": p_mc_yes,
        "p_blend_yes": 0.5 * p_normal_yes + 0.5 * p_mc_yes,
        "p_cons_yes": p_cons_yes,
        "p_cons_no": p_cons_no,
        "yes_edge": yes_edge,
        "no_edge": no_edge,
        "side": side,
        "chosen_prob": chosen_prob,
        "chosen_price": chosen_price,
        "chosen_edge": chosen_edge,
        "technical_score": technical_score,
        "light": light,
        "advice": advice,
        "hard_red_reasons": hard_red_reasons,
        "scores": scores,
    }


def analyse_exact_temperature(
    forecasts,
    climatology,
    target_day,
    target_temp_c,
    half_width_c,
    yes_price,
    risk_profile,
    resolution_risk,
):
    core = build_core_forecast(forecasts, climatology, target_day, risk_profile)
    n_models = len(forecasts)

    lower = target_temp_c - half_width_c
    upper = target_temp_c + half_width_c

    p_normal_yes = prob_exact_normal(core["final_mean"], core["sigma"], lower, upper)
    p_mc_yes = float(np.mean((core["simulated"] >= lower) & (core["simulated"] < upper)))
    p_blend_yes = 0.45 * p_normal_yes + 0.55 * p_mc_yes

    uncertainty_haircut = 0.006 + 0.0025 * core["horizon"] + 0.006 * max(0, 5 - n_models)
    uncertainty_haircut += 0.010 if core["model_spread"] > 1.5 else 0.0
    uncertainty_haircut += 0.015 if core["model_spread"] > 2.2 else 0.0
    uncertainty_haircut += 0.004 * resolution_risk

    p_cons_yes = float(np.clip(p_blend_yes - uncertainty_haircut, 0.0, 1.0))
    p_cons_no = float(np.clip(1.0 - p_blend_yes - uncertainty_haircut, 0.0, 1.0))

    buckets = build_exact_buckets_from_distribution(core["simulated"], half_width_c, target_temp_c)
    same_temp = buckets[np.isclose(buckets["Temperatura"].astype(float), target_temp_c, atol=max(0.001, half_width_c / 10))]
    target_rank = int(same_temp.iloc[0]["Ranking"]) if not same_temp.empty else None
    top_bucket_prob = float(buckets.iloc[0]["Probabilidade"]) if not buckets.empty else 0.0
    target_prob_ratio = p_mc_yes / top_bucket_prob if top_bucket_prob > 0 else 0.0

    distance_c = abs(core["final_mean"] - target_temp_c)
    distance_sigma = distance_c / core["sigma"]

    exact_meta = {
        "n_models": n_models,
        "target_rank": target_rank,
        "top_bucket_prob": top_bucket_prob,
        "target_prob_ratio": target_prob_ratio,
        "distance_c": distance_c,
        "distance_sigma": distance_sigma,
    }

    decision = finish_decision(
        market_mode=MARKET_EXACT,
        core=core,
        p_normal_yes=p_normal_yes,
        p_mc_yes=p_mc_yes,
        p_cons_yes=p_cons_yes,
        p_cons_no=p_cons_no,
        yes_price=yes_price,
        resolution_risk=resolution_risk,
        exact_meta=exact_meta,
    )

    stress_rows = []
    for shock in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        p_yes_shock = prob_exact_normal(core["final_mean"] + shock, core["sigma"], lower, upper)
        p_side = p_yes_shock if decision["side"] == "YES" else 1.0 - p_yes_shock
        price = yes_price if decision["side"] == "YES" else 1.0 - yes_price
        stress_rows.append(
            {
                "Cenário": f"Erro sistemático {shock:+.1f} °C",
                "Prob. YES alvo": p_yes_shock,
                "Prob. lado escolhido": p_side,
                "Edge estimado": p_side - price,
            }
        )

    return {
        **core,
        **decision,
        **exact_meta,
        "market_mode": MARKET_EXACT,
        "target_temp_c": target_temp_c,
        "half_width_c": half_width_c,
        "lower": lower,
        "upper": upper,
        "buckets": buckets,
        "stress": pd.DataFrame(stress_rows),
    }


def analyse_over_temperature(
    forecasts,
    climatology,
    target_day,
    line_c,
    yes_price,
    risk_profile,
    resolution_risk,
):
    core = build_core_forecast(forecasts, climatology, target_day, risk_profile)
    n_models = len(forecasts)

    p_normal_yes = prob_over_normal(core["final_mean"], core["sigma"], line_c)
    p_mc_yes = float(np.mean(core["simulated"] >= line_c))
    p_blend_yes = 0.55 * p_normal_yes + 0.45 * p_mc_yes

    uncertainty_haircut = 0.015 + 0.006 * core["horizon"] + 0.015 * max(0, 5 - n_models)
    uncertainty_haircut += 0.020 if core["model_spread"] > 1.5 else 0.0
    uncertainty_haircut += 0.025 if core["model_spread"] > 2.2 else 0.0
    uncertainty_haircut += 0.010 * resolution_risk

    p_cons_yes = float(np.clip(p_blend_yes - uncertainty_haircut, 0.0, 1.0))
    p_cons_no = float(np.clip(1.0 - p_blend_yes - uncertainty_haircut, 0.0, 1.0))

    margin_c_yes = core["final_mean"] - line_c
    over_meta = {
        "n_models": n_models,
        "line_c": line_c,
        "margin_c_yes": margin_c_yes,
        "margin_sigma": abs(margin_c_yes) / core["sigma"],
    }

    decision = finish_decision(
        market_mode=MARKET_OVER,
        core=core,
        p_normal_yes=p_normal_yes,
        p_mc_yes=p_mc_yes,
        p_cons_yes=p_cons_yes,
        p_cons_no=p_cons_no,
        yes_price=yes_price,
        resolution_risk=resolution_risk,
        over_meta=over_meta,
    )

    stress_rows = []
    for shock in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        p_yes_shock = prob_over_normal(core["final_mean"] + shock, core["sigma"], line_c)
        p_side = p_yes_shock if decision["side"] == "YES" else 1.0 - p_yes_shock
        price = yes_price if decision["side"] == "YES" else 1.0 - yes_price
        stress_rows.append(
            {
                "Cenário": f"Erro sistemático {shock:+.1f} °C",
                "Prob. YES alvo": p_yes_shock,
                "Prob. lado escolhido": p_side,
                "Edge estimado": p_side - price,
            }
        )

    return {
        **core,
        **decision,
        **over_meta,
        "market_mode": MARKET_OVER,
        "stress": pd.DataFrame(stress_rows),
    }


def build_leave_one_out(forecasts, base_result, yes_price):
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

        if base_result["market_mode"] == MARKET_EXACT:
            p_yes = prob_exact_normal(center, base_result["sigma"], base_result["lower"], base_result["upper"])
        else:
            p_yes = prob_over_normal(center, base_result["sigma"], base_result["line_c"])

        side_prob = p_yes if base_result["side"] == "YES" else 1.0 - p_yes
        price = yes_price if base_result["side"] == "YES" else 1.0 - yes_price

        rows.append(
            {
                "Sem o modelo": forecasts.loc[idx, "modelo"],
                "Centro": center,
                "Dispersão": spread,
                "Prob. YES alvo": p_yes,
                "Prob. lado escolhido": side_prob,
                "Edge": side_prob - price,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# CHARTS
# ============================================================

def source_chart(forecasts, result):
    df = forecasts.sort_values("tmax")
    fig = px.bar(
        df,
        x="tmax",
        y="modelo",
        orientation="h",
        text=df["tmax"].map(lambda x: f"{x:.1f}°C"),
        labels={"tmax": "Tmax prevista", "modelo": "Modelo"},
    )
    fig.add_vline(x=result["final_mean"], line_dash="dash", annotation_text="estimativa final")

    if result["market_mode"] == MARKET_EXACT:
        fig.add_vrect(x0=result["lower"], x1=result["upper"], opacity=0.18, line_width=0, annotation_text=f"alvo {result['target_temp_c']:g}°C")
    else:
        fig.add_vline(x=result["line_c"], line_dash="dot", annotation_text="linha")

    fig.update_layout(height=max(380, 48 * len(df)), margin=dict(l=10, r=10, t=25, b=10), showlegend=False)
    return fig


def probability_chart(result):
    mu = result["final_mean"]
    sigma = result["sigma"]
    dist = NormalDist(mu=mu, sigma=sigma)
    xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 450)
    ys = np.array([dist.pdf(float(x)) for x in xs])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Distribuição"))
    fig.add_vline(x=mu, line_dash="dash", annotation_text="estimativa")

    if result["market_mode"] == MARKET_EXACT:
        lower = result["lower"]
        upper = result["upper"]
        fig.add_vrect(x0=lower, x1=upper, opacity=0.22, line_width=0, annotation_text="bucket YES")
        fill_x = xs[(xs >= lower) & (xs < upper)]
    else:
        line_c = result["line_c"]
        fig.add_vline(x=line_c, line_dash="dot", annotation_text="linha")
        fill_x = xs[xs >= line_c]

    fill_y = np.array([dist.pdf(float(x)) for x in fill_x])
    fig.add_trace(
        go.Scatter(x=fill_x, y=fill_y, fill="tozeroy", mode="none", name="Zona YES")
    )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Temperatura máxima diária (°C)",
        yaxis_title="Densidade",
        showlegend=False,
    )
    return fig


def simulation_chart(result):
    fig = px.histogram(
        x=result["simulated"],
        nbins=70,
        labels={"x": "Tmax simulada (°C)", "y": "Frequência"},
    )
    fig.add_vline(x=result["final_mean"], line_dash="dash", annotation_text="estimativa")

    if result["market_mode"] == MARKET_EXACT:
        fig.add_vrect(x0=result["lower"], x1=result["upper"], opacity=0.22, line_width=0, annotation_text="bucket alvo")
    else:
        fig.add_vline(x=result["line_c"], line_dash="dot", annotation_text="linha")

    fig.update_layout(height=360, margin=dict(l=10, r=10, t=25, b=10), showlegend=False)
    return fig


def score_chart(scores):
    df = pd.DataFrame({"Fator": list(scores.keys()), "Score": [round(v, 1) for v in scores.values()]}).sort_values("Score")
    fig = px.bar(df, x="Score", y="Fator", orientation="h", range_x=[0, 100])
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=25, b=10), showlegend=False)
    return fig


def bucket_chart(buckets, target_temp_c):
    df = buckets.head(12).sort_values("Temperatura")
    df["Probabilidade %"] = 100 * df["Probabilidade"]
    fig = px.bar(
        df,
        x="Temperatura",
        y="Probabilidade %",
        text=df["Probabilidade %"].map(lambda x: f"{x:.1f}%"),
        labels={"Temperatura": "Temperatura específica", "Probabilidade %": "Probabilidade"},
    )
    fig.add_vline(x=target_temp_c, line_dash="dot", annotation_text="alvo")
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
            Semáforo para mercados de temperatura máxima: modo temperatura específica ou modo maior que/acima da linha.
            Permite escolher cidade, aeroporto, estação ou coordenadas manuais do ponto de resolução.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

places_preview = []
selected_place_index = 0
manual_lat = None
manual_lon = None
manual_place_name = ""

with st.container(border=True):
    row0a, row0b = st.columns([1.25, 1.0])
    with row0a:
        market_mode = st.selectbox(
            "Tipo de mercado",
            options=[MARKET_EXACT, MARKET_OVER],
            index=0,
            help="Escolhe 'Temperatura específica' para mercados tipo 'vai ser 25°C'. Escolhe 'Maior que' para mercados tipo 'vai ser acima de 25°C'.",
        )
    with row0b:
        yes_price = st.number_input(
            "Preço YES",
            min_value=0.01,
            max_value=0.99,
            value=0.15 if market_mode == MARKET_EXACT else 0.50,
            step=0.01,
        )

    st.markdown("#### 1) Ponto de medição")
    location_mode = st.radio(
        "Como queres escolher o ponto?",
        options=["Pesquisar e escolher resultado", "Coordenadas manuais"],
        horizontal=True,
        help="Usa coordenadas manuais quando o mercado resolve por aeroporto, estação específica, ICAO/IATA ou fonte oficial concreta.",
    )

    if location_mode == "Pesquisar e escolher resultado":
        city = st.text_input(
            "Cidade, aeroporto ou estação",
            value="Lisboa Aeroporto",
            placeholder="Ex.: Lisboa Aeroporto, Heathrow, JFK, Porto Airport, Zurich Airport",
        )

        if city.strip():
            try:
                places_preview = geocode_city(city.strip())
            except Exception as exc:
                places_preview = []
                st.warning(f"Não consegui pesquisar ainda: {exc}")

            if places_preview:
                selected_place_index = st.selectbox(
                    "Escolhe o resultado exato",
                    options=list(range(len(places_preview))),
                    format_func=lambda i: format_place_with_coords(places_preview[i]),
                    index=0,
                    help="Confirma que o ponto corresponde ao aeroporto/estação usado nas regras do mercado.",
                )
            else:
                st.caption("Sem resultados ainda. Tenta escrever também o país ou o nome do aeroporto.")
        else:
            st.caption("Escreve uma cidade, aeroporto ou estação para aparecerem resultados.")
    else:
        city = ""
        manual_place_name = st.text_input(
            "Nome do ponto",
            value="Aeroporto / estação manual",
            help="Só serve para aparecer no relatório; a previsão usa as coordenadas abaixo.",
        )
        coord1, coord2 = st.columns(2)
        with coord1:
            manual_lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=38.7742,
                step=0.0001,
                format="%.6f",
                help="Exemplo: Lisboa Airport ≈ 38.7742",
            )
        with coord2:
            manual_lon = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-9.1342,
                step=0.0001,
                format="%.6f",
                help="Exemplo: Lisboa Airport ≈ -9.1342",
            )
        st.caption("Dica: usa as coordenadas oficiais da estação/aeroporto indicado nas regras do mercado.")

    st.markdown("#### 2) Mercado")

    if market_mode == MARKET_EXACT:
        top1, top2, top3 = st.columns([1.0, 0.95, 0.95])
    else:
        top1, top2 = st.columns([1.0, 1.0])

    with top1:
        target_day = st.date_input(
            "Dia",
            value=date.today() + timedelta(days=3),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=16),
        )

    if market_mode == MARKET_EXACT:
        with top2:
            target_temp_c = st.number_input(
                "Temperatura alvo",
                value=25.0,
                step=0.5,
                format="%.1f",
                help="Ex.: se o mercado é 'Tmax será 25°C', escreve 25.",
            )
        with top3:
            half_width_c = st.number_input(
                "Bucket ± °C",
                min_value=0.01,
                max_value=2.00,
                value=0.50,
                step=0.05,
                format="%.2f",
                help="Para temperatura inteira arredondada: usa 0.50. Para 1 decimal: usa 0.05.",
            )
        st.caption(
            f"Interpretação atual: YES ganha se a Tmax ficar entre **{target_temp_c - half_width_c:.2f}°C** e **{target_temp_c + half_width_c:.2f}°C**."
        )
    else:
        with top2:
            line_c = st.number_input(
                "Linha: Tmax maior que X °C",
                value=25.0,
                step=0.5,
                format="%.1f",
                help="Para variáveis contínuas, 'maior que 25°C' e 'maior ou igual a 25°C' são praticamente iguais. Confirma a regra do mercado.",
            )
        st.caption(
            f"Interpretação atual: YES ganha se a Tmax for **maior que / acima de {line_c:.2f}°C**."
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
                4 if market_mode == MARKET_EXACT else 3,
                help="Aumenta se o mercado usa estação específica, fonte diferente, arredondamento ambíguo ou horário estranho.",
            )
            window_days = st.slider("Janela histórica ± dias", 5, 21, 10, 1)

    calculate = st.button("Analisar mercado", type="primary", use_container_width=True)

if not calculate:
    st.info("Escolhe o tipo de mercado, preenche cidade, dia, linha/alvo e preço YES. Depois clica em **Analisar mercado**.")
    st.stop()

if not selected_model_names:
    st.error("Seleciona pelo menos um modelo.")
    st.stop()

if location_mode == "Pesquisar e escolher resultado":
    if not city.strip():
        st.error("Escreve uma cidade, aeroporto ou estação.")
        st.stop()

    if not places_preview:
        with st.spinner("A encontrar o ponto de medição..."):
            places_preview = geocode_city(city.strip())

    if not places_preview:
        st.error("Não encontrei esse ponto. Tenta escrever também o país ou usa coordenadas manuais.")
        st.stop()

    selected_place_index = min(int(selected_place_index), len(places_preview) - 1)
    place = places_preview[selected_place_index]
    lat = float(place["latitude"])
    lon = float(place["longitude"])
    place_name = format_place_with_coords(place)
else:
    lat = float(manual_lat)
    lon = float(manual_lon)
    place_name = manual_place_name.strip() or "Coordenadas manuais"
    place_name = f"{place_name} · {lat:.4f}, {lon:.4f}"

records = []
errors = []

with st.spinner("A comparar modelos meteorológicos..."):
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

if market_mode == MARKET_EXACT:
    result = analyse_exact_temperature(
        forecasts=forecasts,
        climatology=climatology,
        target_day=target_day,
        target_temp_c=target_temp_c,
        half_width_c=half_width_c,
        yes_price=yes_price,
        risk_profile=risk_profile,
        resolution_risk=resolution_risk,
    )
else:
    result = analyse_over_temperature(
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

if market_mode == MARKET_EXACT:
    rank_text = "sem ranking" if result["target_rank"] is None else f"#{result['target_rank']} bucket mais provável"
    market_text = f"Tmax será {result['target_temp_c']:g}°C"
    target_pill = f"Alvo: {rank_text}"
else:
    market_text = f"Tmax > {result['line_c']:.1f}°C"
    target_pill = f"Margem: {result['margin_c_yes']:+.1f}°C vs linha"

st.caption(f"Local usado: **{place_name}** · Coordenadas: {lat:.4f}, {lon:.4f}")

st.markdown(
    f"""
    <div class="result-card" style="background:{bg}; border-color:{color}55;">
        <div class="result-title" style="color:{color};">{emoji} {light} — {side}</div>
        <div class="result-subtitle">{result['advice']}</div>
        <span class="small-pill">Mercado: {market_text}</span>
        <span class="small-pill">Score técnico: {result['technical_score']:.0f}/100</span>
        <span class="small-pill">Prob. conservadora YES: {100 * result['p_cons_yes']:.1f}%</span>
        <span class="small-pill">Prob. conservadora {side}: {100 * result['chosen_prob']:.1f}%</span>
        <span class="small-pill">Edge {side}: {result['chosen_edge']:+.3f}</span>
        <span class="small-pill">{target_pill}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if result["hard_red_reasons"]:
    st.warning("Motivos principais: " + ", ".join(result["hard_red_reasons"]) + ".")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Tmax estimada", f"{result['final_mean']:.1f} °C")
if market_mode == MARKET_EXACT:
    m2.metric("Bucket YES", f"{result['lower']:.1f}–{result['upper']:.1f} °C")
else:
    m2.metric("Linha YES", f"> {result['line_c']:.1f} °C")
m3.metric("Prob. YES", f"{100 * result['p_cons_yes']:.1f}%")
m4.metric("Edge YES", f"{result['yes_edge']:+.3f}")
m5.metric("Sigma", f"{result['sigma']:.2f} °C")

st.divider()

if market_mode == MARKET_EXACT:
    tabs = st.tabs(["Resumo", "Matemática", "Gráficos", "Temperaturas prováveis", "Stress tests", "Modelos", "Metodologia"])
    summary_tab, math_tab, charts_tab, buckets_tab, stress_tab, models_tab, method_tab = tabs
else:
    tabs = st.tabs(["Resumo", "Matemática", "Gráficos", "Stress tests", "Modelos", "Metodologia"])
    summary_tab, math_tab, charts_tab, stress_tab, models_tab, method_tab = tabs
    buckets_tab = None

with summary_tab:
    left, right = st.columns([1.1, 1.0])

    with left:
        st.markdown("### Decisão")
        if light == "VERDE":
            st.success(f"O lado estatisticamente favorecido é **{side}**.")
        elif light == "AMARELO":
            st.warning(f"O lado matematicamente melhor é **{side}**, mas ainda não é setup limpo.")
        else:
            st.error("O semáforo está vermelho: não há vantagem suficientemente robusta.")

        if market_mode == MARKET_EXACT:
            st.markdown(
                f"""
                - **Mercado analisado:** Tmax será **{result['target_temp_c']:g}°C**
                - **Bucket usado:** `{result['lower']:.2f}°C ≤ Tmax < {result['upper']:.2f}°C`
                - **Ranking do alvo:** {rank_text}
                - **Preço YES informado:** {yes_price:.3f}
                - **Preço justo conservador YES:** {result['p_cons_yes']:.3f}
                - **Preço justo conservador NO:** {result['p_cons_no']:.3f}
                - **Edge YES:** {result['yes_edge']:+.3f}
                - **Edge NO:** {result['no_edge']:+.3f}
                """
            )
            st.info("Num mercado de temperatura específica, o YES normalmente tem probabilidade baixa. O que interessa é se está barato face ao preço justo estimado.")
        else:
            st.markdown(
                f"""
                - **Mercado analisado:** Tmax será **maior que {result['line_c']:.1f}°C**
                - **Preço YES informado:** {yes_price:.3f}
                - **Preço justo conservador YES:** {result['p_cons_yes']:.3f}
                - **Preço justo conservador NO:** {result['p_cons_no']:.3f}
                - **Edge YES:** {result['yes_edge']:+.3f}
                - **Edge NO:** {result['no_edge']:+.3f}
                - **Margem da estimativa vs linha:** {result['margin_c_yes']:+.2f}°C
                """
            )
            st.info("No modo maior que, a app mede a probabilidade de a Tmax ultrapassar a linha, depois desconta incerteza e risco de resolução.")

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

    if market_mode == MARKET_EXACT:
        formula_text = f"""
        <b>Modo temperatura específica:</b><br>
        Este mercado não é P(Tmax ≥ X). É P(a ≤ Tmax < b).<br><br>
        <b>Bucket analisado:</b><br>
        a = temperatura_alvo − bucket = {result['lower']:.2f} °C<br>
        b = temperatura_alvo + bucket = {result['upper']:.2f} °C<br><br>
        <b>Probabilidade YES:</b><br>
        P(YES) = P({result['lower']:.2f} ≤ Tmax < {result['upper']:.2f})
        """
    else:
        formula_text = f"""
        <b>Modo maior que / acima da linha:</b><br>
        Este mercado é P(Tmax > linha). Em previsão contínua, > e ≥ têm diferença praticamente nula.<br><br>
        <b>Linha analisada:</b><br>
        linha = {result['line_c']:.2f} °C<br><br>
        <b>Probabilidade YES:</b><br>
        P(YES) = P(Tmax > {result['line_c']:.2f})
        """

    st.markdown(
        f"""
        <div class="formula-box">
        {formula_text}<br><br>
        <b>Estimativa final:</b><br>
        centro = 55% média ponderada + 30% mediana ponderada + 15% média aparada<br>
        previsão final = peso_previsão × centro + (1 - peso_previsão) × climatologia<br><br>
        <b>Incerteza:</b><br>
        sigma = sqrt(erro_base² + discordância_modelos² + erro_sazonal² + penalização_amostra²) × multiplicador_risco<br><br>
        <b>Probabilidade usada no semáforo:</b><br>
        mistura de distribuição Normal + Monte Carlo, depois com desconto conservador por horizonte, dispersão,
        poucos modelos e risco de resolução.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("P(YES) Normal", f"{100 * result['p_normal_yes']:.1f}%")
    c2.metric("P(YES) Monte Carlo", f"{100 * result['p_mc_yes']:.1f}%")
    c3.metric("P(YES) conservadora", f"{100 * result['p_cons_yes']:.1f}%")

    st.markdown("### Intervalos da Tmax contínua")
    st.write(f"**80%:** {result['interval_80'][0]:.1f} °C a {result['interval_80'][1]:.1f} °C")
    st.write(f"**90%:** {result['interval_90'][0]:.1f} °C a {result['interval_90'][1]:.1f} °C")
    st.write(f"**95%:** {result['interval_95'][0]:.1f} °C a {result['interval_95'][1]:.1f} °C")

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
        st.markdown("### Modelos vs mercado")
        st.plotly_chart(source_chart(forecasts, result), use_container_width=True)
    with col_b:
        st.markdown("### Distribuição estimada")
        st.plotly_chart(probability_chart(result), use_container_width=True)

    st.markdown("### Simulação Monte Carlo")
    st.plotly_chart(simulation_chart(result), use_container_width=True)

if market_mode == MARKET_EXACT and buckets_tab is not None:
    with buckets_tab:
        st.markdown("### Temperaturas específicas mais prováveis")
        st.write("Esta tabela estima a probabilidade de cada temperatura específica/bucket. Se o alvo não está entre os primeiros, o YES tende a ser frágil.")

        st.plotly_chart(bucket_chart(result["buckets"], result["target_temp_c"]), use_container_width=True)

        bucket_table = result["buckets"].head(20).copy()
        bucket_table["Probabilidade"] = (100 * bucket_table["Probabilidade"]).round(2).astype(str) + "%"
        st.dataframe(bucket_table[["Ranking", "Temperatura", "Intervalo", "Probabilidade"]], hide_index=True, use_container_width=True)

with stress_tab:
    st.markdown("### Stress tests")
    if market_mode == MARKET_EXACT:
        st.write("Aqui a app força erros sistemáticos na temperatura prevista. Para YES exato, é mau sinal se o edge desaparece com ±0.5°C.")
    else:
        st.write("Aqui a app força erros sistemáticos na temperatura prevista. Para maior que, é mau sinal se o edge desaparece com ±0.5°C ou ±1.0°C.")

    stress_df = result["stress"].copy()
    display_stress = stress_df.copy()
    display_stress["Prob. YES alvo"] = (100 * display_stress["Prob. YES alvo"]).round(1).astype(str) + "%"
    display_stress["Prob. lado escolhido"] = (100 * display_stress["Prob. lado escolhido"]).round(1).astype(str) + "%"
    display_stress["Edge estimado"] = display_stress["Edge estimado"].map(lambda x: f"{x:+.3f}")

    st.plotly_chart(stress_chart(stress_df), use_container_width=True)
    st.dataframe(display_stress, hide_index=True, use_container_width=True)

    loo = build_leave_one_out(forecasts, result, yes_price)
    if not loo.empty:
        st.markdown("### Sensibilidade leave-one-out")
        st.write("Remove um modelo de cada vez para ver se uma única fonte está a dominar a conclusão.")
        loo_display = loo.copy()
        loo_display["Centro"] = loo_display["Centro"].round(2)
        loo_display["Dispersão"] = loo_display["Dispersão"].round(2)
        loo_display["Prob. YES alvo"] = (100 * loo_display["Prob. YES alvo"]).round(1).astype(str) + "%"
        loo_display["Prob. lado escolhido"] = (100 * loo_display["Prob. lado escolhido"]).round(1).astype(str) + "%"
        loo_display["Edge"] = loo_display["Edge"].map(lambda x: f"{x:+.3f}")
        st.dataframe(loo_display, hide_index=True, use_container_width=True)

with models_tab:
    st.markdown("### Previsões por modelo")
    table = forecasts.copy()
    table["tmax"] = table["tmax"].round(2)
    table["peso"] = table["peso"].round(2)
    st.dataframe(
        table.rename(columns={"modelo": "Modelo", "tmax": "Tmax °C", "hora_max": "Hora local do máximo", "peso": "Peso"})[["Modelo", "Tmax °C", "Hora local do máximo", "Peso"]],
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
        **🟢 Verde** — só aparece quando o preço parece errado o suficiente face à probabilidade conservadora,
        com boa robustez, modelos relativamente alinhados e risco de resolução aceitável.

        **🟡 Amarelo** — existe algum edge, mas não é suficientemente robusto. Normalmente significa esperar melhor preço,
        reduzir stake ou aguardar nova atualização dos modelos.

        **🔴 Vermelho** — não aconselha aposta. Pode ser por edge pequeno, alvo/linha frágil,
        poucos modelos, muita dispersão ou risco de resolução.
        """
    )

    st.markdown("### Modos de mercado")
    st.markdown(
        """
        **Temperatura específica**  
        Usa um bucket, por exemplo 25°C com bucket ±0.50°C significa `24.50°C ≤ Tmax < 25.50°C`.
        Este é o modo certo para mercados tipo “a máxima será 25°C”.

        **Maior que / acima da linha**  
        Usa `P(Tmax > linha)`. Este é o modo certo para mercados tipo “a máxima será acima de 25°C”.
        Em variáveis contínuas, `>` e `≥` têm diferença prática quase nula, mas deves confirmar a regra exata do mercado.
        """
    )

    st.markdown("### Regras práticas usadas")
    st.markdown(
        """
        - O preço justo do **YES** é a probabilidade conservadora do evento definido pelo modo escolhido.
        - O preço justo do **NO** é a probabilidade conservadora do evento contrário.
        - `edge = preço_justo - preço_de_mercado`.
        - A app desconta probabilidade quando há poucos modelos, data distante, dispersão elevada ou risco de resolução.
        - A Tmax por modelo é calculada a partir da temperatura horária do próprio modelo, usando o máximo das horas locais do dia alvo.
        - O Monte Carlo adiciona ruído e caudas pesadas para evitar falsa precisão.
        """
    )

    st.warning(
        "Isto não é garantia de lucro nem recomendação financeira personalizada. Confirma sempre fonte oficial, estação usada, arredondamento, horário, unidade e regra exata de resolução do mercado."
    )


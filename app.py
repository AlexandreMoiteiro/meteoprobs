from datetime import date, timedelta
from statistics import NormalDist
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Weather Edge",
    page_icon="🌡️",
    layout="centered",
)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

MODELS = {
    "Open-Meteo Best Match": "best_match",
    "ECMWF": "ecmwf_ifs025",
    "GFS": "gfs_seamless",
    "ICON": "icon_seamless",
    "Météo-France": "meteofrance_seamless",
    "UK Met Office": "ukmo_seamless",
}

MODEL_WEIGHTS = {
    "best_match": 1.20,
    "ecmwf_ifs025": 1.25,
    "gfs_seamless": 1.00,
    "icon_seamless": 1.05,
    "meteofrance_seamless": 1.05,
    "ukmo_seamless": 1.05,
}


# =========================
# API HELPERS
# =========================

@st.cache_data(ttl=30 * 60)
def get_json(url, params=None):
    r = requests.get(
        url,
        params=params,
        headers={"User-Agent": "weather-edge-simple/1.0"},
        timeout=25,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=24 * 60 * 60)
def geocode_city(city):
    data = get_json(
        OPEN_METEO_GEOCODING_URL,
        {
            "name": city,
            "count": 5,
            "language": "pt",
            "format": "json",
        },
    )
    return data.get("results", []) or []


def place_name(place):
    parts = [place.get("name"), place.get("admin1"), place.get("country")]
    return ", ".join([p for p in parts if p])


def extract_tmax(data, target_day):
    daily = data.get("daily", {})
    days = daily.get("time", [])
    temps = daily.get("temperature_2m_max", [])

    if target_day.isoformat() not in days:
        return None

    idx = days.index(target_day.isoformat())
    if idx >= len(temps):
        return None

    value = temps[idx]
    if value is None:
        return None

    return float(value)


@st.cache_data(ttl=20 * 60)
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

    data = get_json(OPEN_METEO_FORECAST_URL, params)
    tmax = extract_tmax(data, date.fromisoformat(target_day_str))

    if tmax is None:
        raise ValueError("sem temperatura máxima")

    return {
        "modelo": model_name,
        "codigo": model_code,
        "tmax": tmax,
        "peso": MODEL_WEIGHTS.get(model_code, 1.0),
    }


def circular_day_distance(a, b, days=366):
    d = abs(a - b)
    return min(d, days - d)


@st.cache_data(ttl=12 * 60 * 60)
def fetch_climate_reference(lat, lon, target_day_str):
    """
    Referência histórica simples:
    últimos 15 anos, dias próximos da mesma altura do ano.
    Usada só para evitar excesso de confiança.
    """
    target = date.fromisoformat(target_day_str)
    end_year = min(date.today().year - 1, target.year - 1)
    start_year = max(1940, end_year - 14)

    if end_year < start_year:
        return pd.DataFrame(columns=["date", "tmax"])

    data = get_json(
        OPEN_METEO_ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
            "daily": "temperature_2m_max",
            "timezone": "auto",
            "temperature_unit": "celsius",
        },
    )

    daily = data.get("daily", {})
    days = daily.get("time", [])
    temps = daily.get("temperature_2m_max", [])

    target_doy = target.timetuple().tm_yday
    rows = []

    for d_str, temp in zip(days, temps):
        if temp is None:
            continue

        d = date.fromisoformat(d_str)
        if circular_day_distance(d.timetuple().tm_yday, target_doy) <= 10:
            rows.append({"date": d, "tmax": float(temp)})

    return pd.DataFrame(rows)


# =========================
# MATH
# =========================

def weighted_stats(values, weights):
    values = np.array(values, dtype=float)
    weights = np.array(weights, dtype=float)

    mean = float(np.average(values, weights=weights))

    if len(values) <= 1:
        return mean, 0.0

    variance = float(np.average((values - mean) ** 2, weights=weights))
    spread = math.sqrt(max(variance, 0.0))

    return mean, spread


def forecast_error_floor(days_ahead):
    """
    Piso conservador de erro para Tmax diária.
    Quanto mais longe a data, maior a incerteza.
    """
    return 1.15 + 0.20 * max(days_ahead, 0)


def run_analysis(df, climate_df, target_day, line_temp, conservative=True):
    mean, spread = weighted_stats(df["tmax"], df["peso"])

    days_ahead = max(0, (target_day - date.today()).days)
    error_floor = forecast_error_floor(days_ahead)

    climate_mean = None
    climate_std = None

    if not climate_df.empty:
        climate_mean = float(climate_df["tmax"].mean())
        climate_std = float(climate_df["tmax"].std(ddof=1))

        # Quanto mais longe a data, mais puxamos ligeiramente para a climatologia.
        forecast_weight = 1 / (1 + (days_ahead / 18) ** 2)
        final_mean = forecast_weight * mean + (1 - forecast_weight) * climate_mean
    else:
        final_mean = mean
        forecast_weight = 1.0

    seasonal_uncertainty = 0
    if climate_std is not None and not math.isnan(climate_std):
        seasonal_uncertainty = 0.15 * climate_std

    sigma = math.sqrt(spread**2 + error_floor**2 + seasonal_uncertainty**2)

    if conservative:
        sigma *= 1.30

    sigma = max(sigma, 0.80)

    dist = NormalDist(final_mean, sigma)
    p_yes = 1 - dist.cdf(line_temp)
    p_no = 1 - p_yes

    return {
        "mean_models": mean,
        "spread": spread,
        "final_mean": final_mean,
        "sigma": sigma,
        "p_yes": p_yes,
        "p_no": p_no,
        "ci_low": dist.inv_cdf(0.05),
        "ci_high": dist.inv_cdf(0.95),
        "days_ahead": days_ahead,
        "climate_mean": climate_mean,
        "forecast_weight": forecast_weight,
    }


def recommendation(p_yes, yes_price, spread, sigma):
    fair_yes = p_yes
    fair_no = 1 - p_yes
    no_price = 1 - yes_price

    edge_yes = fair_yes - yes_price
    edge_no = fair_no - no_price

    if sigma >= 4 or spread >= 2.5:
        return "Evitar", "Incerteza demasiado alta entre os modelos.", edge_yes, edge_no

    if fair_yes >= 0.85 and edge_yes >= 0.05:
        return "YES", "Há margem estatística favorável para YES.", edge_yes, edge_no

    if fair_no >= 0.85 and edge_no >= 0.05:
        return "NO", "Há margem estatística favorável para NO.", edge_yes, edge_no

    return "Evitar", "A probabilidade ou o preço não dão margem suficiente.", edge_yes, edge_no


# =========================
# CHARTS
# =========================

def model_chart(df, final_mean, line_temp):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["modelo"],
            y=df["tmax"],
            text=[f"{x:.1f}°" for x in df["tmax"]],
            textposition="outside",
            name="Previsão",
        )
    )

    fig.add_hline(
        y=final_mean,
        line_dash="dash",
        annotation_text="estimativa final",
    )

    fig.add_hline(
        y=line_temp,
        line_dash="dot",
        annotation_text="linha do mercado",
    )

    fig.update_layout(
        title="Modelos meteorológicos",
        yaxis_title="Temperatura máxima prevista, °C",
        xaxis_title="",
        height=380,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
    )

    return fig


def probability_chart(final_mean, sigma, line_temp):
    dist = NormalDist(final_mean, sigma)
    xs = np.linspace(final_mean - 4 * sigma, final_mean + 4 * sigma, 300)
    ys = [dist.pdf(float(x)) for x in xs]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="Probabilidade",
        )
    )

    fig.add_vline(
        x=final_mean,
        line_dash="dash",
        annotation_text="estimativa",
    )

    fig.add_vline(
        x=line_temp,
        line_dash="dot",
        annotation_text="linha",
    )

    fig.update_layout(
        title="Incerteza estimada",
        xaxis_title="Temperatura máxima, °C",
        yaxis_title="Densidade",
        height=340,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
    )

    return fig


# =========================
# UI
# =========================

st.title("🌡️ Weather Edge")
st.write("Previsão simples da temperatura máxima para comparar com uma linha de mercado.")

with st.form("inputs"):
    city = st.text_input("Cidade", value="Lisboa")

    col_a, col_b = st.columns(2)

    with col_a:
        target_day = st.date_input(
            "Dia",
            value=date.today() + timedelta(days=3),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=16),
        )

    with col_b:
        line_temp = st.number_input(
            "Linha do mercado, °C",
            value=25.0,
            step=0.5,
        )

    col_c, col_d = st.columns(2)

    with col_c:
        yes_price = st.number_input(
            "Preço YES",
            min_value=0.01,
            max_value=0.99,
            value=0.50,
            step=0.01,
        )

    with col_d:
        conservative = st.toggle(
            "Modo conservador",
            value=True,
            help="Aumenta a incerteza para reduzir excesso de confiança.",
        )

    submitted = st.form_submit_button("Analisar", type="primary", use_container_width=True)

if not submitted:
    st.stop()

if not city.strip():
    st.error("Escreve uma cidade.")
    st.stop()

try:
    places = geocode_city(city.strip())
except Exception as e:
    st.error(f"Erro ao procurar cidade: {e}")
    st.stop()

if not places:
    st.error("Não encontrei essa cidade. Tenta escrever também o país.")
    st.stop()

place = places[0]
lat = float(place["latitude"])
lon = float(place["longitude"])

st.caption(f"Local usado: {place_name(place)}")

rows = []
errors = []

with st.spinner("A calcular previsões..."):
    for model_name, model_code in MODELS.items():
        try:
            rows.append(
                fetch_model_forecast(
                    lat,
                    lon,
                    target_day.isoformat(),
                    model_name,
                    model_code,
                )
            )
        except Exception as e:
            errors.append(f"{model_name}: {e}")

    try:
        climate_df = fetch_climate_reference(lat, lon, target_day.isoformat())
    except Exception:
        climate_df = pd.DataFrame(columns=["date", "tmax"])

if len(rows) < 2:
    st.error("Não consegui obter modelos suficientes para uma análise útil.")
    if errors:
        with st.expander("Erros"):
            for err in errors:
                st.write(err)
    st.stop()

df = pd.DataFrame(rows)
result = run_analysis(df, climate_df, target_day, line_temp, conservative)
pick, reason, edge_yes, edge_no = recommendation(
    result["p_yes"],
    yes_price,
    result["spread"],
    result["sigma"],
)

# =========================
# TOP RESULT
# =========================

st.divider()

if pick == "YES":
    st.success(f"Conclusão: **YES** — {reason}")
elif pick == "NO":
    st.success(f"Conclusão: **NO** — {reason}")
else:
    st.warning(f"Conclusão: **EVITAR** — {reason}")

k1, k2, k3 = st.columns(3)

k1.metric("Tmax estimada", f"{result['final_mean']:.1f} °C")
k2.metric("Prob. YES", f"{100 * result['p_yes']:.1f}%")
k3.metric("Intervalo 90%", f"{result['ci_low']:.1f}–{result['ci_high']:.1f} °C")

e1, e2, e3 = st.columns(3)

e1.metric("Preço justo YES", f"{result['p_yes']:.3f}")
e2.metric("Edge YES", f"{edge_yes:+.3f}")
e3.metric("Edge NO", f"{edge_no:+.3f}")

st.write(
    f"A linha é **{line_temp:.1f} °C**. "
    f"A app estima **{100 * result['p_yes']:.1f}%** de probabilidade de a temperatura máxima ser igual ou superior à linha."
)

st.plotly_chart(model_chart(df, result["final_mean"], line_temp), use_container_width=True)
st.plotly_chart(probability_chart(result["final_mean"], result["sigma"], line_temp), use_container_width=True)

with st.expander("Ver detalhes técnicos"):
    details = df.copy()
    details["tmax"] = details["tmax"].round(2)
    st.dataframe(details[["modelo", "tmax"]], hide_index=True, use_container_width=True)

    st.write(f"Média dos modelos: **{result['mean_models']:.1f} °C**")
    st.write(f"Dispersão entre modelos: **{result['spread']:.2f} °C**")
    st.write(f"Incerteza final: **{result['sigma']:.2f} °C**")
    st.write(f"Dias até ao evento: **{result['days_ahead']}**")

    if result["climate_mean"] is not None:
        st.write(f"Média histórica da época do ano: **{result['climate_mean']:.1f} °C**")

    if errors:
        st.write("Modelos que falharam:")
        for err in errors:
            st.write(f"- {err}")

st.caption(
    "Isto é uma estimativa estatística, não uma garantia. Confirma sempre as regras exatas do mercado antes de apostar."
)

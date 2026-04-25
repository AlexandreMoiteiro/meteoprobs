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
    page_title="Weather Edge",
    page_icon="🌡️",
    layout="centered",
)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HEADERS = {
    "User-Agent": "weather-edge-streamlit/1.0"
}

MODELS = {
    "Open-Meteo Best Match": "best_match",
    "ECMWF IFS": "ecmwf_ifs025",
    "ECMWF AIFS": "ecmwf_aifs025_single",
    "NOAA GFS": "gfs_seamless",
    "DWD ICON": "icon_seamless",
    "Météo-France": "meteofrance_seamless",
    "UK Met Office": "ukmo_seamless",
    "GEM Canadá": "gem_seamless",
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
}

DEFAULT_MODELS = [
    "Open-Meteo Best Match",
    "ECMWF IFS",
    "NOAA GFS",
    "DWD ICON",
    "Météo-France",
    "UK Met Office",
]


# ============================================================
# DATA FUNCTIONS
# ============================================================

@st.cache_data(ttl=30 * 60, show_spinner=False)
def get_json(url: str, params: dict | None = None) -> dict:
    response = requests.get(url, params=params, headers=HEADERS, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Erro HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def geocode_city(city: str) -> list[dict]:
    data = get_json(
        OPEN_METEO_GEOCODING_URL,
        params={
            "name": city,
            "count": 5,
            "language": "pt",
            "format": "json",
        },
    )
    return data.get("results", []) or []


def format_place(place: dict) -> str:
    parts = [place.get("name"), place.get("admin1"), place.get("country")]
    return ", ".join([str(p) for p in parts if p])


def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def parse_tmax(data: dict, target_day: date) -> float | None:
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
def fetch_model_forecast(
    lat: float,
    lon: float,
    target_day_str: str,
    model_name: str,
    model_code: str,
) -> dict:
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
        raise RuntimeError("Sem previsão disponível para este modelo.")

    return {
        "modelo": model_name,
        "codigo": model_code,
        "tmax": tmax,
        "peso": MODEL_WEIGHTS.get(model_code, 1.0),
    }


def doy_distance(a: int, b: int, days: int = 366) -> int:
    diff = abs(a - b)
    return min(diff, days - diff)


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def fetch_climatology(
    lat: float,
    lon: float,
    target_day_str: str,
    years_back: int = 15,
    window_days: int = 10,
) -> pd.DataFrame:
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

def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mean = float(np.average(values, weights=weights))

    if len(values) <= 1:
        return mean, 0.0

    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, math.sqrt(max(variance, 0.0))


def base_forecast_error(horizon_days: int) -> float:
    """
    Piso conservador de erro para Tmax diária.
    Quanto mais longe a data, maior a incerteza.
    """
    h = max(0, horizon_days)
    return 1.05 + 0.18 * h + (0.25 if h >= 5 else 0.0) + (0.25 if h >= 10 else 0.0)


def analyse(forecasts: pd.DataFrame, climatology: pd.DataFrame, target_day: date, line_c: float) -> dict:
    values = forecasts["tmax"].to_numpy(dtype=float)
    weights = forecasts["peso"].to_numpy(dtype=float)

    model_mean, model_spread = weighted_mean_std(values, weights)
    horizon = max(0, (target_day - date.today()).days)

    clim_mean = None
    clim_std = None

    if not climatology.empty:
        clim_mean = float(climatology["tmax"].mean())
        clim_std = float(climatology["tmax"].std(ddof=1))

    # Puxa ligeiramente para a média histórica quando a previsão está longe.
    if clim_mean is not None and not math.isnan(clim_mean):
        forecast_weight = 1.0 / (1.0 + (horizon / 18.0) ** 2)
        final_mean = forecast_weight * model_mean + (1.0 - forecast_weight) * clim_mean
    else:
        forecast_weight = 1.0
        final_mean = model_mean

    error_floor = base_forecast_error(horizon)
    seasonal_error = 0.15 * clim_std if clim_std is not None and not math.isnan(clim_std) else 0.0

    sigma = math.sqrt(model_spread**2 + error_floor**2 + seasonal_error**2)
    sigma = max(0.85, sigma * 1.25)  # multiplicador conservador fixo

    dist = NormalDist(mu=final_mean, sigma=sigma)

    p_yes = 1.0 - dist.cdf(line_c)
    p_no = 1.0 - p_yes

    return {
        "media_modelos": model_mean,
        "dispersao_modelos": model_spread,
        "media_final": final_mean,
        "sigma": sigma,
        "p_yes": p_yes,
        "p_no": p_no,
        "intervalo_80": (dist.inv_cdf(0.10), dist.inv_cdf(0.90)),
        "intervalo_90": (dist.inv_cdf(0.05), dist.inv_cdf(0.95)),
        "clim_mean": clim_mean,
        "clim_std": clim_std,
        "forecast_weight": forecast_weight,
        "horizon": horizon,
    }


def recommendation(p_yes: float, yes_price: float, dispersion: float, sigma: float) -> tuple[str, str]:
    fair_yes = p_yes
    fair_no = 1.0 - p_yes
    edge_yes = fair_yes - yes_price
    edge_no = fair_no - (1.0 - yes_price)

    if dispersion > 2.0 or sigma > 4.0:
        return "Evitar", "Modelos demasiado dispersos ou incerteza alta."

    if p_yes >= 0.85 and edge_yes >= 0.05:
        return "YES forte", "Probabilidade alta e preço abaixo do valor justo estimado."

    if p_yes <= 0.15 and edge_no >= 0.05:
        return "NO forte", "Probabilidade baixa de ultrapassar a linha e preço do NO parece interessante."

    if edge_yes > 0.03:
        return "YES moderado", "Existe edge, mas não é suficientemente grande para chamar baixo risco."

    if edge_no > 0.03:
        return "NO moderado", "Existe edge no NO, mas não é suficientemente grande para chamar baixo risco."

    return "Sem edge claro", "O preço do mercado parece próximo da probabilidade estimada."


# ============================================================
# CHARTS
# ============================================================

def source_chart(forecasts: pd.DataFrame, final_mean: float) -> go.Figure:
    df = forecasts.sort_values("tmax")

    fig = px.bar(
        df,
        x="tmax",
        y="modelo",
        orientation="h",
        text=df["tmax"].map(lambda x: f"{x:.1f}°C"),
        labels={"tmax": "Temperatura máxima prevista", "modelo": "Modelo"},
    )

    fig.add_vline(
        x=final_mean,
        line_dash="dash",
        annotation_text="estimativa final",
    )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
    )

    return fig


def probability_chart(mu: float, sigma: float, line_c: float) -> go.Figure:
    dist = NormalDist(mu=mu, sigma=sigma)
    xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ys = [dist.pdf(float(x)) for x in xs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="probabilidade"))
    fig.add_vline(x=mu, line_dash="dash", annotation_text="estimativa")
    fig.add_vline(x=line_c, line_dash="dot", annotation_text="linha")

    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Temperatura máxima diária (°C)",
        yaxis_title="Densidade",
        showlegend=False,
    )

    return fig


# ============================================================
# UI
# ============================================================

st.title("🌡️ Weather Edge")
st.write("Previsão simples da temperatura máxima para mercados tipo Polymarket.")

with st.container(border=True):
    city_col, date_col = st.columns([1.5, 1])

    with city_col:
        city = st.text_input("Cidade", value="Lisboa")

    with date_col:
        target_day = st.date_input(
            "Dia",
            value=date.today() + timedelta(days=3),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=16),
        )

    line_col, price_col = st.columns(2)

    with line_col:
        line_c = st.number_input(
            "Linha do mercado: Tmax ≥ X °C",
            value=25.0,
            step=0.5,
            format="%.1f",
        )

    with price_col:
        yes_price = st.number_input(
            "Preço YES",
            min_value=0.01,
            max_value=0.99,
            value=0.50,
            step=0.01,
        )

    with st.expander("Modelos usados"):
        selected_model_names = st.multiselect(
            "Modelos meteorológicos",
            options=list(MODELS.keys()),
            default=DEFAULT_MODELS,
        )

    calculate = st.button("Calcular", type="primary", use_container_width=True)

if not calculate:
    st.info("Preenche a cidade, a data e a linha do mercado. Depois clica em Calcular.")
    st.stop()

if not city.strip():
    st.error("Escreve uma cidade.")
    st.stop()

if not selected_model_names:
    st.error("Seleciona pelo menos um modelo.")
    st.stop()

with st.spinner("A procurar localização..."):
    places = geocode_city(city.strip())

if not places:
    st.error("Não encontrei essa cidade. Experimenta escrever também o país, por exemplo: Porto, Portugal.")
    st.stop()

place = places[0]
lat = float(place["latitude"])
lon = float(place["longitude"])
place_name = format_place(place)

st.caption(f"Local usado: {place_name}")

records = []
errors = []

with st.spinner("A comparar modelos meteorológicos..."):
    for model_name in selected_model_names:
        model_code = MODELS[model_name]
        try:
            record = fetch_model_forecast(
                lat=lat,
                lon=lon,
                target_day_str=target_day.isoformat(),
                model_name=model_name,
                model_code=model_code,
            )
            records.append(record)
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

if not records:
    st.error("Nenhum modelo devolveu previsão utilizável para esta cidade/data.")
    with st.expander("Erros"):
        for error in errors:
            st.write("-", error)
    st.stop()

forecasts = pd.DataFrame(records)

with st.spinner("A comparar com histórico da mesma época do ano..."):
    climatology = fetch_climatology(lat, lon, target_day.isoformat())

result = analyse(forecasts, climatology, target_day, line_c)

signal, signal_reason = recommendation(
    p_yes=result["p_yes"],
    yes_price=yes_price,
    dispersion=result["dispersao_modelos"],
    sigma=result["sigma"],
)

fair_yes = result["p_yes"]
fair_no = result["p_no"]
edge_yes = fair_yes - yes_price
edge_no = fair_no - (1.0 - yes_price)

# ============================================================
# RESULTS
# ============================================================

st.divider()
st.subheader("Resultado")

main1, main2, main3 = st.columns(3)
main1.metric("Tmax estimada", f"{result['media_final']:.1f} °C")
main2.metric("Probabilidade YES", f"{100 * result['p_yes']:.1f}%")
main3.metric("Sinal", signal)

if signal in ["YES forte", "NO forte"]:
    st.success(signal_reason)
elif signal == "Evitar":
    st.warning(signal_reason)
else:
    st.info(signal_reason)

with st.container(border=True):
    st.markdown("#### Leitura de mercado")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Preço justo YES", f"{fair_yes:.3f}")
    c2.metric("Edge YES", f"{edge_yes:+.3f}")
    c3.metric("Preço justo NO", f"{fair_no:.3f}")
    c4.metric("Edge NO", f"{edge_no:+.3f}")

    st.write(
        f"A linha é **Tmax ≥ {line_c:.1f} °C**. "
        f"O modelo estima **{100 * fair_yes:.1f}%** para YES e **{100 * fair_no:.1f}%** para NO."
    )

st.markdown("#### Gráficos")
st.plotly_chart(source_chart(forecasts, result["media_final"]), use_container_width=True)
st.plotly_chart(probability_chart(result["media_final"], result["sigma"], line_c), use_container_width=True)

with st.expander("Detalhes técnicos"):
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Média dos modelos", f"{result['media_modelos']:.1f} °C")
    d2.metric("Dispersão", f"{result['dispersao_modelos']:.2f} °C")
    d3.metric("Sigma final", f"{result['sigma']:.2f} °C")
    d4.metric("Horizonte", f"{result['horizon']} dias")

    st.write(
        f"Intervalo 80%: **{result['intervalo_80'][0]:.1f} °C a {result['intervalo_80'][1]:.1f} °C**"
    )
    st.write(
        f"Intervalo 90%: **{result['intervalo_90'][0]:.1f} °C a {result['intervalo_90'][1]:.1f} °C**"
    )

    if result["clim_mean"] is not None:
        st.write(
            f"Média histórica da época: **{result['clim_mean']:.1f} °C**. "
            f"Peso dado à previsão atual: **{100 * result['forecast_weight']:.0f}%**."
        )

    table = forecasts.copy()
    table["tmax"] = table["tmax"].round(2)
    table["peso"] = table["peso"].round(2)
    st.dataframe(table[["modelo", "tmax", "peso"]], hide_index=True, use_container_width=True)

    if errors:
        st.markdown("##### Fontes que falharam")
        for error in errors:
            st.write("-", error)

st.caption(
    "Isto é uma estimativa estatística, não garantia de lucro. Confirma sempre as regras de resolução do mercado, a estação usada, horário, liquidez e spread."
)

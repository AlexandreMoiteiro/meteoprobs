
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
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Weather Edge — Tmax Forecast",
    page_icon="🌡️",
    layout="wide",
)

APP_USER_AGENT = "weather-edge-streamlit/1.0"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
VISUAL_CROSSING_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
METNO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"

MODEL_OPTIONS = {
    "Open‑Meteo Best Match": "best_match",
    "ECMWF IFS 0.25°": "ecmwf_ifs025",
    "ECMWF AIFS 0.25°": "ecmwf_aifs025_single",
    "NOAA/NCEP GFS Seamless": "gfs_seamless",
    "DWD ICON Seamless": "icon_seamless",
    "Météo‑France Seamless": "meteofrance_seamless",
    "UK Met Office Seamless": "ukmo_seamless",
    "GEM Canada Seamless": "gem_seamless",
    "BOM ACCESS Global": "bom_access_global",
    "MET Norway Seamless": "metno_seamless",
    "MeteoSwiss ICON Seamless": "meteoswiss_icon_seamless",
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
    "metno_seamless": 1.00,
    "meteoswiss_icon_seamless": 1.00,
    "visual_crossing": 1.00,
    "metno_api": 0.95,
}

RISK_MULTIPLIERS = {
    "Normal": 1.00,
    "Conservador": 1.25,
    "Muito conservador": 1.50,
}


# ============================================================
# HELPERS
# ============================================================

def get_secret(name: str, default=None):
    """Lê st.secrets sem rebentar localmente quando não existe secrets.toml."""
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def safe_float(value):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def today_local() -> date:
    return date.today()


@st.cache_data(ttl=30 * 60, show_spinner=False)
def get_json(url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 25) -> dict:
    hdrs = headers or {"User-Agent": APP_USER_AGENT}
    response = requests.get(url, params=params, headers=hdrs, timeout=timeout)
    if response.status_code != 200:
        body = response.text[:500] if response.text else ""
        raise RuntimeError(f"HTTP {response.status_code}: {body}")
    return response.json()


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def geocode_city(city: str) -> list[dict]:
    data = get_json(
        OPEN_METEO_GEOCODING_URL,
        params={"name": city, "count": 10, "language": "pt", "format": "json"},
    )
    return data.get("results", []) or []


def format_place(item: dict) -> str:
    parts = [
        item.get("name"),
        item.get("admin1"),
        item.get("country"),
    ]
    text = ", ".join([str(x) for x in parts if x])
    lat = item.get("latitude")
    lon = item.get("longitude")
    return f"{text}  ({lat:.4f}, {lon:.4f})"


def parse_open_meteo_daily_tmax(data: dict, target_day: date) -> float | None:
    daily = data.get("daily", {})
    times = daily.get("time", [])
    if not times:
        return None

    target_str = target_day.isoformat()
    if target_str not in times:
        return None

    idx = times.index(target_str)

    # Com 1 modelo, a chave costuma ser temperature_2m_max.
    # Com vários modelos, a API pode devolver sufixos por modelo.
    candidate_keys = [
        key for key, value in daily.items()
        if key.startswith("temperature_2m_max") and isinstance(value, list)
    ]

    for key in candidate_keys:
        values = daily.get(key, [])
        if idx < len(values):
            value = safe_float(values[idx])
            if value is not None:
                return value
    return None


@st.cache_data(ttl=20 * 60, show_spinner=False)
def fetch_open_meteo_model(lat: float, lon: float, target_day_str: str, model_label: str, model_code: str) -> dict:
    target_day = date.fromisoformat(target_day_str)

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
    tmax = parse_open_meteo_daily_tmax(data, target_day)

    if tmax is None:
        raise RuntimeError("Sem temperature_2m_max para este modelo/data.")

    return {
        "provider": "Open‑Meteo",
        "source": model_label,
        "model_code": model_code,
        "tmax_c": tmax,
        "weight": MODEL_WEIGHTS.get(model_code, 1.0),
    }


@st.cache_data(ttl=20 * 60, show_spinner=False)
def fetch_visual_crossing(lat: float, lon: float, target_day_str: str, api_key: str) -> dict:
    location = f"{lat},{lon}"
    url = f"{VISUAL_CROSSING_URL}/{location}/{target_day_str}/{target_day_str}"

    data = get_json(
        url,
        params={
            "key": api_key,
            "unitGroup": "metric",
            "include": "days",
            "contentType": "json",
            "lang": "pt",
        },
    )

    days = data.get("days", [])
    if not days:
        raise RuntimeError("Visual Crossing não devolveu dados diários.")

    tmax = safe_float(days[0].get("tempmax"))
    if tmax is None:
        raise RuntimeError("Visual Crossing não devolveu tempmax.")

    return {
        "provider": "Visual Crossing",
        "source": "Visual Crossing Timeline",
        "model_code": "visual_crossing",
        "tmax_c": tmax,
        "weight": MODEL_WEIGHTS["visual_crossing"],
    }


@st.cache_data(ttl=20 * 60, show_spinner=False)
def fetch_metno(lat: float, lon: float, target_day_str: str, timezone_name: str, user_agent: str) -> dict:
    headers = {"User-Agent": user_agent}

    data = get_json(
        METNO_URL,
        params={"lat": round(lat, 4), "lon": round(lon, 4)},
        headers=headers,
    )

    tz = ZoneInfo(timezone_name or "UTC")
    values = []

    for item in data.get("properties", {}).get("timeseries", []):
        raw_time = item.get("time")
        if not raw_time:
            continue

        ts = datetime.fromisoformat(raw_time.replace("Z", "+00:00")).astimezone(tz)
        if ts.date().isoformat() != target_day_str:
            continue

        temp = (
            item.get("data", {})
            .get("instant", {})
            .get("details", {})
            .get("air_temperature")
        )
        value = safe_float(temp)
        if value is not None:
            values.append(value)

    if not values:
        raise RuntimeError("MET Norway não tem pontos horários para essa data.")

    return {
        "provider": "MET Norway",
        "source": "MET Norway Locationforecast",
        "model_code": "metno_api",
        "tmax_c": float(np.max(values)),
        "weight": MODEL_WEIGHTS["metno_api"],
    }


def circular_doy_distance(a: int, b: int, days: int = 366) -> int:
    diff = abs(a - b)
    return min(diff, days - diff)


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def fetch_climatology(lat: float, lon: float, target_day_str: str, years_back: int, window_days: int) -> pd.DataFrame:
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

    if not dates or not temps:
        return pd.DataFrame(columns=["date", "tmax_c"])

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


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan

    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    weights = np.where(weights <= 0, 1.0, weights)

    mean = float(np.average(values, weights=weights))

    if len(values) == 1:
        return mean, 0.0

    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, math.sqrt(max(variance, 0.0))


def forecast_rmse_floor(horizon_days: int) -> float:
    """
    Piso de erro empírico simples para Tmax diária.
    Não é uma validação oficial; serve para não confundir consenso aparente com certeza absoluta.
    """
    h = max(0, horizon_days)
    return 1.05 + 0.18 * h + (0.25 if h >= 5 else 0.0) + (0.25 if h >= 10 else 0.0)


def analyse_temperature(
    forecasts: pd.DataFrame,
    clim: pd.DataFrame,
    target_day: date,
    threshold_c: float,
    risk_profile: str,
) -> dict:
    values = forecasts["tmax_c"].to_numpy(dtype=float)
    weights = forecasts["weight"].to_numpy(dtype=float)

    ensemble_mean, model_spread = weighted_mean_std(values, weights)
    horizon_days = max(0, (target_day - today_local()).days)

    clim_mean = None
    clim_std = None
    if not clim.empty:
        clim_mean = float(clim["tmax_c"].mean())
        clim_std = float(clim["tmax_c"].std(ddof=1)) if len(clim) > 1 else None

    # Shrink leve para climatologia quando a data está mais longe.
    # Em dias próximos, a previsão domina. Em horizonte longo, a climatologia ganha algum peso.
    if clim_mean is not None:
        forecast_weight = 1.0 / (1.0 + (horizon_days / 18.0) ** 2)
        posterior_mean = forecast_weight * ensemble_mean + (1.0 - forecast_weight) * clim_mean
    else:
        forecast_weight = 1.0
        posterior_mean = ensemble_mean

    rmse_floor = forecast_rmse_floor(horizon_days)
    seasonal_floor = 0.15 * clim_std if clim_std is not None and not math.isnan(clim_std) else 0.0
    risk_multiplier = RISK_MULTIPLIERS.get(risk_profile, 1.0)

    sigma = math.sqrt(model_spread**2 + rmse_floor**2 + seasonal_floor**2)
    sigma = max(0.75, sigma * risk_multiplier)

    dist = NormalDist(mu=posterior_mean, sigma=sigma)
    p_over = 1.0 - dist.cdf(threshold_c)
    p_under = 1.0 - p_over

    ci80 = (dist.inv_cdf(0.10), dist.inv_cdf(0.90))
    ci90 = (dist.inv_cdf(0.05), dist.inv_cdf(0.95))
    ci95 = (dist.inv_cdf(0.025), dist.inv_cdf(0.975))

    consensus = "Alta"
    if model_spread >= 2.0:
        consensus = "Baixa"
    elif model_spread >= 1.0:
        consensus = "Média"

    if len(forecasts) < 3:
        consensus = "Baixa"

    confidence_score = 100 * min(
        1.0,
        max(
            0.0,
            0.20
            + 0.10 * min(len(forecasts), 5)
            + 0.25 * (1.0 - min(model_spread / 3.0, 1.0))
            + 0.20 * (1.0 - min(horizon_days / 16.0, 1.0))
            + 0.15 * (1.0 - min(sigma / 5.0, 1.0)),
        ),
    )

    return {
        "ensemble_mean": ensemble_mean,
        "model_spread": model_spread,
        "posterior_mean": posterior_mean,
        "sigma": sigma,
        "rmse_floor": rmse_floor,
        "seasonal_floor": seasonal_floor,
        "forecast_weight": forecast_weight,
        "clim_mean": clim_mean,
        "clim_std": clim_std,
        "p_over": p_over,
        "p_under": p_under,
        "ci80": ci80,
        "ci90": ci90,
        "ci95": ci95,
        "consensus": consensus,
        "confidence_score": confidence_score,
        "horizon_days": horizon_days,
    }


def probability_label(p: float) -> str:
    if p >= 0.90:
        return "Muito alta"
    if p >= 0.75:
        return "Alta"
    if p >= 0.60:
        return "Moderada"
    if p >= 0.40:
        return "Incerta"
    if p >= 0.25:
        return "Baixa"
    return "Muito baixa"


def build_distribution_chart(mu: float, sigma: float, threshold: float) -> go.Figure:
    dist = NormalDist(mu=mu, sigma=sigma)
    xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ys = np.array([dist.pdf(float(x)) for x in xs])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Distribuição estimada"))
    fig.add_vline(x=mu, line_dash="dash", annotation_text="estimativa central")
    fig.add_vline(x=threshold, line_dash="dot", annotation_text="linha do mercado")
    fig.update_layout(
        title="Distribuição probabilística da temperatura máxima",
        xaxis_title="Temperatura máxima diária (°C)",
        yaxis_title="Densidade",
        height=380,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def build_sources_chart(forecasts: pd.DataFrame, posterior_mean: float) -> go.Figure:
    df = forecasts.sort_values("tmax_c")
    fig = px.bar(
        df,
        x="tmax_c",
        y="source",
        orientation="h",
        hover_data=["provider", "weight"],
        labels={"tmax_c": "Tmax prevista (°C)", "source": "Fonte/modelo"},
        title="Comparação das previsões por modelo/fonte",
    )
    fig.add_vline(x=posterior_mean, line_dash="dash", annotation_text="estimativa final")
    fig.update_layout(height=max(360, 42 * len(df)), margin=dict(l=10, r=10, t=55, b=10))
    return fig


def build_climatology_chart(clim: pd.DataFrame, target_forecast: float) -> go.Figure:
    fig = px.histogram(
        clim,
        x="tmax_c",
        nbins=30,
        labels={"tmax_c": "Tmax histórica em janela sazonal (°C)"},
        title="Climatologia histórica para a mesma época do ano",
    )
    fig.add_vline(x=target_forecast, line_dash="dash", annotation_text="estimativa atual")
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=55, b=10))
    return fig


# ============================================================
# UI
# ============================================================

st.title("🌡️ Weather Edge — previsão de temperatura máxima")
st.caption(
    "Agrega vários modelos meteorológicos, estima incerteza e calcula a probabilidade de a temperatura máxima ultrapassar uma linha."
)

with st.expander("⚠️ Nota importante sobre apostas e risco", expanded=False):
    st.write(
        """
        Esta app é uma ferramenta estatística, não uma garantia de lucro. Mesmo uma probabilidade estimada de 90% ainda implica risco.
        O objetivo é reduzir decisões impulsivas: comparar fontes, quantificar incerteza e evitar mercados sem margem clara.
        Não automatiza compras/vendas no Polymarket.
        """
    )

with st.sidebar:
    st.header("Parâmetros")

    city = st.text_input("Cidade", value="Lisboa")
    min_day = today_local()
    max_day = min_day + timedelta(days=16)
    target_day = st.date_input(
        "Dia do mercado",
        value=min_day + timedelta(days=3),
        min_value=min_day,
        max_value=max_day,
        help="Open‑Meteo normalmente permite previsões até 16 dias. Algumas fontes opcionais têm horizontes menores.",
    )

    threshold_c = st.number_input(
        "Linha do mercado Polymarket: Tmax ≥ X °C",
        value=25.0,
        step=0.5,
        format="%.1f",
    )

    yes_price = st.number_input(
        "Preço YES no mercado, opcional",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        help="Se não quiseres analisar edge/preço, deixa em 0.50.",
    )

    risk_profile = st.selectbox(
        "Perfil de incerteza",
        options=list(RISK_MULTIPLIERS.keys()),
        index=1,
        help="Conservador aumenta a largura dos intervalos e evita excesso de confiança.",
    )

    st.subheader("Modelos Open‑Meteo")
    default_models = [
        "Open‑Meteo Best Match",
        "ECMWF IFS 0.25°",
        "NOAA/NCEP GFS Seamless",
        "DWD ICON Seamless",
        "Météo‑France Seamless",
        "UK Met Office Seamless",
    ]

    selected_labels = st.multiselect(
        "Seleciona modelos",
        options=list(MODEL_OPTIONS.keys()),
        default=default_models,
    )

    st.subheader("Fontes opcionais")
    use_visual_crossing = st.checkbox("Usar Visual Crossing se houver secret", value=False)
    use_metno = st.checkbox("Usar MET Norway Locationforecast se houver User‑Agent", value=False)

    vc_key_secret = get_secret("VISUAL_CROSSING_KEY", "")
    metno_user_agent_secret = get_secret("METNO_USER_AGENT", "")

    vc_key = st.text_input(
        "VISUAL_CROSSING_KEY",
        value=vc_key_secret,
        type="password",
        help="Opcional. Em produção, usa .streamlit/secrets.toml.",
    )

    metno_user_agent = st.text_input(
        "METNO_USER_AGENT",
        value=metno_user_agent_secret,
        help="Ex.: weather-edge/1.0 teuemail@dominio.com",
    )

    st.subheader("Climatologia")
    years_back = st.slider("Anos históricos", min_value=5, max_value=30, value=15, step=1)
    window_days = st.slider("Janela sazonal ± dias", min_value=3, max_value=21, value=10, step=1)

    run = st.button("Calcular", type="primary")


if not run:
    st.info("Escolhe a cidade, a data e a linha do mercado; depois clica em **Calcular**.")
    st.stop()


# ============================================================
# EXECUÇÃO
# ============================================================

if not city.strip():
    st.error("Escreve uma cidade.")
    st.stop()

try:
    matches = geocode_city(city.strip())
except Exception as exc:
    st.error(f"Erro no geocoding: {exc}")
    st.stop()

if not matches:
    st.error("Não encontrei essa cidade. Tenta escrever também o país, por exemplo: 'Porto, Portugal'.")
    st.stop()

place = st.selectbox(
    "Localização encontrada",
    options=matches,
    format_func=format_place,
)

lat = float(place["latitude"])
lon = float(place["longitude"])
timezone_name = place.get("timezone") or "UTC"

st.write(f"**Local usado:** {format_place(place)}")
st.write(f"**Timezone:** {timezone_name} | **Data alvo:** {target_day.isoformat()}")

warnings = []
records = []

progress = st.progress(0, text="A obter previsões...")

tasks = [(label, MODEL_OPTIONS[label]) for label in selected_labels]
total_tasks = len(tasks) + int(use_visual_crossing) + int(use_metno)

if total_tasks == 0:
    st.error("Seleciona pelo menos uma fonte/modelo.")
    st.stop()

completed = 0

for label, code in tasks:
    try:
        record = fetch_open_meteo_model(lat, lon, target_day.isoformat(), label, code)
        records.append(record)
    except Exception as exc:
        warnings.append(f"{label}: {exc}")
    completed += 1
    progress.progress(completed / total_tasks, text="A obter previsões...")

if use_visual_crossing:
    if vc_key.strip():
        try:
            records.append(fetch_visual_crossing(lat, lon, target_day.isoformat(), vc_key.strip()))
        except Exception as exc:
            warnings.append(f"Visual Crossing: {exc}")
    else:
        warnings.append("Visual Crossing ignorado: falta VISUAL_CROSSING_KEY.")
    completed += 1
    progress.progress(completed / total_tasks, text="A obter previsões...")

if use_metno:
    if metno_user_agent.strip() and "email" not in metno_user_agent.lower():
        try:
            records.append(fetch_metno(lat, lon, target_day.isoformat(), timezone_name, metno_user_agent.strip()))
        except Exception as exc:
            warnings.append(f"MET Norway: {exc}")
    else:
        warnings.append("MET Norway ignorado: falta User‑Agent válido com contacto.")
    completed += 1
    progress.progress(completed / total_tasks, text="A obter previsões...")

progress.empty()

if not records:
    st.error("Nenhuma fonte devolveu previsão utilizável. Experimenta menos modelos ou outra data.")
    if warnings:
        with st.expander("Detalhes dos erros"):
            for warning in warnings:
                st.write(f"- {warning}")
    st.stop()

forecasts = pd.DataFrame(records)

with st.spinner("A obter climatologia histórica..."):
    try:
        clim = fetch_climatology(lat, lon, target_day.isoformat(), years_back, window_days)
    except Exception as exc:
        clim = pd.DataFrame(columns=["date", "tmax_c"])
        warnings.append(f"Climatologia histórica: {exc}")

analysis = analyse_temperature(forecasts, clim, target_day, threshold_c, risk_profile)

# ============================================================
# RESULTADOS
# ============================================================

st.subheader("Resultado principal")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tmax estimada", f"{analysis['posterior_mean']:.1f} °C")
col2.metric("Prob. Tmax ≥ linha", f"{100 * analysis['p_over']:.1f}%")
col3.metric("Intervalo 90%", f"{analysis['ci90'][0]:.1f}–{analysis['ci90'][1]:.1f} °C")
col4.metric("Concordância", analysis["consensus"])

yes_fair = analysis["p_over"]
no_fair = analysis["p_under"]
edge_yes = yes_fair - yes_price
edge_no = no_fair - (1.0 - yes_price)

st.markdown("### Leitura para mercado binário")

mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("Preço justo YES", f"{yes_fair:.3f}")
mcol2.metric("Edge YES vs preço", f"{edge_yes:+.3f}")
mcol3.metric("Edge NO vs preço implícito", f"{edge_no:+.3f}")

p_label = probability_label(yes_fair)
st.write(
    f"Com a linha **Tmax ≥ {threshold_c:.1f} °C**, a probabilidade estimada do **YES** é "
    f"**{100 * yes_fair:.1f}%** ({p_label.lower()})."
)

if yes_fair >= 0.85 and edge_yes >= 0.05 and analysis["consensus"] != "Baixa":
    st.success(
        "Sinal estatístico forte para YES, segundo estes dados. Mesmo assim, confirma regras de resolução, liquidez, spread e hora/local da medição."
    )
elif no_fair >= 0.85 and edge_no >= 0.05 and analysis["consensus"] != "Baixa":
    st.success(
        "Sinal estatístico forte para NO, segundo estes dados. Mesmo assim, confirma regras de resolução, liquidez, spread e hora/local da medição."
    )
elif analysis["consensus"] == "Baixa" or analysis["sigma"] >= 3.5:
    st.warning(
        "Risco elevado: há dispersão relevante entre modelos ou incerteza ampla. Evita tratar isto como aposta de baixo risco."
    )
else:
    st.info(
        "Sinal moderado/indefinido. A margem estatística pode não compensar spread, taxas, liquidez ou erro de resolução do mercado."
    )

st.markdown("### Diagnóstico matemático")

dcol1, dcol2, dcol3, dcol4 = st.columns(4)
dcol1.metric("Média ponderada modelos", f"{analysis['ensemble_mean']:.1f} °C")
dcol2.metric("Dispersão entre modelos", f"{analysis['model_spread']:.2f} °C")
dcol3.metric("Sigma final", f"{analysis['sigma']:.2f} °C")
dcol4.metric("Score técnico", f"{analysis['confidence_score']:.0f}/100")

with st.expander("Como a conta é feita"):
    st.markdown(
        f"""
        1. Cada modelo/fonte dá uma previsão de **Tmax**.
        2. Calcula-se uma média ponderada dos modelos.
        3. Mede-se a dispersão entre fontes.
        4. Soma-se um piso de erro por horizonte temporal: **{analysis['rmse_floor']:.2f} °C**.
        5. Se houver climatologia, aplica-se uma correção leve para a média histórica da época:
           peso da previsão = **{100 * analysis['forecast_weight']:.1f}%**.
        6. Assume-se uma distribuição normal aproximada para obter probabilidades e intervalos.

        Isto é deliberadamente conservador: a incerteza final mistura discordância entre modelos,
        erro mínimo esperado e variabilidade histórica sazonal.
        """
    )

tab1, tab2, tab3 = st.tabs(["Gráficos", "Tabela de fontes", "Avisos"])

with tab1:
    st.plotly_chart(build_sources_chart(forecasts, analysis["posterior_mean"]), use_container_width=True)
    st.plotly_chart(build_distribution_chart(analysis["posterior_mean"], analysis["sigma"], threshold_c), use_container_width=True)

    if not clim.empty:
        st.plotly_chart(build_climatology_chart(clim, analysis["posterior_mean"]), use_container_width=True)
    else:
        st.info("Sem climatologia histórica disponível para este local/janela.")

with tab2:
    table = forecasts.copy()
    table["tmax_c"] = table["tmax_c"].round(2)
    table["weight"] = table["weight"].round(2)
    st.dataframe(
        table[["provider", "source", "model_code", "tmax_c", "weight"]],
        use_container_width=True,
        hide_index=True,
    )

    if not clim.empty:
        st.write(
            f"Climatologia usada: **{len(clim)}** observações históricas; "
            f"média **{clim['tmax_c'].mean():.1f} °C**, "
            f"desvio-padrão **{clim['tmax_c'].std(ddof=1):.1f} °C**."
        )

with tab3:
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("Sem avisos das fontes.")

st.caption(
    "Uso responsável: esta app não prevê eventos extremos locais perfeitamente e não substitui a leitura das regras exatas do mercado."
)

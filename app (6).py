# app.py
# Dashboard Pilotos OptiRoute (MVP) — Sustrend style
# - Robusto para Streamlit Cloud
# - Carga CSV/XLSX (NO .xls)
# - Normaliza columnas por mapeo
# - QA + KPIs + visualizaciones (Altair opcional)

import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
st.set_page_config(
    page_title="OptiRoute | Pilotos",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Dashboard Pilotos OptiRoute (MVP)"
APP_SUBTITLE = "Validación rápida (datos secundarios) + QA + KPIs estandarizados | Consultoría Sustrend SpA"

# =========================
# Paleta (clara, elegante; sin naranjo en textos)
# =========================
TXT_NAVY = "#0B2D4D"
TXT_DARK = "#334155"
TXT_MUTED = "#64748B"
ACCENT = "#009B72"      # menta/verde
ACCENT_2 = "#008CCF"    # cian/azul
BG = "#FFFFFF"
BORDER = "rgba(2, 6, 23, 0.10)"
SHADOW = "rgba(2, 6, 23, 0.08)"
GRID = "#E5E7EB"

# =========================
# Estilo (forzar claro + cards)
# =========================
st.markdown(
    f"""
<style>
:root {{ color-scheme: only light; }}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background: {BG} !important;
    color: {TXT_DARK} !important;
}}

[data-testid="stHeader"] {{
    background: rgba(255,255,255,0.92) !important;
}}

[data-testid="stSidebar"] {{
    background: {BG} !important;
    border-right: 1px solid {BORDER};
}}

* {{ color: {TXT_DARK}; }}
a, a * {{ color: {ACCENT_2} !important; }}

.main-title {{
    font-size: 34px;
    font-weight: 900;
    letter-spacing: -0.02em;
    margin: 0 0 6px 0;
    color: {TXT_NAVY} !important;
}}
.main-subtitle {{
    font-size: 13px;
    font-weight: 500;
    margin: 0 0 14px 0;
    color: {TXT_MUTED} !important;
}}

.hr {{
    border: 0;
    border-top: 1px solid {BORDER};
    margin: 10px 0 14px 0;
}}

.card {{
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 14px 16px;
    background: #fff;
    box-shadow: 0 12px 30px {SHADOW};
}}

.card-title {{
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: {TXT_NAVY} !important;
    margin-bottom: 8px;
}}

.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    border: 1px solid {BORDER};
    background: rgba(0,0,0,0.02);
    font-size: 12px;
    color: {TXT_MUTED} !important;
}}

.kpi-wrap {{
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 12px 14px;
    background: #fff;
    box-shadow: 0 10px 22px {SHADOW};
}}
.kpi-label {{
    font-size: 12px;
    color: {TXT_MUTED} !important;
    margin-bottom: 6px;
}}
.kpi-value {{
    font-size: 22px;
    font-weight: 900;
    letter-spacing: -0.02em;
    color: {TXT_NAVY} !important;
}}
.small {{
    font-size: 12px;
    color: {TXT_MUTED} !important;
}}

.note {{
    border-left: 4px solid {ACCENT};
    padding: 10px 12px;
    background: rgba(0, 155, 114, 0.06);
    border-radius: 12px;
    border: 1px solid rgba(0, 155, 114, 0.20);
}}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Optional charts
# =========================
try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False

# =========================
# Helpers
# =========================
def _normalize_colname(c: str) -> str:
    return re.sub(r"\s+", "_", str(c).strip().lower())

def _find_col(df: pd.DataFrame, candidates):
    norm_map = {_normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_colname(cand)
        if key in norm_map:
            return norm_map[key]
    return None

def _to_dt(series: pd.Series):
    return pd.to_datetime(series, errors="coerce")

def _to_num(series: pd.Series):
    return pd.to_numeric(series, errors="coerce")

def _fmt_pct(x: float):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "ND"
    return f"{x:.1%}"

def _kpi_card(label: str, value: str, sub: str | None = None):
    sub_html = f"<div class='small'>{sub}</div>" if sub else ""
    st.markdown(
        f"""
<div class="kpi-wrap">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  {sub_html}
</div>
""",
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def _read_csv_bytes(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    return pd.read_csv(bio)

@st.cache_data(show_spinner=False)
def _read_excel_bytes(file_name: str, file_bytes: bytes) -> dict:
    """
    Returns dict(sheet_name -> df) for robust sheet selection.
    """
    bio = io.BytesIO(file_bytes)
    xls = pd.ExcelFile(bio, engine="openpyxl")
    out = {}
    for sh in xls.sheet_names:
        out[sh] = pd.read_excel(xls, sheet_name=sh)
    return out

def read_any(uploaded_file) -> tuple[pd.DataFrame, str]:
    """
    Robust reader for Streamlit Cloud: uses bytes, forces openpyxl for xlsx.
    Returns (df, sheet_info).
    """
    fname = uploaded_file.name
    ext = fname.lower().split(".")[-1]
    data = uploaded_file.getvalue()

    if ext == "csv":
        df = _read_csv_bytes(fname, data)
        return df, "CSV"

    if ext in ("xlsx", "xlsm"):
        sheets = _read_excel_bytes(fname, data)
        # if multiple sheets, choose in sidebar
        if len(sheets) == 1:
            sh = list(sheets.keys())[0]
            return sheets[sh], f"XLSX · {sh}"
        else:
            sh = st.sidebar.selectbox(f"Hoja para {fname}", list(sheets.keys()), index=0)
            return sheets[sh], f"XLSX · {sh}"

    raise ValueError(f"Formato no soportado: .{ext}. Use CSV o XLSX (no .xls).")

# =========================
# Column maps (editables)
# =========================
ORDERS_MAP = {
    "order_id": ["order_id", "pedido_id", "id_pedido", "num_pedido", "pedido", "id"],
    "fecha_pedido": ["fecha_pedido", "fecha", "fecha_creacion", "created_at", "created", "timestamp"],
    "direccion": ["direccion", "address", "dirección"],
    "comuna": ["comuna", "ciudad", "city", "localidad"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "lng", "longitude"],
    "demanda": ["demanda", "peso", "volumen", "carga", "load"],
    "win_start": ["ventana_inicio", "window_start", "inicio_ventana", "inicio_ventana_horaria"],
    "win_end": ["ventana_fin", "window_end", "fin_ventana", "fin_ventana_horaria"],
}

ROUTES_MAP = {
    "route_id": ["route_id", "ruta_id", "id_ruta", "ruta", "id"],
    "fecha_ruta": ["fecha_ruta", "fecha", "dia_ruta", "day", "date"],
    "vehicle_id": ["vehicle_id", "vehiculo", "vehículo", "patente", "camion", "camión"],
    "stops": ["stops", "paradas", "num_paradas", "#paradas", "n_paradas"],
    "km_plan": ["km", "kms", "distancia_km", "distancia", "distance"],
    "dur_plan": ["duracion", "duración", "duracion_hhmm", "tiempo_ruta", "duration", "duracion_min"],
    "entregado": ["entregado", "delivered_pct", "%entregado", "entregado_pct"],
    "a_tiempo": ["a_tiempo", "on_time_pct", "%a_tiempo", "a_tiempo_pct"],
    "capacidad": ["capacidad", "capacity", "capacidad_vehiculo"],
    "carga": ["carga", "load", "demanda_total", "carga_asignada"],
}

def standardize(df: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, dict]:
    """
    Returns standardized df + mapping_used for traceability.
    """
    out = pd.DataFrame()
    used = {}
    for k, candidates in mapping.items():
        col = _find_col(df, candidates)
        if col:
            out[k] = df[col]
            used[k] = col
        else:
            out[k] = np.nan
            used[k] = None
    return out, used

def qa_report(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    rep = []
    rep.append(("Filas", len(df)))
    for col in required:
        if col in df.columns:
            rep.append((f"{col} nulos %", float(df[col].isna().mean())))
        else:
            rep.append((f"{col} nulos %", np.nan))
    return pd.DataFrame(rep, columns=["check", "value"])

def compute_kpis(routes: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    df = routes.copy()

    # normalize types
    df["fecha_ruta"] = _to_dt(df["fecha_ruta"])
    df["km_plan"] = _to_num(df["km_plan"])
    df["stops"] = _to_num(df["stops"])

    # delivered / ontime if present (accept 0-1 or 0-100)
    for col in ["entregado", "a_tiempo"]:
        if col in df.columns:
            x = _to_num(df[col])
            if x.notna().any() and x.max(skipna=True) > 1.5:
                x = x / 100.0
            df[col] = x

    vehicles_used = df["vehicle_id"].nunique(dropna=True) if df["vehicle_id"].notna().any() else np.nan

    kpi = {
        "rutas": int(len(df)),
        "km_plan_total": float(df["km_plan"].sum(skipna=True)) if df["km_plan"].notna().any() else np.nan,
        "km_plan_prom": float(df["km_plan"].mean(skipna=True)) if df["km_plan"].notna().any() else np.nan,
        "stops_prom": float(df["stops"].mean(skipna=True)) if df["stops"].notna().any() else np.nan,
        "vehiculos_usados": int(vehicles_used) if not (isinstance(vehicles_used, float) and np.isnan(vehicles_used)) else np.nan,
        "entregado_prom": float(df["entregado"].mean(skipna=True)) if df.get("entregado") is not None and df["entregado"].notna().any() else np.nan,
        "a_tiempo_prom": float(df["a_tiempo"].mean(skipna=True)) if df.get("a_tiempo") is not None and df["a_tiempo"].notna().any() else np.nan,
    }

    return kpi, df

def day_aggregate(routes_df: pd.DataFrame) -> pd.DataFrame:
    d = routes_df.copy()
    if "fecha_ruta" not in d.columns or d["fecha_ruta"].isna().all():
        return pd.DataFrame()
    d["day"] = d["fecha_ruta"].dt.date
    agg = d.groupby("day", dropna=True).agg(
        rutas=("route_id", "count"),
        km_plan=("km_plan", "sum"),
        stops=("stops", "sum"),
        vehiculos=("vehicle_id", lambda s: s.nunique(dropna=True)),
        entregado=("entregado", "mean"),
        a_tiempo=("a_tiempo", "mean"),
    ).reset_index()
    return agg

# =========================
# Header
# =========================
st.markdown(f"<div class='main-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='main-subtitle'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# =========================
# Sidebar — carga y configuración
# =========================
st.sidebar.markdown("### Carga de datos")
st.sidebar.caption("Formatos soportados: CSV / XLSX (no .xls)")

f_orders = st.sidebar.file_uploader("Pedidos (CSV/XLSX)", type=["csv", "xlsx", "xlsm"])
f_routes = st.sidebar.file_uploader("Planes de ruta (CSV/XLSX)", type=["csv", "xlsx", "xlsm"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Supuestos (opcional)")
consumo_l_100 = st.sidebar.number_input("Consumo promedio (L/100km)", min_value=0.0, value=0.0, step=0.1)
fe_kgco2_l = st.sidebar.number_input("Factor emisión (kgCO₂/L)", min_value=0.0, value=0.0, step=0.01)

if not f_orders or not f_routes:
    st.markdown(
        "<div class='note'><b>Para comenzar</b><br/>Cargar <i>Pedidos</i> y <i>Planes de ruta</i> en la barra lateral.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# =========================
# Read
# =========================
try:
    raw_orders, orders_sheet = read_any(f_orders)
    raw_routes, routes_sheet = read_any(f_routes)
except Exception as e:
    st.error("Error al leer archivos. Verificar formato (CSV/XLSX) y que no sea .xls.")
    st.exception(e)
    st.stop()

# Show loaded filenames
st.sidebar.markdown("---")
st.sidebar.markdown("### Archivos cargados")
st.sidebar.markdown(f"<span class='badge'>{f_orders.name} · {orders_sheet}</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='badge'>{f_routes.name} · {routes_sheet}</span>", unsafe_allow_html=True)

# =========================
# Standardize
# =========================
orders, orders_used = standardize(raw_orders, ORDERS_MAP)
routes, routes_used = standardize(raw_routes, ROUTES_MAP)

# Parse key dates
orders["fecha_pedido"] = _to_dt(orders["fecha_pedido"])
routes["fecha_ruta"] = _to_dt(routes["fecha_ruta"])

# Filters (by date)
min_d = routes["fecha_ruta"].min() if routes["fecha_ruta"].notna().any() else None
max_d = routes["fecha_ruta"].max() if routes["fecha_ruta"].notna().any() else None

st.sidebar.markdown("---")
st.sidebar.markdown("### Filtros")
if min_d is not None and max_d is not None:
    date_range = st.sidebar.date_input("Rango de fechas (rutas)", value=(min_d.date(), max_d.date()))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d0, d1 = date_range
        routes = routes[(routes["fecha_ruta"].dt.date >= d0) & (routes["fecha_ruta"].dt.date <= d1)]
else:
    st.sidebar.caption("No se detectaron fechas de rutas válidas para filtrar.")

# =========================
# Compute
# =========================
kpi, routes_kpi = compute_kpis(routes)
daily = day_aggregate(routes_kpi)

# CO2 estimate (optional)
co2_info = None
if consumo_l_100 > 0 and fe_kgco2_l > 0 and routes_kpi["km_plan"].notna().any():
    litros = routes_kpi["km_plan"].fillna(0) * (consumo_l_100 / 100.0)
    co2 = litros * fe_kgco2_l
    co2_info = {
        "litros_total": float(litros.sum()),
        "co2_total_kg": float(co2.sum()),
    }

# =========================
# Tabs
# =========================
tab_resumen, tab_qa, tab_rutas = st.tabs(["📌 Resumen", "🧪 QA / Trazabilidad", "🚚 Rutas"])

# ---------- TAB RESUMEN ----------
with tab_resumen:
    left, right = st.columns([1.5, 1.0])
    with left:
        st.markdown(
            """
<div class="card">
  <div class="card-title">Propósito</div>
  <div class="small">
    Este tablero estandariza <b>Pedidos</b> y <b>Planes de ruta</b>, calcula KPIs comparables y expone calidad de datos (QA).
    Si se ingresan supuestos, estima combustible y CO₂ a partir de km planificados.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
<div class="card">
  <div class="card-title">Estado de carga</div>
  <div class="small"><b>Pedidos:</b> {len(raw_orders):,} filas | <span class="badge">{orders_sheet}</span></div>
  <div class="small"><b>Planes:</b> {len(raw_routes):,} filas | <span class="badge">{routes_sheet}</span></div>
  <div class="small"><b>Última actualización:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.write("")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: _kpi_card("Rutas", f"{kpi['rutas']:,}")
    with c2: _kpi_card("Km total (plan)", "ND" if np.isnan(kpi["km_plan_total"]) else f"{kpi['km_plan_total']:.1f}")
    with c3: _kpi_card("Km prom/ruta", "ND" if np.isnan(kpi["km_plan_prom"]) else f"{kpi['km_plan_prom']:.1f}")
    with c4: _kpi_card("Stops prom", "ND" if np.isnan(kpi["stops_prom"]) else f"{kpi['stops_prom']:.1f}")
    with c5: _kpi_card("% Entregado", _fmt_pct(kpi["entregado_prom"]))
    with c6: _kpi_card("% A tiempo", _fmt_pct(kpi["a_tiempo_prom"]))

    if co2_info:
        st.write("")
        st.markdown(
            f"""
<div class="note">
  <b>Estimación CO₂ (basada en supuestos)</b><br/>
  Litros estimados: <b>{co2_info['litros_total']:.1f} L</b> · CO₂: <b>{co2_info['co2_total_kg']:.1f} kg</b><br/>
  <span class="small">Supuestos: consumo {consumo_l_100:.1f} L/100km · factor {fe_kgco2_l:.2f} kgCO₂/L</span>
</div>
""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("#### Evolución diaria (si hay fecha_ruta)")
    if daily.empty:
        st.info("No hay fechas válidas para construir series diarias.")
    else:
        if ALTAIR_OK:
            base = alt.Chart(daily).encode(
                x=alt.X("day:T", title=None, axis=alt.Axis(labelColor=TXT_MUTED, titleColor=TXT_MUTED)),
                tooltip=["day:T", "rutas:Q", "km_plan:Q", "stops:Q", "vehiculos:Q"],
            ).properties(height=260).configure_view(strokeOpacity=0).configure_axis(gridColor=GRID).configure(background=BG)

            ch_km = base.mark_line(color=ACCENT_2, strokeWidth=2).encode(
                y=alt.Y("km_plan:Q", title="Km (plan)", axis=alt.Axis(labelColor=TXT_MUTED, titleColor=TXT_MUTED)),
            )
            ch_rutas = base.mark_line(color=ACCENT, strokeWidth=2).encode(
                y=alt.Y("rutas:Q", title="Rutas", axis=alt.Axis(labelColor=TXT_MUTED, titleColor=TXT_MUTED)),
            )

            l1, l2 = st.columns(2)
            with l1:
                st.altair_chart(ch_km, use_container_width=True)
            with l2:
                st.altair_chart(ch_rutas, use_container_width=True)
        else:
            st.warning("Altair no está disponible. Se muestra tabla diaria.")
        st.dataframe(daily, use_container_width=True, hide_index=True)

# ---------- TAB QA ----------
with tab_qa:
    st.markdown("### QA / Trazabilidad de variables")
    qa1, qa2 = st.columns(2)
    with qa1:
        st.markdown("<div class='card-title'>Pedidos (estandarizado)</div>", unsafe_allow_html=True)
        st.dataframe(qa_report(orders, ["order_id", "fecha_pedido", "direccion"]), use_container_width=True, hide_index=True)
    with qa2:
        st.markdown("<div class='card-title'>Planes de ruta (estandarizado)</div>", unsafe_allow_html=True)
        st.dataframe(qa_report(routes_kpi, ["route_id", "fecha_ruta", "km_plan"]), use_container_width=True, hide_index=True)

    st.write("")
    st.markdown("#### Mapeo de columnas (origen → estándar)")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("<div class='card-title'>Pedidos</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"campo_estandar": list(orders_used.keys()), "columna_origen": list(orders_used.values())}),
                     use_container_width=True, hide_index=True)
    with m2:
        st.markdown("<div class='card-title'>Planes de ruta</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"campo_estandar": list(routes_used.keys()), "columna_origen": list(routes_used.values())}),
                     use_container_width=True, hide_index=True)

    st.write("")
    st.markdown("#### Alertas simples")
    alerts = []

    # duplicates
    if orders["order_id"].notna().any():
        dup_orders = int(orders["order_id"].duplicated(keep=False).sum())
        if dup_orders > 0:
            alerts.append(f"Pedidos: {dup_orders} filas con order_id duplicado.")
    if routes_kpi["route_id"].notna().any():
        dup_routes = int(routes_kpi["route_id"].duplicated(keep=False).sum())
        if dup_routes > 0:
            alerts.append(f"Rutas: {dup_routes} filas con route_id duplicado (revisar llave).")

    # missing km
    if routes_kpi["km_plan"].isna().mean() > 0.2:
        alerts.append("Más de 20% de km_plan está vacío: los KPIs de distancia serán débiles.")

    if alerts:
        for a in alerts:
            st.warning(a)
    else:
        st.success("Sin alertas críticas en las reglas simples definidas.")

# ---------- TAB RUTAS ----------
with tab_rutas:
    st.markdown("### Rutas (tabla + filtros)")

    # route filters
    fcol1, fcol2, fcol3 = st.columns([1.2, 1.2, 1.0])
    with fcol1:
        veh_list = sorted([v for v in routes_kpi["vehicle_id"].dropna().astype(str).unique().tolist()])
        sel_veh = st.multiselect("Vehículo (opcional)", veh_list, default=[])
    with fcol2:
        # km filter
        km_min = float(routes_kpi["km_plan"].min()) if routes_kpi["km_plan"].notna().any() else 0.0
        km_max = float(routes_kpi["km_plan"].max()) if routes_kpi["km_plan"].notna().any() else 0.0
        if km_max > 0:
            sel_km = st.slider("Km plan (rango)", min_value=float(km_min), max_value=float(km_max), value=(float(km_min), float(km_max)))
        else:
            sel_km = None
    with fcol3:
        topn = st.selectbox("Mostrar", [50, 100, 200, 500], index=2)

    df_show = routes_kpi.copy()
    if sel_veh:
        df_show = df_show[df_show["vehicle_id"].astype(str).isin(sel_veh)]
    if sel_km:
        a, b = sel_km
        df_show = df_show[(df_show["km_plan"] >= a) & (df_show["km_plan"] <= b)]

    # nicer ordering
    if df_show["fecha_ruta"].notna().any():
        df_show = df_show.sort_values("fecha_ruta", ascending=False)

    st.dataframe(df_show.head(int(topn)), use_container_width=True, hide_index=True)

    st.write("")
    st.markdown("#### Distribución de km por ruta")
    if df_show["km_plan"].notna().any():
        if ALTAIR_OK:
            h = alt.Chart(df_show.dropna(subset=["km_plan"])).mark_bar(color=ACCENT_2).encode(
                x=alt.X("km_plan:Q", bin=alt.Bin(maxbins=20), title="Km (plan)"),
                y=alt.Y("count():Q", title="Nº rutas"),
                tooltip=["count():Q"]
            ).properties(height=260).configure_view(strokeOpacity=0).configure_axis(gridColor=GRID).configure(background=BG)
            st.altair_chart(h, use_container_width=True)
        else:
            st.write(df_show["km_plan"].describe())
    else:
        st.info("No hay km_plan para graficar distribución.")

# =========================
# Footer
# =========================
st.sidebar.markdown("---")
st.sidebar.caption(f"Tiempo local: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# app_optiroute.py
import re
import numpy as np
import pandas as pd
import streamlit as st

# --------- helpers base (igual a tu patrón) ----------
def _normalize_colname(c: str) -> str:
    return re.sub(r"\s+", "_", str(c).strip().lower())

def _find_col(df: pd.DataFrame, candidates):
    norm_map = {_normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_colname(cand)
        if key in norm_map:
            return norm_map[key]
    return None

@st.cache_data
def read_any(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

# --------- mapeos sugeridos ----------
ORDERS_MAP = {
  "order_id": ["order_id","pedido_id","id_pedido","num_pedido","pedido"],
  "fecha_pedido": ["fecha_pedido","fecha","fecha_creacion","created_at"],
  "direccion": ["direccion","address","dirección"],
  "comuna": ["comuna","ciudad","city"],
  "lat": ["lat","latitude"],
  "lon": ["lon","lng","longitude"],
  "demanda": ["demanda","peso","volumen","carga"],
  "win_start": ["ventana_inicio","window_start","inicio_ventana"],
  "win_end": ["ventana_fin","window_end","fin_ventana"],
}

ROUTES_MAP = {
  "route_id": ["route_id","ruta_id","id_ruta","ruta"],
  "fecha_ruta": ["fecha_ruta","fecha","dia_ruta"],
  "vehicle_id": ["vehicle_id","vehiculo","vehículo","patente"],
  "stops": ["stops","paradas","num_paradas","#paradas"],
  "km_plan": ["km","kms","distancia_km","distancia"],
  "dur_plan": ["duracion","duración","duracion_hhmm","tiempo_ruta"],
  "entregado": ["entregado","delivered_pct","%entregado"],
  "a_tiempo": ["a_tiempo","on_time_pct","%a_tiempo"],
  "capacidad": ["capacidad","capacity"],
  "carga": ["carga","load","demanda_total"],
}

def standardize(df, mapping):
    out = pd.DataFrame()
    for k, candidates in mapping.items():
        col = _find_col(df, candidates)
        out[k] = df[col] if col else np.nan
    return out

def qa_report_orders(orders):
    rep = []
    rep.append(("filas", len(orders)))
    rep.append(("order_id nulos %", float(orders["order_id"].isna().mean())))
    rep.append(("fecha_pedido nulos %", float(orders["fecha_pedido"].isna().mean())))
    rep.append(("direccion nulos %", float(orders["direccion"].isna().mean())))
    return pd.DataFrame(rep, columns=["check","value"])

def qa_report_routes(routes):
    rep = []
    rep.append(("filas", len(routes)))
    rep.append(("route_id nulos %", float(routes["route_id"].isna().mean())))
    rep.append(("fecha_ruta nulos %", float(routes["fecha_ruta"].isna().mean())))
    rep.append(("km_plan nulos %", float(routes["km_plan"].isna().mean())))
    return pd.DataFrame(rep, columns=["check","value"])

def compute_kpis(routes):
    df = routes.copy()
    df["km_plan"] = pd.to_numeric(df["km_plan"], errors="coerce")
    df["stops"] = pd.to_numeric(df["stops"], errors="coerce")
    # ejemplo: KPIs simples agregados
    kpi = {
        "rutas": int(len(df)),
        "km_plan_total": float(df["km_plan"].sum(skipna=True)),
        "km_plan_prom": float(df["km_plan"].mean(skipna=True)) if df["km_plan"].notna().any() else np.nan,
        "stops_prom": float(df["stops"].mean(skipna=True)) if df["stops"].notna().any() else np.nan,
        "vehiculos_usados": int(df["vehicle_id"].nunique()) if df["vehicle_id"].notna().any() else np.nan,
    }
    return kpi, df

# --------- UI ----------
st.set_page_config(page_title="Optiroute | Pilotos", layout="wide")
st.title("Dashboard Pilotos Optiroute (MVP)")

st.sidebar.header("Carga de datos")
f_orders = st.sidebar.file_uploader("Pedidos (CSV/XLSX)", type=["csv","xlsx","xls"])
f_routes = st.sidebar.file_uploader("Planes de ruta (CSV/XLSX)", type=["csv","xlsx","xls"])

if not f_orders or not f_routes:
    st.info("Cargar Pedidos y Planes de ruta para comenzar.")
    st.stop()

raw_orders = read_any(f_orders)
raw_routes = read_any(f_routes)

orders = standardize(raw_orders, ORDERS_MAP)
routes = standardize(raw_routes, ROUTES_MAP)

# QA
st.subheader("QA (calidad de datos)")
c1, c2 = st.columns(2)
c1.dataframe(qa_report_orders(orders), use_container_width=True, hide_index=True)
c2.dataframe(qa_report_routes(routes), use_container_width=True, hide_index=True)

# KPIs
kpi, routes_kpi = compute_kpis(routes)
st.subheader("Resumen")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rutas", kpi["rutas"])
m2.metric("Km total (plan)", f"{kpi['km_plan_total']:.1f}")
m3.metric("Km prom/ruta", "ND" if pd.isna(kpi["km_plan_prom"]) else f"{kpi['km_plan_prom']:.1f}")
m4.metric("Stops prom", "ND" if pd.isna(kpi["stops_prom"]) else f"{kpi['stops_prom']:.1f}")
m5.metric("Vehículos usados", "ND" if pd.isna(kpi["vehiculos_usados"]) else int(kpi["vehiculos_usados"]))

st.subheader("Tabla de rutas (estandarizada)")
st.dataframe(routes_kpi.head(200), use_container_width=True, hide_index=True)

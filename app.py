import io
import csv
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Prediksi Stunting Balita", layout="wide")

DEFAULT_DATA_PATH = "data/dinkes-od_17148_persentase_balita_stunting__kabupatenkota_v3_data.csv"

FEATURES = ["lag1", "lag2", "lag3", "mean_prev", "slope_prev"]

# =========================
# UTIL: SAFE CSV READER (auto delimiter + encoding-ish)
# =========================
def _sniff_sep(sample_text: str) -> str:
    # fallback safe
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        # if it contains many ';' then use ';'
        if sample_text.count(";") > sample_text.count(","):
            return ";"
        return ","

def read_csv_safely(file_bytes: bytes) -> pd.DataFrame:
    # Try utf-8 first, then fallback ignore errors
    try:
        text = file_bytes.decode("utf-8")
    except Exception:
        text = file_bytes.decode("utf-8", errors="ignore")

    sample = text[:5000]
    sep = _sniff_sep(sample)

    # Use StringIO for pandas
    buf = io.StringIO(text)
    df = pd.read_csv(buf, sep=sep)
    return df

def normalize_col(c: str) -> str:
    return (
        str(c)
        .strip()
        .lower()
        .replace("\ufeff", "")  # BOM
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset beda-beda sering kolomnya tidak persis.
    Kita auto-map ke 5 kolom wajib:
    - nama_provinsi
    - kode_kabupaten_kota
    - nama_kabupaten_kota
    - tahun
    - persentase_balita_stunting
    """
    original_cols = list(df.columns)
    col_map_norm = {normalize_col(c): c for c in original_cols}

    # kandidat nama kolom
    candidates = {
        "nama_provinsi": ["nama_provinsi", "provinsi", "nama_prov", "prov"],
        "kode_kabupaten_kota": ["kode_kabupaten_kota", "kode_kabupaten", "kode_kota", "kode_kab_kota", "kode_wilayah", "kode"],
        "nama_kabupaten_kota": ["nama_kabupaten_kota", "kabupaten_kota", "nama_kabupaten", "nama_kota", "kab_kota", "kabupaten", "kota", "wilayah"],
        "tahun": ["tahun", "year"],
        "persentase_balita_stunting": ["persentase_balita_stunting", "persentase_stunting", "stunting", "persentase", "percentage", "nilai"],
    }

    resolved = {}
    for target, opts in candidates.items():
        found = None
        for opt in opts:
            if opt in col_map_norm:
                found = col_map_norm[opt]
                break
        resolved[target] = found

    missing = [k for k, v in resolved.items() if v is None]
    if missing:
        raise ValueError(
            "Kolom wajib tidak ditemukan: "
            + ", ".join(missing)
            + "\n\nKolom yang ada di CSV:\n- "
            + "\n- ".join([str(c) for c in original_cols])
        )

    df2 = df.rename(columns={
        resolved["nama_provinsi"]: "nama_provinsi",
        resolved["kode_kabupaten_kota"]: "kode_kabupaten_kota",
        resolved["nama_kabupaten_kota"]: "nama_kabupaten_kota",
        resolved["tahun"]: "tahun",
        resolved["persentase_balita_stunting"]: "persentase_balita_stunting",
    })

    return df2

@st.cache_data(show_spinner=False)
def load_data_from_repo(path: str) -> pd.DataFrame:
    # read as bytes to reuse safe reader behaviour
    with open(path, "rb") as f:
        b = f.read()
    df = read_csv_safely(b)
    return df

def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = map_columns(df_raw)

    # strip strings
    df["nama_provinsi"] = df["nama_provinsi"].astype(str).str.strip()
    df["nama_kabupaten_kota"] = df["nama_kabupaten_kota"].astype(str).str.strip()

    # numeric conversions
    df["tahun"] = pd.to_numeric(df["tahun"], errors="coerce")
    df["persentase_balita_stunting"] = pd.to_numeric(df["persentase_balita_stunting"], errors="coerce")

    df = df.dropna(subset=[
        "nama_provinsi",
        "kode_kabupaten_kota",
        "nama_kabupaten_kota",
        "tahun",
        "persentase_balita_stunting",
    ])

    df["tahun"] = df["tahun"].astype(int)

    return df

def build_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grp_cols = ["nama_provinsi", "kode_kabupaten_kota", "nama_kabupaten_kota"]

    for (prov, kode, kab), g in df.groupby(grp_cols):
        g = g.sort_values("tahun")
        years = g["tahun"].values.astype(int)
        vals = g["persentase_balita_stunting"].values.astype(float)

        # target = vals[i], feature from vals[i-1], vals[i-2], vals[i-3]
        for i in range(1, len(vals)):
            lag1 = vals[i - 1]
            lag2 = vals[i - 2] if i - 2 >= 0 else np.nan
            lag3 = vals[i - 3] if i - 3 >= 0 else np.nan

            mean_prev = np.nanmean([lag1, lag2, lag3])

            xi = years[max(0, i - 3): i]
            yi = vals[max(0, i - 3): i]
            slope = np.polyfit(xi, yi, 1)[0] if len(xi) >= 2 else 0.0

            rows.append({
                "provinsi": prov,
                "kode_kabupaten_kota": str(kode),
                "kabupaten_kota": kab,
                "year_target": years[i],
                "lag1": lag1,
                "lag2": lag2,
                "lag3": lag3,
                "mean_prev": mean_prev,
                "slope_prev": slope,
                "target": vals[i],
            })

    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError("Data training kosong. Pastikan tiap kab/kota punya minimal 2 tahun data.")

    # fill NaN features with column mean
    for c in FEATURES:
        data[c] = data[c].astype(float)
    data[FEATURES] = data[FEATURES].fillna(data[FEATURES].mean(numeric_only=True))

    return data

def eval_metrics(y_true, y_pred) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    denom = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)
    acc = float(100.0 - mape) if np.isfinite(mape) else float("nan")

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE(%)": mape, "Accuracy(%)": acc}

@st.cache_resource(show_spinner=False)
def train_model(df_clean: pd.DataFrame, algo: str):
    data = build_training_rows(df_clean)
    X = data[FEATURES]
    y = data["target"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if algo == "Decision Tree (cepat)":
        model = DecisionTreeRegressor(random_state=42, max_depth=6)
    else:
        # RF dibuat ringan biar cloud nggak ngendon
        model = RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = eval_metrics(y_test.values, pred)
    return model, metrics

def classify_priority(x: float) -> str:
    if x < 10:
        return "Prioritas Rendah"
    elif x <= 20:
        return "Prioritas Sedang"
    return "Prioritas Tinggi"

def compute_lags(series: pd.DataFrame):
    series = series.sort_values("tahun").copy()
    vals = series["persentase_balita_stunting"].values.astype(float)
    years = series["tahun"].values.astype(int)

    lag1 = vals[-1]
    lag2 = vals[-2] if len(vals) >= 2 else np.nan
    lag3 = vals[-3] if len(vals) >= 3 else np.nan

    mean_prev = np.nanmean([lag1, lag2, lag3])

    xi = years[-3:] if len(years) >= 3 else years
    yi = vals[-3:] if len(vals) >= 3 else vals
    slope = np.polyfit(xi, yi, 1)[0] if len(xi) >= 2 else 0.0

    return lag1, lag2, lag3, mean_prev, slope

def predict_n_years(model, series: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    series = series.sort_values("tahun").copy()
    out = []

    for _ in range(n):
        lag1, lag2, lag3, mean_prev, slope = compute_lags(series)

        feat = np.array([lag1, lag2, lag3, mean_prev, slope], dtype=float)

        # impute NaN
        if np.any(np.isnan(feat)):
            m = np.nanmean(feat)
            feat = np.where(np.isnan(feat), m if np.isfinite(m) else 0.0, feat)

        yhat = float(model.predict(feat.reshape(1, -1))[0])
        next_year = int(series["tahun"].max()) + 1

        out.append({
            "tahun": next_year,
            "prediksi_persen": yhat,
            "prioritas": classify_priority(yhat),
        })

        # append predicted to series for iterative forecasting
        series = pd.concat([series, pd.DataFrame([{
            "tahun": next_year,
            "persentase_balita_stunting": yhat
        }])], ignore_index=True)

    return pd.DataFrame(out)

# =========================
# UI
# =========================
st.title("🌱 Prediksi Stunting Balita (Provinsi → Kabupaten/Kota)")
st.caption("Pilih wilayah → tampilkan lag otomatis → prediksi 1–3 tahun ke depan. Template ini anti-crash & jelas kalau error.")

with st.sidebar:
    st.header("Sumber Data")
    uploaded = st.file_uploader("Upload CSV (opsional)", type=["csv"])
    algo = st.selectbox("Model", ["Decision Tree (cepat)", "Random Forest (lebih berat)"], index=0)

    st.markdown("---")
    st.header("Debug")
    show_debug = st.checkbox("Tampilkan debug info", value=False)

# Load data with robust fallback
try:
    if uploaded is not None:
        raw = read_csv_safely(uploaded.getvalue())
        df = clean_data(raw)
        source_label = "Uploaded CSV"
    else:
        raw = load_data_from_repo(DEFAULT_DATA_PATH)
        df = clean_data(raw)
        source_label = f"Repo file: {DEFAULT_DATA_PATH}"

except Exception as e:
    st.error("Gagal membaca dataset. Ini penyebabnya:")
    st.exception(e)
    st.info("Fix cepat:\n- Pastikan CSV benar-benar ada di repo (folder data/)\n- Pastikan kolom provinsi/kabupaten/tahun/persentase ada\n- Atau upload CSV lewat sidebar")
    st.stop()

# Train model
try:
    model, metrics = train_model(df, algo)
except Exception as e:
    st.error("Gagal training model. Ini penyebabnya:")
    st.exception(e)
    st.info("Fix cepat:\n- Pastikan setiap kab/kota punya minimal 2 data tahun\n- Kurangi model ke Decision Tree (cepat)\n- Cek apakah kolom persentase & tahun valid")
    st.stop()

# Header metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sumber", source_label)
c2.metric("Accuracy (%)", f"{metrics['Accuracy(%)']:.2f}" if np.isfinite(metrics["Accuracy(%)"]) else "N/A")
c3.metric("MAE", f"{metrics['MAE']:.3f}")
c4.metric("RMSE", f"{metrics['RMSE']:.3f}")
c5.metric("R2", f"{metrics['R2']:.3f}")

if show_debug:
    with st.expander("Debug: columns & sample"):
        st.write("Kolom setelah mapping:", list(df.columns))
        st.dataframe(df.head(20), use_container_width=True)

# Filters
prov_list = sorted(df["nama_provinsi"].unique().tolist())
selected_prov = st.selectbox("Pilih Provinsi", prov_list)

dfp = df[df["nama_provinsi"] == selected_prov].copy()
kab_list = sorted(dfp["nama_kabupaten_kota"].unique().tolist())
search = st.text_input("Cari Kabupaten/Kota (opsional)", "")

if search.strip():
    kab_list = [k for k in kab_list if search.lower() in k.lower()]

selected_kab = st.selectbox("Pilih Kabupaten/Kota", kab_list)

series = dfp[dfp["nama_kabupaten_kota"] == selected_kab][["tahun", "persentase_balita_stunting"]].dropna()
series = series.sort_values("tahun")

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Riwayat Stunting (Persen)")
    st.dataframe(series, use_container_width=True, hide_index=True)

    lag1, lag2, lag3, mean_prev, slope = compute_lags(series)
    a, b, c, d, e = st.columns(5)
    a.metric("Lag-1 (tahun terakhir)", f"{lag1:.2f}")
    b.metric("Lag-2", f"{lag2:.2f}" if np.isfinite(lag2) else "N/A")
    c.metric("Lag-3", f"{lag3:.2f}" if np.isfinite(lag3) else "N/A")
    d.metric("Rata-rata (lag1-3)", f"{mean_prev:.2f}" if np.isfinite(mean_prev) else "N/A")
    e.metric("Tren (slope)", f"{slope:.4f}")

with right:
    st.subheader("Prediksi")
    horizon = st.selectbox("Prediksi berapa tahun ke depan?", [1, 2, 3], index=2)
    if st.button("🔮 Prediksi Sekarang", use_container_width=True):
        try:
            pred_df = predict_n_years(model, series, int(horizon))
            st.success("Prediksi berhasil.")
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            first = pred_df.iloc[0]
            st.info(f"Prediksi {int(first['tahun'])}: **{first['prediksi_persen']:.2f}%** — {first['prioritas']}")
        except Exception as e:
            st.error("Prediksi gagal. Ini penyebabnya:")
            st.exception(e)

st.caption("Prediksi 2–3 tahun dibuat iteratif: hasil prediksi tahun sebelumnya dipakai sebagai input tahun berikutnya.")

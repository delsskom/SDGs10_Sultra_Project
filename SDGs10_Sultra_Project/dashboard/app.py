import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import config
import utils

st.set_page_config(
    page_title="Dashboard SDGs 10 Sultra",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard SDGs 10 – Provinsi Sulawesi Tenggara")
st.write("Visualisasi berbagai indikator: Kemiskinan, Ekonomi, Pendidikan, Demografi, Akses Pelayanan Publik, dan Ketimpangan Pengeluaran.")

# ===========================
# SIDEBAR
# ===========================
st.sidebar.header("Pilih Dataset")

dataset_name = st.sidebar.selectbox(
    "Dataset:", 
    list(config.DATASET_PATHS.keys())
)

# Load dataset
df = pd.read_csv(config.DATASET_PATHS[dataset_name])

st.subheader(f"📁 Dataset: **{dataset_name}**")

# ===========================
# TAMPILKAN DATAFRAME
# ===========================
st.write("### 🔍 Data Preview")
st.dataframe(df, use_container_width=True)

# ===========================
# STATISTIK
# ===========================
st.write("### 📈 Statistik Deskriptif")
st.dataframe(df.describe(), use_container_width=True)

# ===========================
# PLOT 1 — Bar Chart otomatis
# ===========================
st.write("### 📊 Visualisasi Bar Chart")

numeric_cols = utils.get_numeric_columns(df)
category_cols = utils.get_category_columns(df)

if len(numeric_cols) > 0 and len(category_cols) > 0:
    x_col = st.selectbox("Kolom kategori (X-axis):", category_cols)
    y_col = st.selectbox("Kolom numerik (Y-axis):", numeric_cols)

    bar_fig = px.bar(df, x=x_col, y=y_col, color=x_col, title=f"{y_col} berdasarkan {x_col}")
    st.plotly_chart(bar_fig, use_container_width=True)
else:
    st.info("Dataset ini tidak memiliki kombinasi kolom kategori & numerik yang cocok untuk bar chart.")

# ===========================
# PLOT 2 — Line Chart
# ===========================
st.write("### 📈 Line Chart")

if len(numeric_cols) >= 1:
    line_col = st.selectbox("Pilih variabel numerik:", numeric_cols, key="line")
    line_fig = px.line(df, y=line_col, title=f"Perubahan {line_col}")
    st.plotly_chart(line_fig, use_container_width=True)
else:
    st.info("Dataset tidak memiliki kolom numerik untuk line chart.")

# ===========================
# PLOT 3 — Correlation Heatmap
# ===========================
st.write("### 🔥 Correlation Heatmap")

if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=.5, ax=ax)
    st.pyplot(fig)
else:
    st.info("Dataset tidak memiliki cukup kolom numerik untuk heatmap korelasi.")

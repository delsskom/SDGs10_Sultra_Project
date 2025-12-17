import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- CONFIGURATION & STYLE ---
st.set_page_config(
    page_title="Sultra Development Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS agar tampilan lebih bersih (Menghilangkan padding berlebih)
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 0rem;}
        h1 {color: #2c3e50;}
        h2 {color: #34495e;}
        .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)

# --- CACHING FUNCTION (Supaya cepat, tidak load ulang terus) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../data/data_final_sultra.csv')
        if 'ipm_total' not in df.columns:
            df['ipm_total'] = (df['ipm_l'] + df['ipm_p']) / 2
        return df
    except Exception as e:
        st.error(f"Gagal load data: {e}")
        return None

df = load_data()

if df is None:
    st.error("⚠️ File 'data_final_sultra.csv' tidak ditemukan! Jalankan kode simpan CSV di Colab dulu.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎛️ Kontrol Panel")
    st.info("Dashboard Analisis SDGs 10\nProvinsi Sulawesi Tenggara")
    st.markdown("---")
    
    # Filter Interaktif (Opsional untuk tabel)
    selected_regions = st.multiselect(
        "Filter Wilayah Tertentu:",
        options=df['Kabupaten/Kota'].unique(),
        default=df['Kabupaten/Kota'].unique()
    )
    
    st.markdown("---")
    st.caption("Created by: Kelompok 1")

# Filter Dataframe berdasarkan sidebar
df_filtered = df[df['Kabupaten/Kota'].isin(selected_regions)]

# --- MAIN HEADER ---
st.title("📈 Dashboard Ketimpangan & Pembangunan Wilayah")
st.markdown("Analisis Komprehensif Berbasis Data Science (EDA & K-Means Clustering)")

# Scorecards (Baris Atas)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Wilayah Dianalisis", f"{len(df_filtered)} Kab/Kota")
col2.metric("Rata-rata Kemiskinan", f"{df_filtered['persen_miskin_pct'].mean():.2f}%")
col3.metric("Rata-rata IPM Total", f"{df_filtered['ipm_total'].mean():.2f}")
col4.metric("Avg PDRB (Juta)", f"Rp {df_filtered['pdrb_perkapita_jt'].mean():.1f}")

st.markdown("---")

# --- TABS NAVIGATION (Agar Rapi & Terstruktur) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ Ketimpangan Multidimensi", 
    "2️⃣ Infrastruktur vs Kemiskinan",
    "3️⃣ Pola Ekonomi (Bubble)",
    "4️⃣ Gender & Pendidikan",
    "5️⃣ Modeling (Clustering)"
])

# ==============================================================================
# TAB 1: KETIMPANGAN MULTIDIMENSI (HEATMAP)
# ==============================================================================
with tab1:
    st.header("1. Peta Ketimpangan Multidimensional")
    st.markdown("Visualisasi ini menjawab: *Bagaimana posisi relatif setiap daerah pada dimensi Ekonomi, Sosial, dan Infrastruktur?*")
    
    col_viz, col_desc = st.columns([2, 1])
    
    with col_viz:
        # Normalisasi Data untuk Heatmap
        cols_dimensi = ['pdrb_perkapita_jt', 'ipm_l', 'ipm_p', 'akses_internet_pct', 'persen_miskin_pct']
        scaler = MinMaxScaler()
        df_norm = df_filtered.copy()
        df_norm[cols_dimensi] = scaler.fit_transform(df_filtered[cols_dimensi])
        df_norm['persen_miskin_pct'] = 1 - df_norm['persen_miskin_pct'] # Balik logika (Hijau = Bagus)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            df_norm.set_index('Kabupaten/Kota')[cols_dimensi],
            annot=df_filtered.set_index('Kabupaten/Kota')[cols_dimensi],
            fmt=".1f", cmap='RdYlGn', linewidths=0.5, ax=ax
        )
        plt.title('Heatmap Kualitas Pembangunan (Hijau = Baik)')
        st.pyplot(fig)

    with col_desc:
        st.info("""
        **Cara Membaca:**
        - **Warna Hijau:** Indikator sangat baik (PDRB tinggi, Miskin rendah).
        - **Warna Merah:** Indikator tertinggal.
        
        **Insight:**
        Perhatikan perbedaan kontras antara Kota Kendari (Dominan Hijau) dengan wilayah Kepulauan (Dominan Merah/Kuning).
        """)

# ==============================================================================
# TAB 2: INFRASTRUKTUR VS KEMISKINAN
# ==============================================================================
with tab2:
    st.header("2. Dampak Layanan Publik terhadap Kesejahteraan")
    st.markdown("Analisis Korelasi: *Apakah internet dan sanitasi benar-benar menurunkan kemiskinan?*")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Akses Internet vs Kemiskinan")
        fig2a = plt.figure(figsize=(8, 5))
        sns.regplot(data=df_filtered, x='akses_internet_pct', y='persen_miskin_pct', 
                    scatter_kws={'s':100, 'color':'green'}, line_kws={'color':'red'})
        plt.xlabel("Akses Internet (%)")
        plt.ylabel("Penduduk Miskin (%)")
        plt.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig2a)
        
    with col_b:
        st.subheader("Sanitasi vs Kemiskinan")
        fig2b = plt.figure(figsize=(8, 5))
        sns.regplot(data=df_filtered, x='akses_sanitasi_pct', y='persen_miskin_pct', 
                    scatter_kws={'s':100, 'color':'blue'}, line_kws={'color':'orange'})
        plt.xlabel("Akses Sanitasi Layak (%)")
        plt.ylabel("")
        plt.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig2b)
    
    st.success("**Kesimpulan:** Internet memiliki korelasi negatif terkuat. Wilayah dengan akses internet tinggi secara konsisten memiliki tingkat kemiskinan lebih rendah.")

# ==============================================================================
# TAB 3: BUBBLE CHART (EKONOMI VS DAYA BELI)
# ==============================================================================
with tab3:
    st.header("3. Peta Variasi Pembangunan (Bubble Chart)")
    st.markdown("Hubungan 3 Dimensi: **PDRB (X)** vs **Kemiskinan (Y)** vs **Daya Beli (Besar Lingkaran)**.")
    
    fig3 = plt.figure(figsize=(12, 7))
    
    # Hitung pengeluaran ribuan agar size wajar
    df_filtered['size_bubble'] = df_filtered['pengeluaran_rp'] / 1000
    
    sns.scatterplot(
        data=df_filtered,
        x='pdrb_perkapita_jt', y='persen_miskin_pct',
        size='size_bubble', sizes=(100, 1000),
        hue='Kabupaten/Kota', palette='tab20', legend=False, alpha=0.7
    )
    
    # Labeling
    for i in range(df_filtered.shape[0]):
        plt.text(
            df_filtered.iloc[i]['pdrb_perkapita_jt']+1,
            df_filtered.iloc[i]['persen_miskin_pct'],
            df_filtered.iloc[i]['Kabupaten/Kota'],
            fontsize=9
        )
        
    # Garis Kuadran
    plt.axvline(x=df_filtered['pdrb_perkapita_jt'].mean(), color='red', linestyle='--', label='Avg PDRB')
    plt.axhline(y=df_filtered['persen_miskin_pct'].mean(), color='blue', linestyle='--', label='Avg Miskin')
    
    plt.xlabel("PDRB per Kapita (Ekonomi)")
    plt.ylabel("Tingkat Kemiskinan")
    plt.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig3)
    
    st.caption("Semakin besar lingkaran = Semakin tinggi daya beli masyarakat.")

# ==============================================================================
# TAB 4: GENDER GAP (DUMBBELL PLOT)
# ==============================================================================
with tab4:
    st.header("4. Analisis Disparitas Gender (SDGs 5)")
    
    # Prepare Data
    df_sorted = df_filtered.copy()
    df_sorted['gap'] = df_sorted['ipm_l'] - df_sorted['ipm_p']
    df_sorted = df_sorted.sort_values('gap', ascending=False)
    
    fig4 = plt.figure(figsize=(12, 8))
    plt.hlines(y=df_sorted['Kabupaten/Kota'], xmin=df_sorted['ipm_p'], xmax=df_sorted['ipm_l'], color='grey', alpha=0.5, linewidth=3)
    plt.scatter(df_sorted['ipm_p'], df_sorted['Kabupaten/Kota'], color='red', s=100, label='Perempuan')
    plt.scatter(df_sorted['ipm_l'], df_sorted['Kabupaten/Kota'], color='navy', s=100, label='Laki-laki')
    
    plt.legend()
    plt.title("Gap IPM Laki-laki vs Perempuan (Semakin panjang garis = Semakin timpang)")
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    st.pyplot(fig4)

# ==============================================================================
# TAB 5: CLUSTERING (THE MAIN EVENT)
# ==============================================================================
with tab5:
    st.header("5. Pemodelan Klaster Wilayah (K-Means)")
    
    col_model, col_res = st.columns([3, 1])
    
    with col_model:
        # --- PROSES MODELING ---
        features_model = ['pdrb_perkapita_jt', 'persen_miskin_pct', 'ipm_total', 'akses_internet_pct']
        X = df_filtered[features_model]
        
        # Scaling
        scaler_model = StandardScaler()
        X_scaled_model = scaler_model.fit_transform(X)
        
        # KMeans (K=3)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_filtered['Cluster'] = kmeans.fit_predict(X_scaled_model)
        
        # --- LOGIKA LABEL OTOMATIS YANG LEBIH CERDAS ---
        # 1. Hitung rata-rata PDRB dan IPM tiap klaster
        clus_profile = df_filtered.groupby('Cluster')[['pdrb_perkapita_jt', 'ipm_total']].mean()
        
        # 2. Cari Klaster mana yang "Kota" (IPM Tertinggi)
        cluster_kota = clus_profile['ipm_total'].idxmax()
        
        # 3. Cari Klaster mana yang "Tambang/Industri" (PDRB Tertinggi)
        # (Kecuali jika PDRB tertinggi itu sama dengan Kota, maka cari PDRB tertinggi kedua)
        cluster_tambang = clus_profile['pdrb_perkapita_jt'].idxmax()
        
        # Jika kebetulan Kota juga PDRB-nya tertinggi (jarang terjadi di Sultra), logic sederhana:
        if cluster_tambang == cluster_kota:
             # Cari PDRB tertinggi kedua untuk label tambang
             clus_profile_temp = clus_profile.drop(cluster_kota)
             cluster_tambang = clus_profile_temp['pdrb_perkapita_jt'].idxmax()

        # 4. Sisanya adalah "Tertinggal/Berkembang"
        all_clusters = set(df_filtered['Cluster'].unique())
        cluster_tertinggal = list(all_clusters - {cluster_kota, cluster_tambang})[0]
        
        # 5. Buat Dictionary Mapping
        mapping = {
            cluster_kota: 'Maju (Kota/Jasa)',         # Kendari, Baubau (IPM Tinggi)
            cluster_tambang: 'Kaya SDA (Tambang)',    # Kolaka, Konut (PDRB Tinggi)
            cluster_tertinggal: 'Tertinggal (Kepulauan)' # Buton Raya, Muna (Rendah semua)
        }
        
        df_filtered['Label_Cluster'] = df_filtered['Cluster'].map(mapping)
        
        # --- VISUALISASI DENGAN LABEL ---
        fig5 = plt.figure(figsize=(11, 7))
        sns.scatterplot(
            data=df_filtered, 
            x='pdrb_perkapita_jt', y='persen_miskin_pct', 
            hue='Label_Cluster', palette='viridis', 
            s=250, style='Label_Cluster', edgecolor='black'
        )
        
        # LOOPING LABEL NAMA KOTA (INI YANG KAMU MINTA)
        # Menggunakan adjustText simple logic (geser dikit)
        for i in range(df_filtered.shape[0]):
            plt.text(
                x=df_filtered.iloc[i]['pdrb_perkapita_jt'] + 1, # Geser X dikit ke kanan
                y=df_filtered.iloc[i]['persen_miskin_pct'] + 0.2, # Geser Y dikit ke atas
                s=df_filtered.iloc[i]['Kabupaten/Kota'],
                fontsize=9,
                color='black',
                weight='bold'
            )

        plt.title('Peta Pengelompokan Wilayah (Dengan Label)')
        plt.xlabel('PDRB per Kapita (Juta Rp)')
        plt.ylabel('Tingkat Kemiskinan (%)')
        plt.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig5)
        
    with col_res:
        st.subheader("Anggota Klaster")
        
        for label in ['Maju (Kota)', 'Transisi (Industri)', 'Tertinggal (Kepulauan)']:
            st.markdown(f"**{label}**")
            members = df_filtered[df_filtered['Label_Cluster'] == label]['Kabupaten/Kota'].tolist()
            st.code("\n".join(members))

# Tampilkan Profiling Rata-rata di bawah
    st.markdown("---")
    st.subheader("Profil Karakteristik Klaster")
    
    # HAPUS .reset_index() agar Label_Cluster menjadi Index (Judul Baris), bukan kolom data
    profile_df = df_filtered.groupby('Label_Cluster')[features_model].mean()
    
    # Sekarang aman untuk diformat karena isinya hanya angka
    st.dataframe(profile_df.style.format("{:.2f}").background_gradient(cmap="Greens"))
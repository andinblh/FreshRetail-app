import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.express as px
import os

# === Load Model dan File History ===
model = joblib.load("model_rf.pkl")
history_file = "riwayat_prediksi.csv"

st.set_page_config(page_title="Ritel Segar", page_icon="🛒", layout="wide")

# === Sidebar Navigasi ===
with st.sidebar:
    st.sidebar.title("🧺 Panel Admin Toko Segar")
    st.sidebar.markdown("Gunakan menu ini untuk memilih aksi.")
    st.sidebar.markdown("---")
    st.sidebar.info("Halaman ini hanya untuk admin.")

    if "page" not in st.session_state:
        st.session_state.page = "Prediksi"

    if st.sidebar.button("📊 Prediksi"):
        st.session_state.page = "Prediksi"

    if st.sidebar.button("🕘 Riwayat Prediksi"):
        st.session_state.page = "Riwayat"

# ======================================================================
# ======================== HALAMAN PREDIKSI ============================
# ======================================================================

if st.session_state.page == "Prediksi":
    st.markdown("## 🥦 Dashboard Prediksi Fresh Retail")
    st.write("Masukkan variabel yang memengaruhi penjualan dan dapatkan hasil prediksi")

    col1, col2 = st.columns(2)
    today = datetime.today().date()
    with col1:
        start_date = st.date_input("Tanggal Mulai Prediksi", value=today)
    with col2:
        end_date = st.date_input("Tanggal Akhir Prediksi", value=today + timedelta(days=6))

    if start_date > end_date:
        st.warning("Tanggal mulai tidak boleh lebih dari tanggal akhir.")
    elif st.button("Prediksi Sekarang"):
        dates = pd.date_range(start=start_date, end=end_date)
        df_pred = pd.DataFrame({'dt': dates})
        df_pred['day'] = df_pred['dt'].dt.day
        df_pred['month'] = df_pred['dt'].dt.month
        df_pred['dayofweek'] = df_pred['dt'].dt.dayofweek

        # Prediksi
        X_pred = df_pred[['day', 'month', 'dayofweek']]
        predictions = model.predict(X_pred)  # ← ini penting
        df_pred['Prediksi Penjualan'] = np.round(predictions).astype(int)
        df_pred['tanggal_prediksi'] = datetime.now()

        # Tampilkan hasil
        st.success("✅ Berhasil memprediksi penjualan!")
        st.balloons()
        df_pred['Tanggal'] = df_pred['dt'].dt.date
        st.dataframe(df_pred[['Tanggal', 'Prediksi Penjualan']])

        # Grafik
        st.subheader("📊 Grafik Hasil Prediksi")
        fig = px.line(df_pred, x='dt', y='Prediksi Penjualan', color='tanggal_prediksi',
                      title="Prediksi dari Berbagai Tanggal", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        # Simpan ke riwayat
        riwayat_data = df_pred[['Tanggal', 'Prediksi Penjualan']].copy()
        riwayat_data['Tanggal Prediksi Dibuat'] = df_pred['tanggal_prediksi'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Cek apakah file sudah ada
        if os.path.exists(history_file):
            df_history = pd.read_csv(history_file)
            df_history = pd.concat([df_history, riwayat_data], ignore_index=True)
        else:
            df_history = riwayat_data

        df_history.to_csv(history_file, index=False)

# ======================================================================
# ======================== RIWAYAT PREDIKSI ============================
# ======================================================================

elif st.session_state.page == "Riwayat":
    st.markdown("## 🕘 Riwayat Prediksi Penjualan")

    if os.path.exists(history_file):
        df_history = pd.read_csv(history_file)

        # Konversi kolom tanggal
        df_history['Tanggal'] = pd.to_datetime(df_history['Tanggal'])
        df_history['Tanggal Prediksi Dibuat'] = pd.to_datetime(df_history['Tanggal Prediksi Dibuat'])

        # Rename kolom agar sesuai grafik sebelumnya
        df_history = df_history.rename(columns={
            'Tanggal': 'dt',
            'Tanggal Prediksi Dibuat': 'tanggal_prediksi'
        })

        # TABEL RIWAYAT
        st.markdown("### 📋 Tabel Riwayat Prediksi Penjualan")
        st.dataframe(df_history, use_container_width=True)

        # GRAFIK
        st.markdown("### 📊 Grafik Riwayat")
        st.markdown("**Prediksi dari Berbagai Tanggal**")

        fig = px.line(
            df_history,
            x='dt',
            y='Prediksi Penjualan',
            color='tanggal_prediksi',
            markers=True,
            title="Prediksi Penjualan dari Berbagai Tanggal"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tombol hapus
        st.subheader("🧹 Opsi Riwayat")
        if st.button("🗑️ Hapus Semua Riwayat"):
            os.remove(history_file)
            st.success("Semua riwayat berhasil dihapus. Silakan refresh halaman.")
    else:
        st.info("Belum ada data riwayat prediksi yang tersimpan.")


    
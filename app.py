import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================

st.set_page_config(
    page_title="Prediksi Kepadatan Wisata Jeju",
    layout="wide"
)

st.title("🏝️ Prediksi Kepadatan Wisata Jeju")
st.write(
    "Analisis Faktor yang Mempengaruhi Kepadatan Wisata "
    "Menggunakan Multiple Linear Regression"
)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    arrivals = pd.read_csv(
        "1_Daily_Tourism_Arrivals_Jeju_2019-2025.csv"
    )

    weather = pd.read_csv(
        "4_Extreme_Weather_Indicators_Jeju_2019-2025.csv"
    )

    weather = weather[
        weather['Station_name'] == 'Jeju (184)'
    ].copy()

    arrivals['Date'] = pd.to_datetime(
        arrivals['Date']
    )

    weather['Date'] = pd.to_datetime(
        weather['Date']
    )

    df = pd.merge(
        arrivals,
        weather,
        on='Date',
        how='inner'
    )

    df = df.rename(columns={
        'Heatwave (>=33C, 0/1)': 'Heatwave',
        'Heavyrain (>=80mm, 0/1)': 'HeavyRain',
        'Strongwind (>=14m/s, 0/1)': 'StrongWind'
    })

    # Feature Engineering

    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter

    df['Weekend'] = np.where(
        df['DayOfWeek'] >= 5,
        1,
        0
    )

    # Lag Feature

    df = df.sort_values('Date')

    df['Lag_1'] = df['Total_arrivals'].shift(1)
    df['Lag_7'] = df['Total_arrivals'].shift(7)

    df = df.dropna()

    return df

df = load_data()

# =====================================================
# FITUR DAN TARGET
# =====================================================

X = df[
[
    'Heatwave',
    'HeavyRain',
    'StrongWind',
    'Month',
    'Year',
    'Day',
    'DayOfWeek',
    'Quarter',
    'Weekend',
    'Lag_1',
    'Lag_7'
]
]

y = df['Total_arrivals']

# =====================================================
# INFO DATASET
# =====================================================

st.subheader("📁 Informasi Dataset")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Jumlah Data",
        len(df)
    )

with col2:
    st.metric(
        "Jumlah Fitur",
        len(X.columns)
    )

# =====================================================
# TRAIN MODEL
# =====================================================

st.markdown("---")

st.subheader("⚙️ Training Model")

if st.button("Train Model"):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    st.session_state["model"] = model
    st.session_state["mae"] = mae
    st.session_state["rmse"] = rmse
    st.session_state["r2"] = r2
    st.session_state["y_test"] = y_test
    st.session_state["y_pred"] = y_pred

    st.success("Model berhasil dilatih!")

# =====================================================
# EVALUASI MODEL
# =====================================================

if "model" in st.session_state:

    st.markdown("---")

    st.subheader("📊 Evaluasi Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "MAE",
            f"{st.session_state['mae']:,.2f}"
        )

    with col2:
        st.metric(
            "RMSE",
            f"{st.session_state['rmse']:,.2f}"
        )

    with col3:
        st.metric(
            "R²",
            f"{st.session_state['r2']:.4f}"
        )

    st.info(
        f"""
        Model mampu menjelaskan sekitar
        {st.session_state['r2']*100:.2f}%
        variasi jumlah wisatawan.
        """
    )

# =====================================================
# GRAFIK AKTUAL VS PREDIKSI
# =====================================================

if "model" in st.session_state:

    st.markdown("---")

    st.subheader("📈 Grafik Aktual vs Prediksi")

    chart_df = pd.DataFrame({
        "Aktual":
        st.session_state["y_test"].values[:100],

        "Prediksi":
        st.session_state["y_pred"][:100]
    })

    st.line_chart(chart_df)

# =====================================================
# INPUT PREDIKSI
# =====================================================

st.markdown("---")

st.subheader("🔮 Prediksi Wisatawan")

tanggal = st.date_input(
    "Tanggal"
)

heatwave = st.selectbox(
    "Heatwave",
    [0, 1]
)

heavyrain = st.selectbox(
    "Heavy Rain",
    [0, 1]
)

strongwind = st.selectbox(
    "Strong Wind",
    [0, 1]
)

# =====================================================
# PREDIKSI
# =====================================================

if "model" in st.session_state:

    if st.button("Prediksi"):

        tanggal = pd.to_datetime(
            tanggal
        )

        month = tanggal.month
        year = tanggal.year
        day = tanggal.day
        dayofweek = tanggal.dayofweek

        quarter = (
            (month - 1) // 3
        ) + 1

        weekend = (
            1 if dayofweek >= 5
            else 0
        )

        lag_1 = df[
            'Total_arrivals'
        ].iloc[-1]

        lag_7 = df[
            'Total_arrivals'
        ].iloc[-7]

        input_user = pd.DataFrame(
            [[
                heatwave,
                heavyrain,
                strongwind,
                month,
                year,
                day,
                dayofweek,
                quarter,
                weekend,
                lag_1,
                lag_7
            ]],
            columns=X.columns
        )

        hasil = st.session_state[
            "model"
        ].predict(
            input_user
        )

        st.success(
            f"""
            Prediksi Jumlah Wisatawan:
            {hasil[0]:,.0f} orang
            """
        )

else:

    st.warning(
        "Silakan klik Train Model terlebih dahulu."
    )
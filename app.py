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
# JUDUL
# =====================================================

st.set_page_config(
    page_title="Prediksi Kepadatan Wisata Jeju",
    layout="wide"
)

st.title("🏝️ Prediksi Kepadatan Wisata Jeju")
st.markdown(
    "### Analisis Faktor yang Mempengaruhi Kepadatan Wisata Menggunakan Multiple Linear Regression"
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
# SPLIT DATA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# TRAIN MODEL
# =====================================================

@st.cache_resource
def train_model():

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    return model

model = train_model()

# =====================================================
# EVALUASI
# =====================================================

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

# =====================================================
# SIDEBAR INPUT
# =====================================================

st.sidebar.header("Input Prediksi")

tanggal = st.sidebar.date_input(
    "Tanggal"
)

heatwave = st.sidebar.selectbox(
    "Heatwave",
    [0, 1]
)

heavyrain = st.sidebar.selectbox(
    "Heavy Rain",
    [0, 1]
)

strongwind = st.sidebar.selectbox(
    "Strong Wind",
    [0, 1]
)

# =====================================================
# EKSTRAK TANGGAL
# =====================================================

tanggal = pd.to_datetime(tanggal)

month = tanggal.month
year = tanggal.year
day = tanggal.day
dayofweek = tanggal.dayofweek

quarter = ((month - 1) // 3) + 1

weekend = 1 if dayofweek >= 5 else 0

# =====================================================
# DATA HISTORIS
# =====================================================

lag_1 = df['Total_arrivals'].iloc[-1]
lag_7 = df['Total_arrivals'].iloc[-7]

# =====================================================
# PREDIKSI
# =====================================================

if st.button("Prediksi Jumlah Wisatawan"):

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

    hasil = model.predict(
        input_user
    )

    st.success(
        f"Prediksi Jumlah Wisatawan : {hasil[0]:,.0f} orang"
    )

# =====================================================
# EVALUASI MODEL
# =====================================================

st.markdown("---")

st.subheader("📊 Evaluasi Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        f"{mae:,.2f}"
    )

with col2:
    st.metric(
        "RMSE",
        f"{rmse:,.2f}"
    )

with col3:
    st.metric(
        "R²",
        f"{r2:.4f}"
    )

st.info(
    f"""
    Model mampu menjelaskan sekitar
    {r2*100:.2f}% variasi jumlah wisatawan.
    """
)

# =====================================================
# INFORMASI DATASET
# =====================================================

st.markdown("---")

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

st.write("Fitur yang digunakan:")

st.write(list(X.columns))

# =====================================================
# KOEFISIEN REGRESI
# =====================================================

st.markdown("---")

st.subheader("📈 Koefisien Regresi")

coef_df = pd.DataFrame({
    'Variabel': X.columns,
    'Koefisien': model.coef_
})

coef_df = coef_df.sort_values(
    by='Koefisien',
    key=abs,
    ascending=False
)

st.dataframe(
    coef_df,
    use_container_width=True
)

# =====================================================
# VISUALISASI KOEFISIEN
# =====================================================

fig, ax = plt.subplots(figsize=(10,5))

ax.bar(
    coef_df['Variabel'],
    coef_df['Koefisien']
)

plt.xticks(rotation=45)

plt.title(
    "Pengaruh Faktor Terhadap Jumlah Wisatawan"
)

st.pyplot(fig)

# =====================================================
# AKTUAL VS PREDIKSI
# =====================================================

st.markdown("---")

st.subheader("📉 Aktual vs Prediksi")

fig2, ax2 = plt.subplots(figsize=(6,6))

ax2.scatter(
    y_test,
    y_pred
)

ax2.set_xlabel("Aktual")

ax2.set_ylabel("Prediksi")

ax2.set_title(
    "Aktual vs Prediksi"
)

st.pyplot(fig2)
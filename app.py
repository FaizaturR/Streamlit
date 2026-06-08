import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# =====================================================
# JUDUL
# =====================================================

st.title("Prediksi Kepadatan Wisata Jeju")
st.subheader("Multiple Linear Regression")

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

    df = df.sort_values('Date')

    df['Lag_1'] = df['Total_arrivals'].shift(1)
    df['Lag_7'] = df['Total_arrivals'].shift(7)

    df = df.dropna()

    return df

df = load_data()

# =====================================================
# FITUR
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
# TRAIN MODEL
# =====================================================

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

# =====================================================
# SIDEBAR
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

if st.button("Prediksi"):

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
        f"Prediksi Jumlah Wisatawan : "
        f"{hasil[0]:,.0f} orang"
    )

# =====================================================
# INFORMASI MODEL
# =====================================================

st.markdown("---")

st.write("Jumlah Data :", len(df))

st.write("Jumlah Fitur :", len(X.columns))

st.write("Fitur yang digunakan:")

st.write(list(X.columns))
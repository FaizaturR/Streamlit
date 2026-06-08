import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================
# CACHE: LOAD & PREPROCESS DATA
# ======================
@st.cache_data
def load_and_preprocess(arrivals_path, weather_path):
    arrivals = pd.read_csv(arrivals_path)
    weather = pd.read_csv(weather_path)
    
    # Filter stasiun Jeju
    weather = weather[weather['Station_name'] == 'Jeju (184)'].copy()
    
    # Konversi tanggal
    arrivals['Date'] = pd.to_datetime(arrivals['Date'])
    weather['Date'] = pd.to_datetime(weather['Date'])
    
    # Merge
    df = pd.merge(arrivals, weather, on='Date', how='inner')
    
    # Rename kolom ekstrem
    df = df.rename(columns={
        'Heatwave (>=33C, 0/1)': 'Heatwave',
        'Heavyrain (>=80mm, 0/1)': 'HeavyRain',
        'Strongwind (>=14m/s, 0/1)': 'Strongwind'
    })
    
    # Fitur tanggal
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    df['Weekend'] = np.where(df['DayOfWeek'] >= 5, 1, 0)
    
    # Urutkan untuk lag
    df = df.sort_values('Date')
    df['Lag_1'] = df['Total_arrivals'].shift(1)
    df['Lag_7'] = df['Total_arrivals'].shift(7)
    df = df.dropna().reset_index(drop=True)
    
    return df

@st.cache_resource
def train_model(df):
    feature_cols = [
        'Heatwave', 'HeavyRain', 'Strongwind',
        'Month', 'Year', 'Day', 'DayOfWeek', 'Quarter', 'Weekend',
        'Lag_1', 'Lag_7'
    ]
    X = df[feature_cols]
    y = df['Total_arrivals']
    
    # Train/test split untuk evaluasi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R²': r2}
    return model, feature_cols, metrics

def predict_arrival(model, feature_cols, df_history, input_date, heatwave, heavyrain, strongwind):
    month = input_date.month
    year = input_date.year
    day = input_date.day
    dayofweek = input_date.weekday()
    quarter = (month - 1) // 3 + 1
    weekend = 1 if dayofweek >= 5 else 0
    
    # Ambil lag_1 dan lag_7 dari data historis
    lag_1_date = input_date - pd.Timedelta(days=1)
    lag_7_date = input_date - pd.Timedelta(days=7)
    
    def get_lag_value(target_date):
        row = df_history[df_history['Date'] == target_date]
        if not row.empty:
            return row['Total_arrivals'].values[0]
        else:
            # Fallback ke nilai terbaru jika tidak ada data
            st.warning(f"Data untuk {target_date.date()} tidak ditemukan. Menggunakan nilai lag terbaru.")
            return df_history['Total_arrivals'].iloc[-1]
    
    lag_1 = get_lag_value(lag_1_date)
    lag_7 = get_lag_value(lag_7_date)
    
    input_data = pd.DataFrame([[heatwave, heavyrain, strongwind,
                                month, year, day, dayofweek, quarter, weekend,
                                lag_1, lag_7]], columns=feature_cols)
    prediction = model.predict(input_data)[0]
    return prediction, lag_1, lag_7

# ======================
# MAIN STREAMLIT APP
# ======================
st.set_page_config(page_title="Prediksi Wisatawan Jeju", layout="centered")
st.title("🌏 Prediksi Jumlah Wisatawan Harian - Pulau Jeju")
st.markdown("Model regresi linear dengan fitur cuaca ekstrem, komponen tanggal, dan lag 1/7 hari.")

# Upload atau input file
arrivals_file = st.text_input("Nama file kedatangan", "1_Daily_Tourism_Arrivals_Jeju_2019-2025.csv")
weather_file = st.text_input("Nama file cuaca ekstrem", "4_Extreme_Weather_Indicators_Jeju_2019-2025.csv")

if st.button("Load Data & Train Model"):
    try:
        with st.spinner("Memuat data dan melatih model..."):
            df = load_and_preprocess(arrivals_file, weather_file)
            model, feature_cols, metrics = train_model(df)
            st.session_state['df'] = df
            st.session_state['model'] = model
            st.session_state['feature_cols'] = feature_cols
            st.session_state['metrics'] = metrics
        st.success("Data berhasil dimuat dan model siap!")
    except Exception as e:
        st.error(f"Error: {e}. Pastikan file CSV ada atau unggah file manual.")

# Tampilkan performa jika model sudah ada
if 'model' in st.session_state:
    st.subheader("📊 Evaluasi Model (test split 20%)")
    m = st.session_state['metrics']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{m['MAE']:,.0f}")
    col2.metric("MSE", f"{m['MSE']:,.0f}")
    col3.metric("RMSE", f"{m['RMSE']:,.0f}")
    col4.metric("R²", f"{m['R²']:.3f}")
    
    st.subheader("🔮 Prediksi untuk tanggal baru")
    input_date = st.date_input("Tanggal kunjungan", datetime(2025, 9, 15))
    heatwave = st.radio("Heatwave (>=33°C)", [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)
    heavyrain = st.radio("Heavy Rain (>=80mm)", [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)
    strongwind = st.radio("Strong Wind (>=14m/s)", [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)
    
    if st.button("Prediksi Sekarang"):
        pred, lag1, lag7 = predict_arrival(
            st.session_state['model'],
            st.session_state['feature_cols'],
            st.session_state['df'],
            input_date, heatwave, heavyrain, strongwind
        )
        st.success(f"📈 **Prediksi jumlah wisatawan:** **{pred:,.0f} orang**")
        st.caption(f"Lag-1 yang digunakan: {lag1:,.0f} | Lag-7 yang digunakan: {lag7:,.0f}")
        
        if st.checkbox("Tampilkan koefisien fitur"):
            coef_df = pd.DataFrame({
                'Fitur': st.session_state['feature_cols'],
                'Koefisien': st.session_state['model'].coef_
            }).sort_values('Koefisien', key=abs, ascending=False)
            st.dataframe(coef_df.style.format({'Koefisien': '{:,.2f}'}))
else:
    st.info("Silakan masukkan nama file CSV dan klik 'Load Data & Train Model'.")
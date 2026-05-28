# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Estimasi Harga Mobil Bekas Cepat & Mudah",
    page_icon="🚗",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

    /* Main Container */
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Title */
    h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
    }

    /* Paragraph */
    .stMarkdown p {
        font-size: 1.05rem;
        line-height: 1.7;
    }

    /* Section Header */
    h4 {
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-weight: 700 !important;
    }

    /* Metric Card */
    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 20px;
        border-radius: 16px;
    }

    /* Button */
    .stButton > button {
        height: 3.2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 12px;
    }

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL & DATA
# ==========================================
@st.cache_resource
def load_assets():

    model_path = os.path.join(BASE_DIR, 'regresi_berganda.pkl')
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    fitur_path = os.path.join(BASE_DIR, 'fitur.pkl')
    csv_path = os.path.join(BASE_DIR, 'car_details_v4.csv')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(fitur_path, 'rb') as file:
        fitur = pickle.load(file)

    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()
    df = df.dropna().reset_index(drop=True)

    # ==========================================
    # REMOVE OUTLIER
    # ==========================================
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)

    IQR = Q3 - Q1

    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR

    df = df[
        (df['Price'] >= batas_bawah) &
        (df['Price'] <= batas_atas)
    ].reset_index(drop=True)

    # ==========================================
    # TARGET ENCODING
    # ==========================================
    target_enc_cols = ['Make', 'Model', 'Location']

    global_mean = df['Price'].mean()

    smoothing_weight = 10

    deployment_mappings = {}

    for col in target_enc_cols:

        stats = df.groupby(col)['Price'].agg(['count', 'mean'])

        counts = stats['count']
        means = stats['mean']

        smoothed_vals = (
            (counts * means + smoothing_weight * global_mean)
            / (counts + smoothing_weight)
        )

        deployment_mappings[col] = smoothed_vals.to_dict()

    unique_makes = sorted(df['Make'].unique())
    unique_locations = sorted(df['Location'].unique())

    return (
        model,
        scaler,
        fitur,
        df,
        deployment_mappings,
        unique_makes,
        unique_locations,
        global_mean
    )


(
    model,
    scaler,
    fitur,
    df,
    deployment_mappings,
    unique_makes,
    unique_locations,
    global_mean
) = load_assets()

# ==========================================
# HEADER
# ==========================================
st.title("🚗 Estimasi Harga Mobil Bekas")

st.markdown("""
Masukkan spesifikasi kendaraan Anda untuk mendapatkan estimasi harga pasar
menggunakan algoritma Machine Learning.
""")

st.divider()

# ==========================================
# SECTION 1
# ==========================================
st.markdown("#### 📋 Detail Dasar Kendaraan")

with st.container(border=True):

    sec1_col1, sec1_col2 = st.columns(2)

    with sec1_col1:

        make = st.selectbox(
            "Merek Mobil",
            unique_makes,
            index=0
        )

        filtered_models = sorted(
            df[df['Make'] == make]['Model'].unique()
        )

        model_car = st.selectbox(
            "Model Mobil",
            filtered_models,
            index=0
        )

        year = st.number_input(
            "Tahun Pembuatan",
            min_value=1990,
            max_value=2026,
            step=1,
            value=2018
        )

    with sec1_col2:

        kilometer = st.number_input(
            "Jarak Tempuh (Km)",
            min_value=0,
            step=1000,
            value=50000
        )

        location = st.selectbox(
            "Lokasi Penjualan",
            unique_locations,
            index=0
        )

        color = st.selectbox(
            "Warna Mobil",
            [
                'White',
                'Silver',
                'Grey',
                'Black',
                'Blue',
                'Red',
                'Brown',
                'Maroon',
                'Gold',
                'Bronze',
                'Green',
                'Orange',
                'Yellow',
                'Purple',
                'Others'
            ],
            index=0
        )

st.write("")

# ==========================================
# SECTION 2
# ==========================================
st.markdown("#### ⚙️ Spesifikasi Mesin")

with st.container(border=True):

    sec2_col1, sec2_col2 = st.columns(2)

    with sec2_col1:

        engine = st.number_input(
            "Kapasitas Mesin (cc)",
            min_value=500,
            max_value=8000,
            step=100,
            value=1200
        )

        power = st.number_input(
            "Tenaga Maksimal (bhp)",
            min_value=30.0,
            max_value=800.0,
            step=1.0,
            value=85.0
        )

        torque = st.number_input(
            "Torsi Maksimal (Nm)",
            min_value=50.0,
            max_value=1000.0,
            step=1.0,
            value=115.0
        )

    with sec2_col2:

        fuel = st.selectbox(
            "Jenis Bahan Bakar",
            [
                'Petrol',
                'Diesel',
                'CNG',
                'LPG',
                'CNG + CNG',
                'Petrol + CNG'
            ],
            index=0
        )

        transmission = st.selectbox(
            "Transmisi",
            ['Manual', 'Automatic'],
            index=0
        )

        drivetrain = st.selectbox(
            "Penggerak",
            ['FWD', 'RWD', 'AWD/4WD'],
            index=0
        )

st.write("")

# ==========================================
# SECTION 3
# ==========================================
st.markdown("#### 📏 Dimensi & Riwayat Kendaraan")

with st.container(border=True):

    sec3_col1, sec3_col2 = st.columns(2)

    with sec3_col1:

        length = st.number_input(
            "Panjang (mm)",
            min_value=2000.0,
            max_value=6000.0,
            value=3990.0
        )

        width = st.number_input(
            "Lebar (mm)",
            min_value=1000.0,
            max_value=2500.0,
            value=1680.0
        )

        height = st.number_input(
            "Tinggi (mm)",
            min_value=1000.0,
            max_value=2500.0,
            value=1500.0
        )

    with sec3_col2:

        seating = st.number_input(
            "Kapasitas Penumpang",
            min_value=2,
            max_value=14,
            step=1,
            value=5
        )

        tank = st.number_input(
            "Kapasitas Tangki (Liter)",
            min_value=15.0,
            max_value=100.0,
            value=35.0
        )

        owner = st.selectbox(
            "Kepemilikan Ke-",
            ['First', 'Second', 'Third', 'UnRegistered Car'],
            index=0
        )

        seller = st.selectbox(
            "Tipe Penjual",
            ['Individual', 'Corporate'],
            index=0
        )

# ==========================================
# BUTTON
# ==========================================
st.divider()

if st.button(
    "🔍 Hitung Estimasi Harga",
    use_container_width=True,
    type="primary"
):

    # ==========================================
    # CREATE INPUT
    # ==========================================
    input_data = pd.DataFrame(
        columns=fitur,
        data=np.zeros((1, len(fitur)))
    )

    # ==========================================
    # TARGET ENCODING
    # ==========================================
    input_data['Make'] = deployment_mappings['Make'].get(
        make,
        global_mean
    )

    input_data['Model'] = deployment_mappings['Model'].get(
        model_car,
        global_mean
    )

    input_data['Location'] = deployment_mappings['Location'].get(
        location,
        global_mean
    )

    # ==========================================
    # NUMERICAL
    # ==========================================
    input_data['Year'] = year
    input_data['Kilometer'] = kilometer
    input_data['Length'] = length
    input_data['Width'] = width
    input_data['Height'] = height
    input_data['Seating Capacity'] = seating
    input_data['Fuel Tank Capacity'] = tank
    input_data['Power_num'] = power
    input_data['Torque_num'] = torque
    input_data['Engine_num'] = engine

    # ==========================================
    # ONE HOT ENCODING
    # ==========================================
    fuel_col = f'Fuel Type_{fuel}'

    if fuel_col in input_data.columns:
        input_data[fuel_col] = 1

    if transmission == 'Manual':
        if 'Transmission_Manual' in input_data.columns:
            input_data['Transmission_Manual'] = 1

    color_col = f'Color_{color}'

    if color_col in input_data.columns:
        input_data[color_col] = 1

    owner_col = f'Owner_{owner}'

    if owner_col in input_data.columns:
        input_data[owner_col] = 1

    seller_col = f'Seller Type_{seller}'

    if seller_col in input_data.columns:
        input_data[seller_col] = 1

    dt_col = f'Drivetrain_{drivetrain}'

    if dt_col in input_data.columns:
        input_data[dt_col] = 1

    # ==========================================
    # SCALING
    # ==========================================
    scaled_data = scaler.transform(input_data)

    # ==========================================
    # PREDICTION
    # ==========================================
    prediction = model.predict(scaled_data)

    estimasi_harga = int(prediction[0])

    st.write("")

    # ==========================================
    # OUTPUT
    # ==========================================
    if estimasi_harga < 0:

        st.error(
            "⚠️ Kombinasi data menghasilkan estimasi tidak valid. "
            "Silakan periksa kembali input kendaraan."
        )

    else:

        st.success("Estimasi harga berhasil dikalkulasi!")

        st.metric(
            label="Estimasi Harga Mobil (INR)",
            value=f"₹ {estimasi_harga:,}"
        )
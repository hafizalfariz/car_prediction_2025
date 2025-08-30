import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
from PIL import Image
import base64
from io import BytesIO

# --- Load pipeline dan model ---
with open("./src/preprocessing_pipeline.pkl", "rb") as f:
    preprocessing_pipeline = cloudpickle.load(f)
with open("./src/best_xgb_rand.pkl", "rb") as f:
    model = cloudpickle.load(f)

def image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode()

# --- Tampilkan logo terpusat dengan ukuran fleksibel ---
def show_centered_logo(image_path: str, caption: str = "", width: int = 300):
    try:
        img = Image.open(image_path)
        img_base64 = image_to_base64(img)

        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{img_base64}" width="{width}"/>
                <p><em>{caption}</em></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"⚠️ Failed to load image: {e}")



def run():
    # Menampilkan logo
    show_centered_logo("./src/Logo2.PNG", "Cars Prediction 2025 by Hafiz Alfariz", width=700)

    # Header
    st.markdown("<h1 style='text-align: center;'>CARS PREDICTION 2025</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><em>Estimate car prices based on technical specifications</em></h5>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Form Input ---
    with st.form(key='cars_predictions_2025'):
        company_names = st.text_input('Company Name')
        cars_names = st.text_input('Car Model')
        engines = st.text_input('Engine Type')
        horsepower = st.number_input('Horsepower (hp)', min_value=22.0, max_value=1200.0, value=100.0, step=1.0)
        total_speed = st.number_input('Top Speed (km/h)', min_value=60.0, max_value=500.0, value=100.0, step=1.0)
        performance0_100_km_h = st.slider('0–100 km/h Performance (sec)', min_value=1.5, max_value=22.0, value=10.0, step=0.1)
        fuel_types = st.selectbox('Fuel Type', options=[
            'Petrol', 'Diesel', 'Electric', 'Hybrid', 'Alternative', 'Other_Fuel'
        ])
        seats = st.number_input('Number of Seats', min_value=1, max_value=20, value=5, step=1)
        torque = st.number_input('Torque (Nm)', min_value=40.5, max_value=1400.0, value=100.0, step=1.0)
        engine_cc = st.number_input('Engine CC', min_value=0.0, max_value=9200.5, step=0.1, help='Enter 0 if electric')
        battery_capacity_kwh = st.number_input('Battery Capacity (kWh)', min_value=0.0, max_value=215.0, step=0.1, help='Enter 0 if not electric')

        # --- Currency Conversion ---
        currency = st.selectbox("Currency Output", options=["USD", "IDR", "EUR", "JPY", "Other"], key="currency_select")
        exchange_rate = st.number_input(
            "Exchange Rate (1 USD to selected currency)",
            min_value=0.0,
            max_value=100000.0,
            help="Enter the current exchange rate manually",
            key="exchange_rate_input"
        )

        submit_button = st.form_submit_button('🔍 Predict')

    # --- Prediction Logic ---
    if submit_button:
        try:
            # Validasi exchange rate hanya jika currency ≠ USD
            if currency != "USD" and exchange_rate == 0.0:
                st.warning("⚠️ Please enter a valid exchange rate for currency conversion.")

            # Buat dataframe input
            raw_data = pd.DataFrame([{
                'company_names': company_names,
                'cars_names': cars_names,
                'engines': engines,
                'horsepower': horsepower,
                'total_speed': total_speed,
                'performance0__100_km_h': performance0_100_km_h,
                'fuel_types': fuel_types,
                'seats': seats,
                'torque': torque,
                'engine_cc': engine_cc,
                'battery_capacity_kwh': battery_capacity_kwh,
                'cars_prices': 0  # dummy target for pipeline compatibility
            }])

            # Transform input using pipeline
            X_new = preprocessing_pipeline.transform(raw_data)
            pred_log = model.predict(X_new)
            pred_price = np.expm1(pred_log)
            raw_data['predicted_price_usd'] = pred_price

            # Konversi ke mata uang pilihan
            converted_price = pred_price * exchange_rate
            currency_symbol = {
                "USD": "$",
                "IDR": "Rp",
                "EUR": "€",
                "JPY": "¥",
                "Other": ""
            }.get(currency, "")

            raw_data['predicted_price_converted'] = converted_price
            raw_data['predicted_price_converted'] = raw_data['predicted_price_converted'].apply(lambda x: f"{currency_symbol} {x:,.0f}")
            raw_data['predicted_price_usd'] = raw_data['predicted_price_usd'].apply(lambda x: f"$ {x:,.0f}")

            # Format fitur numerik
            raw_data['horsepower'] = raw_data['horsepower'].apply(lambda x: f"{x:.0f} hp")
            raw_data['torque'] = raw_data['torque'].apply(lambda x: f"{x:.0f} Nm")
            raw_data['total_speed'] = raw_data['total_speed'].apply(lambda x: f"{x:.0f} km/h")
            raw_data['engine_cc'] = raw_data['engine_cc'].apply(lambda x: f"{x:.0f} cc")
            raw_data['battery_capacity_kwh'] = raw_data['battery_capacity_kwh'].apply(lambda x: f"{x:.1f} kWh")
            raw_data['performance0__100_km_h'] = raw_data['performance0__100_km_h'].apply(lambda x: f"{x:.1f} sec")

            # Tampilkan hasil utama
            if currency != "USD" and exchange_rate > 0:
                st.success(f"✅ Estimated Car Price: {raw_data['predicted_price_converted'].iloc[0]} ({currency})")
                st.markdown(f"💱 Exchange Rate Used: **1 USD = {exchange_rate:,.0f} {currency}**")
            else:
                st.success(f"✅ Estimated Car Price: {raw_data['predicted_price_usd'].iloc[0]} (USD)")

            # Tampilkan tabel hasil
            display_cols = [
                'company_names', 'cars_names', 'engines', 'horsepower', 'total_speed',
                'performance0__100_km_h', 'fuel_types', 'seats', 'torque',
                'engine_cc', 'battery_capacity_kwh', 'predicted_price_usd'
            ]
            if currency != "USD" and exchange_rate > 0:
                display_cols.append('predicted_price_converted')

            st.subheader("📋 Prediction Result")
            st.dataframe(raw_data[display_cols])

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# --- Run App ---
if __name__ == "__main__":
    run()
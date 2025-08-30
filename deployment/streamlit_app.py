import streamlit as st
import edatest
import prediction

# Konfigurasi halaman
st.set_page_config(
    page_title='Car Prediction',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Sidebar untuk memilih halaman
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    'Choose Page:',
    ('EDA', 'Prediction')
)

# Jalankan halaman yang dipilih
if page == 'EDA':
    edatest.run()
elif page == 'Prediction':
    prediction.run()

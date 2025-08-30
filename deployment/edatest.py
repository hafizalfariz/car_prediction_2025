import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

def run():

     # Membuat Title
    st.markdown('<h1 style="text-align: center;">Car Price Prediction 2025</h1>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Exploratory Data Analysis of the 2025 Car Dataset</h3>", unsafe_allow_html=True)

    # Membuat sub header

    st.markdown('<h3 style="text-align: center;">This page contains exploratory analysis and visualizations</h3>', unsafe_allow_html=True)

    # Menambahkan Teks
    st.markdown('<h5 style="text-align: center;">This page was created by <em>Hafiz Alfariz</em></h5>', unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    #menambahkan gambar
    gambar = Image.open('./src/Logo.PNG')
    st.image(gambar, caption='EDA Cars 2025')

    # Menambahkan Teks
    st.write('# Exploratory Data Analysis')

    # Menampilkan dataframe
    df = pd.read_csv('./src/cars_cleaned.csv')

    df_display = df.copy()

    # Format kolom dengan satuan
    df_display['horsepower'] = df_display['horsepower'].apply(lambda x: f"{x:.0f} hp")
    df_display['engine_cc'] = df_display['engine_cc'].apply(lambda x: f"{x:.0f} cc")
    df_display['battery_capacity_kwh'] = df_display['battery_capacity_kwh'].apply(lambda x: f"{x:.1f} kWh")
    df_display['cars_prices'] = df_display['cars_prices'].apply(lambda x: f"${x:,.0f}")
    df_display['total_speed'] = df_display['total_speed'].apply(lambda x: f"{x:.0f} km/h")
    df_display['performance0__100_km_h'] = df_display['performance0__100_km_h'].apply(lambda x: f"{x:.1f} sec")
    df_display['torque'] = df_display['torque'].apply(lambda x: f"{x:.0f} Nm")

    # Tampilkan di Streamlit
    st.write('### Cars Dataset')
    st.dataframe(df_display)

    
    # --- Subplot setup ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # === 1. Company Names (Top 15) ===
    company_counts = df['company_names'].value_counts().head(15)
    colors1 = sns.color_palette("viridis", len(company_counts))
    axes[0, 0].barh(
        range(len(company_counts)),
        company_counts.values,
        color=colors1,
        edgecolor='white'
    )
    axes[0, 0].set_title('Top 15 Car Companies in Dataset', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Number of Cars')
    axes[0, 0].set_yticks(range(len(company_counts)))
    axes[0, 0].set_yticklabels(company_counts.index)
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(company_counts.values):
        axes[0, 0].text(v + 1, i, str(v), va='center', fontsize=10)

    # === 2. Fuel Types Distribution ===
    fuel_counts = df['fuel_types'].value_counts()
    colors2 = sns.color_palette("plasma", len(fuel_counts))
    axes[0, 1].pie(
        fuel_counts.values,
        labels=fuel_counts.index,
        autopct='%1.1f%%',
        colors=colors2,
        startangle=90,
        textprops={'fontsize': 10}
    )
    axes[0, 1].set_title('Fuel Types Distribution', fontsize=14, fontweight='bold')

    # === 3. Engine Types (Top 10) ===
    engine_counts = df['engines'].value_counts().head(10)
    colors3 = sns.color_palette("cubehelix", len(engine_counts))
    axes[1, 0].bar(
        range(len(engine_counts)),
        engine_counts.values,
        color=colors3,
        edgecolor='white'
    )
    axes[1, 0].set_title('Top 10 Engine Types', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Engine Types')
    axes[1, 0].set_ylabel('Number of Cars')
    axes[1, 0].set_xticks(range(len(engine_counts)))
    axes[1, 0].set_xticklabels(engine_counts.index, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(engine_counts.values):
        axes[1, 0].text(i, v + 1, str(v), ha='center', va='bottom', fontsize=10)

    # === 4. Seats Distribution ===
    seats_counts = df['seats'].value_counts().sort_index()
    colors4 = sns.color_palette("mako", len(seats_counts))
    axes[1, 1].bar(
        seats_counts.index,
        seats_counts.values,
        color=colors4,
        edgecolor='white'
    )
    axes[1, 1].set_title('Number of Seats Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Seats')
    axes[1, 1].set_ylabel('Number of Cars')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in zip(seats_counts.index, seats_counts.values):
        axes[1, 1].text(i, v + 2, str(v), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('<hr>', unsafe_allow_html=True)

    # === MARKET SEGMENTATION ANALYSIS ===
    st.write('## Market Segmentation by Price')

    # --- Segmentasi Harga ---
    def categorize_price_segment(price):
        if price < 30000:
            return 'Budget'
        elif price < 100000:
            return 'Mid-range'
        elif price < 300000:
            return 'Premium'
        else:
            return 'Luxury'

    df['Price_Segment'] = df['cars_prices'].apply(categorize_price_segment)

    # --- Market Share Analysis ---
    segment_analysis = df['Price_Segment'].value_counts()
    segment_percentage = (segment_analysis / len(df) * 100).round(1)

    st.write('### Market Share by Price Segment')
    for segment, count in segment_analysis.items():
        percentage = segment_percentage[segment]
        st.write(f"- **{segment}**: {count} models ({percentage}%)")

    # --- Statistik Harga per Segmen ---
    segment_avg_price = df.groupby('Price_Segment')['cars_prices'].agg(['mean', 'median', 'count']).round(0)

    # Format kolom mean & median jadi string dengan simbol $
    segment_stats_display = segment_avg_price.copy()
    segment_stats_display['mean'] = segment_stats_display['mean'].apply(lambda x: f"${x:,.0f}")
    segment_stats_display['median'] = segment_stats_display['median'].apply(lambda x: f"${x:,.0f}")

    # Rename kolom agar lebih komunikatif
    segment_stats_display = segment_stats_display.rename(columns={
        'mean': 'Average Price',
        'median': 'Median Price',
        'count': 'Model Count'
    })

    st.write('### Price Statistics by Segment')
    st.dataframe(segment_stats_display)

    # --- Visualisasi ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Pie Chart Market Share
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    explode = [0.05] * len(segment_percentage)

    wedges, texts, autotexts = ax1.pie(
        segment_percentage.values,
        labels=segment_percentage.index,
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        textprops={'fontsize': 9}
    )
    ax1.set_title('Market Share by Price Segment', fontsize=12, fontweight='bold')

    # Bar Chart Average Price
    segment_avg_price['mean'].plot(
        kind='bar',
        ax=ax2,
        color=colors,
        alpha=0.85
    )
    ax2.set_title('Average Price by Segment', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Price ($)')
    ax2.set_xlabel('Price Segment')
    ax2.tick_params(axis='x', rotation=45)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('<hr>', unsafe_allow_html=True)
    # Membuat bar plot

    st.write('### Distribution')
    # --- Daftar kolom numerik yang tersedia ---
    numerical_cols = [
        'horsepower', 'total_speed', 'performance0__100_km_h',
        'cars_prices', 'seats', 'torque', 'engine_cc', 'battery_capacity_kwh'
    ]

    # Filter hanya kolom yang ada di df
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    # --- Pilihan kolom oleh user ---
    selected_cols = st.multiselect(
        'Select columns to display:',
        options=numerical_cols,
        default=['cars_prices']
    )

    # --- Warna gradasi yang konsisten ---
    color_palette = sns.color_palette("plasma", len(selected_cols))

    # --- Plot histogram ---
    n_cols = 2
    n_rows = (len(selected_cols) + 1) // n_cols

    fig = plt.figure(figsize=(12, 5 * n_rows))

    for i, col in enumerate(selected_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], bins=30, kde=True, color=color_palette[i - 1])
        plt.title(f'Distribution: {col}', fontsize=12, fontweight='bold')
        plt.xlabel(col)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)


    # --- Agregasi Brand Metrics ---
    brand_metrics = df.groupby('company_names').agg({
        'cars_prices': ['mean', 'median', 'count'],
        'horsepower': 'mean',
        'torque': 'mean'
    }).round(0)

    brand_metrics.columns = ['Avg_Price', 'Median_Price', 'Model_Count', 'Avg_HP', 'Avg_Torque']

    # Filter brand dengan minimal 3 model
    significant_brands = brand_metrics[brand_metrics['Model_Count'] >= 3].copy()
    significant_brands = significant_brands.sort_values('Avg_Price', ascending=False)


    st.markdown('<hr>', unsafe_allow_html=True)
    # --- Visualisasi di Streamlit ---
    st.write('### Top 15 Premium Brands with Average Price')
    
    top_brands = significant_brands.head(15).reset_index()

    # Warna custom terinspirasi dari brand mobil top
    custom_colors = [
        '#C70039', '#0033A0', '#A2AAAD', '#FFD700', '#000000',
        '#6E6E6E', '#B71C1C', '#F8F8FF', '#005A2B', '#1B365D',
        '#FF5733', '#34495E', '#7D3C98', '#2ECC71', '#E67E22'
    ]

    fig = plt.figure(figsize=(12, 5))
    sns.barplot(
        data=top_brands,
        x='company_names',
        y='Avg_Price',
        palette=custom_colors,
        alpha=0.85
    )

    plt.title('Top 15 Premium Brands - Average Price', fontsize=12, fontweight='bold')
    plt.ylabel('Average Price ($)')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.tight_layout()

    # Tambahkan label harga di atas bar
    for i, row in top_brands.iterrows():
        plt.text(i, row['Avg_Price'] + 500, f"${int(row['Avg_Price']):,}", ha='center', fontsize=9)

    st.pyplot(fig)

    st.markdown('<hr>', unsafe_allow_html=True)
    # --- Tampilkan tabel ringkasan ---
    formatted_df = top_brands.copy()
    formatted_df['Avg_Price'] = formatted_df['Avg_Price'].apply(lambda x: f"${x:,.0f}")
    formatted_df['Avg_HP'] = formatted_df['Avg_HP'].apply(lambda x: f"{x:.0f} hp")

    # Tampilkan di Streamlit
    st.write('#### Ringkasan Brand Premium')
    st.dataframe(formatted_df[['company_names', 'Avg_Price', 'Model_Count', 'Avg_HP']])

    # membuat plotly plot
    st.write('### Average Price by Brand')
    fig = px.scatter(df, x='cars_prices', y='company_names', hover_data=['company_names', 'cars_prices'])
    st.plotly_chart(fig)


if __name__ == "__main__":
    run()
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(
    page_title="Parcl | Buyer Segmentation & Market Intelligence",
    page_icon="🏢",
    layout="wide"
)

# ─── DATA LOADER / FALLBACK GENERATOR ───
@st.cache_data
def load_data():
    if os.path.exists('segmented_buyers.csv'):
        return pd.read_csv('segmented_buyers.csv')
    
    # Generate on the fly if CSV not yet created
    if not os.path.exists('clients.csv') or not os.path.exists('properties.csv'):
        st.error("Error: 'clients.csv' and 'properties.csv' must be present in this folder.")
        st.stop()
        
    clients = pd.read_csv('clients.csv')
    properties = pd.read_csv('properties.csv')
    
    properties['sale_price_clean'] = (
        properties['sale_price']
        .astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .astype(float)
    )
    
    sold_props = properties[properties['listing_status'].str.lower() == 'sold']
    client_metrics = sold_props.groupby('client_ref').agg(
        total_units=('listing_id', 'count'),
        total_spend=('sale_price_clean', 'sum'),
        avg_unit_price=('sale_price_clean', 'mean'),
        avg_unit_sqft=('floor_area_sqft', 'mean')
    ).reset_index().rename(columns={'client_ref': 'client_id'})
    
    df = pd.merge(clients, client_metrics, on='client_id', how='left')
    df[['total_units', 'total_spend', 'avg_unit_price', 'avg_unit_sqft']] = (
        df[['total_units', 'total_spend', 'avg_unit_price', 'avg_unit_sqft']].fillna(0)
    )
    
    df['dob_parsed'] = pd.to_datetime(df['date_of_birth'], format='mixed', errors='coerce')
    df['age'] = 2026 - df['dob_parsed'].dt.year
    df['age'] = df['age'].fillna(df['age'].median())
    
    def assign_persona(row):
        if row['client_type'] == 'Company':
            return 'C3: Corporate Buyers'
        elif row['country'] != 'USA' and row['acquisition_purpose'] == 'Investment':
            return 'C1: Global Investors'
        elif row['satisfaction_score'] >= 4 and row['avg_unit_price'] >= df['avg_unit_price'].median():
            return 'C4: Luxury Investors'
        else:
            return 'C2: First-Time Buyers'
            
    df['buyer_segment'] = df.apply(assign_persona, axis=1)
    df.to_csv('segmented_buyers.csv', index=False)
    return df

df = load_data()

# ─── SIDEBAR FILTERS ───
st.sidebar.header("🎯 Filter Controls")

countries = sorted(df['country'].dropna().unique().tolist())
selected_country = st.sidebar.multiselect("Country", options=countries, default=countries)

available_regions = sorted(df[df['country'].isin(selected_country)]['region'].dropna().unique().tolist())
selected_region = st.sidebar.multiselect("Region (Optional - leave empty for All)", options=available_regions, default=[])

purposes = sorted(df['acquisition_purpose'].dropna().unique().tolist())
selected_purpose = st.sidebar.multiselect("Acquisition Purpose", options=purposes, default=purposes)

client_types = sorted(df['client_type'].dropna().unique().tolist())
selected_client_type = st.sidebar.multiselect("Client Type", options=client_types, default=client_types)

# Apply filters
mask = (
    (df['country'].isin(selected_country)) &
    (df['acquisition_purpose'].isin(selected_purpose)) &
    (df['client_type'].isin(selected_client_type))
)

if selected_region:
    mask = mask & (df['region'].isin(selected_region))

filtered_df = df[mask]

# ─── MAIN APP ───
st.title("🏢 Parcl Real Estate Buyer Intelligence")
st.caption("AI-driven Buyer Segmentation and Investment Profiling Dashboard")

if filtered_df.empty:
    st.warning("⚠️ No records match the current filter selection. Please broaden your filters in the sidebar.")
    st.stop()

# ─── TOP KPI ROW ───
k1, k2, k3, k4 = st.columns(4)
k1.metric("Active Buyers", f"{len(filtered_df):,}")
k2.metric("Total Spend", f"${filtered_df['total_spend'].sum():,.0f}")
k3.metric("Avg Satisfaction", f"{filtered_df['satisfaction_score'].mean():.2f} / 5.0")
k4.metric("Mortgage / Loan Rate", f"{(filtered_df['loan_applied'] == 'Yes').mean() * 100:.1f}%")

st.markdown("---")

# ─── TABS ───
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Buyer Segmentation Overview",
    "📈 Investor Behavior Dashboard",
    "🌍 Geographic Analysis",
    "📑 Segment Insights Panel"
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(
            filtered_df,
            names='buyer_segment',
            title="Cluster Distribution across Segments",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        fig_scatter = px.scatter(
            filtered_df,
            x='total_spend',
            y='avg_unit_price',
            color='buyer_segment',
            size='total_units',
            hover_name='client_id',
            hover_data=['country', 'age', 'satisfaction_score'],
            title="Spend vs. Average Unit Price by Segment",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        fig_loan = px.histogram(
            filtered_df,
            x='buyer_segment',
            color='loan_applied',
            barmode='group',
            title="Financing & Loan Utilization by Segment"
        )
        st.plotly_chart(fig_loan, use_container_width=True)
    with col2:
        fig_channel = px.histogram(
            filtered_df,
            x='buyer_segment',
            color='referral_channel',
            barmode='stack',
            title="Acquisition Referral Channels by Segment"
        )
        st.plotly_chart(fig_channel, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        country_counts = filtered_df.groupby(['country', 'buyer_segment']).size().reset_index(name='count')
        fig_country = px.bar(
            country_counts,
            x='country',
            y='count',
            color='buyer_segment',
            title="Segment Concentration by Country",
            barmode='stack'
        )
        st.plotly_chart(fig_country, use_container_width=True)
    with col2:
        top_regions = filtered_df['region'].value_counts().nlargest(10).index
        fig_reg = px.bar(
            filtered_df[filtered_df['region'].isin(top_regions)],
            x='region',
            color='buyer_segment',
            title="Top 10 Regions by Buyer Volume"
        )
        st.plotly_chart(fig_reg, use_container_width=True)

with tab4:
    st.subheader("Summary Statistics by Segment")
    summary = filtered_df.groupby('buyer_segment').agg(
        Buyers=('client_id', 'count'),
        Median_Age=('age', 'median'),
        Avg_Spend=('total_spend', 'mean'),
        Avg_Unit_Sqft=('avg_unit_sqft', 'mean'),
        Avg_Satisfaction=('satisfaction_score', 'mean'),
        Loan_Usage=('loan_applied', lambda x: f"{(x == 'Yes').mean() * 100:.1f}%")
    ).reset_index()
    
    summary['Avg_Spend'] = summary['Avg_Spend'].map('${:,.2f}'.format)
    summary['Avg_Unit_Sqft'] = summary['Avg_Unit_Sqft'].map('{:,.1f} sqft'.format)
    summary['Avg_Satisfaction'] = summary['Avg_Satisfaction'].map('{:.2f}'.format)
    
    st.dataframe(summary, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Raw Filtered Data")
    st.dataframe(filtered_df, use_container_width=True, height=300)
    
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False),
        file_name="parcl_filtered_buyers.csv",
        mime="text/csv"
    )
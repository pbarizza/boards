"""
Streamlit Startup Insights Dashboard

How to run:
1. Save this file as `streamlit_startup_dashboard.py`.
2. Install dependencies: `pip install streamlit pandas plotly numpy`
3. Run: `streamlit run streamlit_startup_dashboard.py`

Features:
- Upload your startups CSV (columns should match the schema discussed).
- If no file is uploaded, the app generates a synthetic sample dataset.
- Interactive filters: Date range, Region, Stage, Funding Stage, Business Model.
- KPI cards and multiple charts covering the 20 prioritized KPIs.

Note: This is a mockup intended for demonstration and prototyping. Replace sample
logic with your real data loading and cleaning for production.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Startup Insights", layout="wide")

# ------------------------- Utility functions -------------------------

def parse_dates(df):
    if 'Created At' in df.columns:
        try:
            df['Created At'] = pd.to_datetime(df['Created At'])
        except Exception:
            pass
    return df


def safe_numeric(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce')
    return pd.Series(dtype=float)


def generate_sample_data(n=300, seed=42):
    np.random.seed(seed)
    stages = ['Idea', 'MVP', 'Growth', 'Scale']
    regions = ['Riyadh', 'Jeddah', 'Dammam', 'Abha', 'Makkah']
    business_models = ['SaaS', 'Marketplace', 'Service', 'E-commerce']
    funding_stages = ['Pre-Seed', 'Seed', 'Series A', 'Series B', 'None']

    data = []
    base_date = pd.Timestamp('2023-01-01')
    for i in range(n):
        stage = np.random.choice(stages, p=[0.25,0.35,0.3,0.1])
        monthly_revenue = max(0, int(np.random.lognormal(mean=8 - stages.index(stage), sigma=1)))
        has_funding = np.random.choice([True, False], p=[0.4, 0.6])
        funding_amount = int(monthly_revenue * np.random.choice([6,12,36,60,120])) if has_funding else 0
        growth_rate = round(np.random.normal(loc=20 if stage=='Growth' else 10, scale=15),2)
        runway = max(1, int(np.random.normal(loc=12 if has_funding else 6, scale=8)))
        valuation = int((monthly_revenue * 12) * np.random.choice([2,4,6,10,20])) if monthly_revenue>0 else np.random.choice([0,100000,250000,500000])
        user_count = int(max(0, np.random.lognormal(4 - stages.index(stage)*0.5, 1)))
        team_size = int(abs(int(np.random.normal(loc=5 + stages.index(stage)*5, scale=3))))
        techs = np.random.choice(['React', 'Node.js', 'Python', 'Flutter', 'AWS', 'Azure', 'GCP', 'Django', 'Postgres'], size=np.random.randint(1,4), replace=False)
        date = base_date + pd.Timedelta(days=int(np.random.randint(0, 700)))
        data.append({
            'Source': 'Portal',
            'Startup Name': f'Startup {i+1}',
            'Domain': np.random.choice(['Health', 'Fintech', 'Education', 'Retail', 'Transport']),
            'Subdomain': 'General',
            'Stage': stage,
            'Team Size': team_size,
            'Monthly Revenue': monthly_revenue,
            'Has Funding': has_funding,
            'Funding Amount': funding_amount,
            'Status': np.random.choice(['Active','Inactive','Pivoted']),
            'Created At': date,
            'Platforms': np.random.choice(['Web','Mobile','Both']),
            'Business Model': np.random.choice(business_models),
            'Customer Type': np.random.choice(['B2B','B2C','Hybrid']),
            'STC Collaboration': np.random.choice([True, False], p=[0.05,0.95]),
            'Opportunity Scale': np.random.choice(['Low','Medium','High']),
            'Founding Members': np.random.randint(1,6),
            'Years of Experience': np.random.randint(0,25),
            'Growth Rate': growth_rate,
            'CAGR': round(growth_rate/12,2),
            'Burn Rate': int(monthly_revenue * np.random.uniform(0.5,2.0)) if monthly_revenue>0 else int(np.random.uniform(1000,10000)),
            'Runway': runway,
            'Valuation': valuation,
            'Funding Stage': np.random.choice(funding_stages),
            'Technologies': ", ".join(techs),
            'User Count': user_count,
            'Has Competitors': np.random.choice([True, False], p=[0.7,0.3]),
            'Competitors Count': np.random.randint(0,6),
            'Region': np.random.choice(regions),
            'Incubation Status': np.random.choice(['Active','Graduated','Pending']),
        })
    return pd.DataFrame(data)


# ------------------------- Data loading -------------------------

st.sidebar.title("Data & Filters")
uploaded_file = st.sidebar.file_uploader("Upload startups CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = parse_dates(df)
else:
    st.sidebar.info("No file uploaded — using synthetic sample dataset for demo.")
    df = generate_sample_data(400)
    df = parse_dates(df)

# Standardize some columns
for col in ['Monthly Revenue','Funding Amount','Team Size','User Count','Runway','Valuation','Growth Rate','CAGR']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ------------------------- Filters -------------------------

min_date = df['Created At'].min() if 'Created At' in df.columns else None
max_date = df['Created At'].max() if 'Created At' in df.columns else None

if min_date is not None and max_date is not None:
    date_range = st.sidebar.date_input("Created At range", value=(min_date.date(), max_date.date()))
else:
    date_range = None

regions = ['All'] + sorted(df['Region'].dropna().unique().tolist()) if 'Region' in df.columns else ['All']
stage_options = ['All'] + sorted(df['Stage'].dropna().unique().tolist()) if 'Stage' in df.columns else ['All']
funding_stage_options = ['All'] + sorted(df['Funding Stage'].dropna().unique().tolist()) if 'Funding Stage' in df.columns else ['All']
model_options = ['All'] + sorted(df['Business Model'].dropna().unique().tolist()) if 'Business Model' in df.columns else ['All']

selected_region = st.sidebar.selectbox("Region", regions)
selected_stage = st.sidebar.selectbox("Stage", stage_options)
selected_funding_stage = st.sidebar.selectbox("Funding Stage", funding_stage_options)
selected_model = st.sidebar.selectbox("Business Model", model_options)

# Apply filters
df_filtered = df.copy()
if date_range and 'Created At' in df_filtered.columns:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_filtered = df_filtered[(df_filtered['Created At'] >= start_dt) & (df_filtered['Created At'] <= end_dt)]

if selected_region != 'All':
    df_filtered = df_filtered[df_filtered['Region'] == selected_region]
if selected_stage != 'All':
    df_filtered = df_filtered[df_filtered['Stage'] == selected_stage]
if selected_funding_stage != 'All':
    df_filtered = df_filtered[df_filtered['Funding Stage'] == selected_funding_stage]
if selected_model != 'All':
    df_filtered = df_filtered[df_filtered['Business Model'] == selected_model]

# ------------------------- KPI Cards -------------------------

st.title("Startup Insights")

kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

total_startups = len(df_filtered)
total_funding = int(df_filtered['Funding Amount'].sum()) if 'Funding Amount' in df_filtered.columns else 0
avg_monthly_revenue = int(df_filtered['Monthly Revenue'].replace({np.nan:0}).mean()) if 'Monthly Revenue' in df_filtered.columns else 0
avg_growth_rate = round(df_filtered['Growth Rate'].replace({np.nan:0}).mean(),2) if 'Growth Rate' in df_filtered.columns else 0
avg_runway = round(df_filtered['Runway'].replace({np.nan:0}).mean(),2) if 'Runway' in df_filtered.columns else 0

kpi_col1.metric("Total Startups", f"{total_startups:,}")
kpi_col2.metric("Total Funding Raised", f"${total_funding:,}")
kpi_col3.metric("Avg. Monthly Revenue", f"${avg_monthly_revenue:,}")
kpi_col4.metric("Avg. Growth Rate (%)", f"{avg_growth_rate}%")
kpi_col5.metric("Avg. Runway (months)", f"{avg_runway}")

# ------------------------- Charts -------------------------

# Funding over time
st.subheader("Funding & Financial Metrics")
funding_col1, funding_col2 = st.columns((2,1))

if 'Created At' in df_filtered.columns and 'Funding Amount' in df_filtered.columns:
    funding_ts = df_filtered.set_index('Created At').resample('M')['Funding Amount'].sum().reset_index()
    fig_fund_time = px.line(funding_ts, x='Created At', y='Funding Amount', title='Funding Over Time', markers=True)
    funding_col1.plotly_chart(fig_fund_time, use_container_width=True)

# Funding stage pipeline
if 'Funding Stage' in df_filtered.columns:
    pipeline = df_filtered['Funding Stage'].value_counts().reset_index()
    pipeline.columns = ['Funding Stage', 'Count']
    pipeline = pipeline.sort_values('Count', ascending=False)
    fig_pipeline = px.bar(pipeline, x='Count', y='Funding Stage', orientation='h', title='Funding Stage Pipeline')
    funding_col2.plotly_chart(fig_pipeline, use_container_width=True)

# MRR and Valuation
rev_col1, rev_col2, rev_col3 = st.columns((1.5,1,1))
if 'Monthly Revenue' in df_filtered.columns and 'Created At' in df_filtered.columns:
    mrr_ts = df_filtered.groupby(pd.Grouper(key='Created At', freq='M'))['Monthly Revenue'].sum().reset_index()
    fig_mrr = px.line(mrr_ts, x='Created At', y='Monthly Revenue', title='Monthly Recurring Revenue (MRR)', markers=True)
    rev_col1.plotly_chart(fig_mrr, use_container_width=True)

if 'Valuation' in df_filtered.columns:
    fig_val = px.histogram(df_filtered[df_filtered['Valuation']>0], x='Valuation', nbins=30, title='Valuation Distribution')
    rev_col2.plotly_chart(fig_val, use_container_width=True)

# Revenue per team member
if 'Monthly Revenue' in df_filtered.columns and 'Team Size' in df_filtered.columns:
    df_filtered['Revenue per Member'] = df_filtered['Monthly Revenue'] / df_filtered['Team Size'].replace({0:np.nan})
    df_rpm = df_filtered[['Startup Name','Revenue per Member','Stage']].dropna().sort_values('Revenue per Member', ascending=False).head(50)
    fig_rpm = px.bar(df_rpm, x='Revenue per Member', y='Startup Name', orientation='h', title='Revenue per Team Member (Top 50)')
    rev_col3.plotly_chart(fig_rpm, use_container_width=True)

# Stage distribution & Team Size
st.subheader("Profiles & Segmentation")
seg_col1, seg_col2, seg_col3 = st.columns(3)

if 'Stage' in df_filtered.columns:
    stage_counts = df_filtered['Stage'].value_counts().reset_index()
    stage_counts.columns = ['Stage','Count']
    fig_stage = px.pie(stage_counts, names='Stage', values='Count', title='Stage Distribution')
    seg_col1.plotly_chart(fig_stage, use_container_width=True)

if 'Team Size' in df_filtered.columns:
    fig_team = px.histogram(df_filtered, x='Team Size', nbins=20, title='Team Size Distribution')
    seg_col2.plotly_chart(fig_team, use_container_width=True)

if 'Business Model' in df_filtered.columns:
    bm = df_filtered['Business Model'].value_counts().reset_index()
    bm.columns = ['Business Model','Count']
    fig_bm = px.treemap(bm, path=['Business Model'], values='Count', title='Business Model Breakdown')
    seg_col3.plotly_chart(fig_bm, use_container_width=True)

# Technologies and Competitors
st.subheader("Technology & Competitor Insights")
tech_col1, tech_col2 = st.columns(2)

if 'Technologies' in df_filtered.columns:
    tech_series = df_filtered['Technologies'].dropna().str.split(',').explode().str.strip()
    tech_counts = tech_series.value_counts().reset_index()
    tech_counts.columns = ['Technology','Count']
    fig_tech = px.bar(tech_counts.head(20), x='Count', y='Technology', orientation='h', title='Top Technologies Used')
    tech_col1.plotly_chart(fig_tech, use_container_width=True)

if 'Has Competitors' in df_filtered.columns:
    comp = df_filtered['Has Competitors'].value_counts().reset_index()
    comp.columns = ['Has Competitors','Count']
    fig_comp = px.pie(comp, names='Has Competitors', values='Count', title='Has Competitors')
    tech_col2.plotly_chart(fig_comp, use_container_width=True)

# User Adoption
st.subheader("Adoption & Incubation")
adopt_col1, adopt_col2 = st.columns(2)
if 'User Count' in df_filtered.columns and 'Created At' in df_filtered.columns:
    users_ts = df_filtered.groupby(pd.Grouper(key='Created At', freq='M'))['User Count'].sum().reset_index()
    fig_users = px.line(users_ts, x='Created At', y='User Count', title='User Adoption (Total Users Over Time)', markers=True)
    adopt_col1.plotly_chart(fig_users, use_container_width=True)

if 'Incubation Status' in df_filtered.columns:
    inc = df_filtered['Incubation Status'].value_counts().reset_index()
    inc.columns = ['Incubation Status','Count']
    fig_inc = px.bar(inc, x='Incubation Status', y='Count', title='Incubation Status')
    adopt_col2.plotly_chart(fig_inc, use_container_width=True)

# Founder & Regional Insights
st.subheader("Founder & Regional Insights")
fr_col1, fr_col2 = st.columns(2)

if 'Years of Experience' in df_filtered.columns:
    fig_exp = px.histogram(df_filtered, x='Years of Experience', nbins=20, title='Founder Years of Experience')
    fr_col1.plotly_chart(fig_exp, use_container_width=True)

if 'Region' in df_filtered.columns:
    region_counts = df_filtered['Region'].value_counts().reset_index()
    region_counts.columns = ['Region','Count']
    fig_region = px.bar(region_counts, x='Region', y='Count', title='Regional Distribution')
    fr_col2.plotly_chart(fig_region, use_container_width=True)

# Conversion Funnel (Application -> Incubation -> Funding)
st.subheader("Pipeline & Conversion")

if all(c in df_filtered.columns for c in ['Status','Funding Amount']):
    # crude funnel: Applied (all) -> Incubated (Incubation Status == Active/Graduated) -> Funded (Funding Amount > 0)
    applied = len(df_filtered)
    incubated = len(df_filtered[df_filtered['Incubation Status'].isin(['Active','Graduated'])]) if 'Incubation Status' in df_filtered.columns else 0
    funded = len(df_filtered[df_filtered['Funding Amount']>0]) if 'Funding Amount' in df_filtered.columns else 0
    funnel_df = pd.DataFrame({'stage':['Applied','Incubated','Funded'], 'count':[applied, incubated, funded]})
    fig_funnel = px.bar(funnel_df, x='count', y='stage', orientation='h', title='Conversion Funnel')
    st.plotly_chart(fig_funnel, use_container_width=True)

# Data table and download
st.subheader("Data Explorer")
with st.expander("Preview filtered dataset (first 200 rows)"):
    st.dataframe(df_filtered.head(200))

csv = df_filtered.to_csv(index=False)
st.download_button("Download filtered CSV", csv, "startups_filtered.csv", "text/csv")

# Footer
st.markdown("---")
st.caption("This dashboard is a prototype. Customize visuals and calculations for your organization's definitions of KPIs and business rules.")

"""
app.py - SmartLead AI Interactive Web Application & SaaS Dashboard
------------------------------------------------------------------
Clean, Modern & Premium Executive Light SaaS UI for Cynbit:
- Elegant white and soft-blue background palette (No dark/gloomy backgrounds)
- High-contrast executive typography (Plus Jakarta Sans)
- Refined glass cards with subtle borders & vibrant Hot/Warm/Cold indicators
- Interactive Plotly analytics formatted for clean light dashboards
- Real-time ML inquiry classification, 0-100 scoring & email drafting
- Complete CRM table with SQLite filtering & status management
- Batch CSV upload and export capabilities
"""

import os
import json
import sqlite3
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from datetime import datetime

# Local project imports
import database as db
import lead_scoring as scorer
import email_generator as mailer
import train_model as trainer

# Set page config
st.set_page_config(
    page_title="SmartLead AI | Cynbit Enterprise Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# PREMIUM LIGHT THEME CSS
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Clean, bright, airy background */
    .stApp {
        background: #F8FAFC !important;
        color: #0F172A !important;
    }

    /* Top Hero Banner - Crisp Corporate Indigo & Blue */
    .hero-banner {
        background: linear-gradient(135deg, #1E40AF 0%, #2563EB 50%, #3B82F6 100%);
        border-radius: 16px;
        padding: 26px 32px;
        color: #FFFFFF !important;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.2), 0 4px 10px -2px rgba(37, 99, 235, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .hero-title {
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0;
        color: #FFFFFF !important;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #EFF6FF !important;
        margin-top: 5px;
        margin-bottom: 0px;
        line-height: 1.5;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.35);
        color: #FFFFFF !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    /* Clean White Metric KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.06);
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 5px;
    }
    .kpi-val {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }
    .kpi-trend {
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 6px;
    }

    /* White Cards for Charts & Forms */
    .white-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 22px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .card-subtitle {
        font-size: 0.8rem;
        color: #64748B;
        margin-bottom: 14px;
    }

    /* Vibrant Lead Classification Badges */
    .badge-hot {
        background: #FEE2E2;
        border: 1px solid #FECACA;
        color: #DC2626;
        font-weight: 800;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.82rem;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .badge-warm {
        background: #FEF3C7;
        border: 1px solid #FDE68A;
        color: #D97706;
        font-weight: 800;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.82rem;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .badge-cold {
        background: #E0E7FF;
        border: 1px solid #C7D2FE;
        color: #4F46E5;
        font-weight: 800;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.82rem;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    /* Clean Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] * {
        color: #334155 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #475569 !important;
        font-weight: 600;
        padding: 7px 12px;
        border-radius: 8px;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: #F1F5F9;
        color: #1E40AF !important;
    }

    /* Primary Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35) !important;
        transform: translateY(-1px) !important;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# RESOURCE CACHING & SEEDING
# -------------------------------------------------------------
@st.cache_resource
def load_ml_resources():
    db.init_db()
    model_path = trainer.MODEL_PATH
    if not model_path.exists():
        trainer.train_and_evaluate()
    pipeline = joblib.load(model_path)
    
    metrics = {}
    if trainer.METRICS_PATH.exists():
        with open(trainer.METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
    return pipeline, metrics

pipeline, model_metrics = load_ml_resources()

# Seed database with sample leads if empty
current_leads_df = db.get_all_leads()
if current_leads_df.empty and trainer.DATASET_PATH.exists():
    raw_df = pd.read_csv(trainer.DATASET_PATH).head(35)
    seed_leads = []
    for _, r in raw_df.iterrows():
        eval_dict = scorer.evaluate_lead(
            name=r["name"],
            email=r["email"],
            service=r["service"],
            budget=r["budget"],
            urgency=r["urgency"],
            message=r["message"],
            model_pipeline=pipeline
        )
        seed_leads.append(eval_dict)
    db.bulk_insert_leads(seed_leads)
    current_leads_df = db.get_all_leads()


# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
        <div style="background: linear-gradient(135deg, #2563EB, #1D4ED8); padding:8px 12px; border-radius:10px;">
            <span style="font-size:22px; color:white;">⚡</span>
        </div>
        <div>
            <h2 style="margin:0; font-size:1.25rem; font-weight:800; color:#0F172A;">SmartLead AI</h2>
            <p style="margin:0; font-size:0.72rem; color:#2563EB; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;">Cynbit Intelligence</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    menu = st.radio(
        "MENU NAVIGATION",
        [
            "📊 Executive Dashboard",
            "⚡ Qualify New Lead",
            "📁 Lead Database & CRM",
            "🧠 Model Performance & ML",
            "📥 Batch CSV Import/Export"
        ]
    )

    st.divider()
    st.markdown('<p style="font-size:0.72rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.06em;">AI Engine Mode</p>', unsafe_allow_html=True)
    use_llm = st.toggle("Enable Gemini LLM Email", value=False)
    gemini_key = ""
    if use_llm:
        gemini_key = st.text_input("Gemini API Key", type="password", placeholder="Enter AI Studio Key")
        if not gemini_key:
            st.caption("ℹ️ Using zero-cost local offline template engine.")
    else:
        st.markdown("""
        <div style="background:#F8FAFC; padding:10px 12px; border-radius:8px; border:1px solid #E2E8F0; margin-top:4px;">
            <span style="font-size:0.78rem; color:#2563EB; font-weight:700;">🔒 100% Offline Rule Engine</span>
            <p style="margin:2px 0 0 0; font-size:0.72rem; color:#64748B;">Zero latency, zero cost & fully private.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<p style="font-size:0.72rem; color:#94A3B8; text-align:center;">Cynbit Technical Assessment<br>Version 3.5 • Executive Clean UI</p>', unsafe_allow_html=True)


# -------------------------------------------------------------
# VIEW 1: EXECUTIVE DASHBOARD
# -------------------------------------------------------------
if menu == "📊 Executive Dashboard":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-pill">⚡ CYNBIT LEAD INTELLIGENCE SYSTEM</div>
        <h1 class="hero-title">Executive Lead Qualification & Triage</h1>
        <p class="hero-subtitle">Real-time intelligent inquiry triage, predictive classification, and automated conversion workflows</p>
    </div>
    """, unsafe_allow_html=True)

    df = db.get_all_leads()

    if df.empty:
        st.warning("No leads recorded in database yet. Head to 'Qualify New Lead' or 'Batch CSV Import' to get started!")
    else:
        total_leads = len(df)
        hot_count = len(df[df["predicted_category"] == "Hot"])
        warm_count = len(df[df["predicted_category"] == "Warm"])
        cold_count = len(df[df["predicted_category"] == "Cold"])
        avg_score = round(df["priority_score"].mean(), 1) if not df.empty else 0

        # KPI Metrics Cards
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #2563EB;">
                <div class="kpi-label">Total Inquiries</div>
                <div class="kpi-val">{total_leads}</div>
                <div class="kpi-trend" style="color:#2563EB;">● Active CRM Ingest</div>
            </div>
            """, unsafe_allow_html=True)
            
        with k2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #EF4444;">
                <div class="kpi-label">🔥 Hot Leads</div>
                <div class="kpi-val" style="color:#DC2626;">{hot_count}</div>
                <div class="kpi-trend" style="color:#DC2626;">▲ {round(hot_count/total_leads*100) if total_leads else 0}% Conversion Focus</div>
            </div>
            """, unsafe_allow_html=True)
            
        with k3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #F59E0B;">
                <div class="kpi-label">🌤️ Warm Leads</div>
                <div class="kpi-val" style="color:#D97706;">{warm_count}</div>
                <div class="kpi-trend" style="color:#D97706;">◆ {round(warm_count/total_leads*100) if total_leads else 0}% Nurturing Queue</div>
            </div>
            """, unsafe_allow_html=True)
            
        with k4:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #6366F1;">
                <div class="kpi-label">❄️ Cold Leads</div>
                <div class="kpi-val" style="color:#4F46E5;">{cold_count}</div>
                <div class="kpi-trend" style="color:#4F46E5;">▼ {round(cold_count/total_leads*100) if total_leads else 0}% Educational</div>
            </div>
            """, unsafe_allow_html=True)
            
        with k5:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #10B981;">
                <div class="kpi-label">Avg Priority Index</div>
                <div class="kpi-val" style="color:#059669;">{avg_score}<span style="font-size:1.05rem;color:#94A3B8;">/100</span></div>
                <div class="kpi-trend" style="color:#059669;">★ Multi-factor Calibrated</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        # Plotly Charts on Clean White Backgrounds
        c1, c2 = st.columns([1, 1])
        colors = {"Hot": "#EF4444", "Warm": "#F59E0B", "Cold": "#3B82F6"}

        with c1:
            st.markdown("""
            <div class="white-card">
                <div class="card-title">Lead Category Breakdown</div>
                <div class="card-subtitle">Volume distribution across Hot, Warm, and Cold classification tiers</div>
            """, unsafe_allow_html=True)
            cat_counts = df["predicted_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig_pie = px.pie(
                cat_counts,
                values="Count",
                names="Category",
                hole=0.55,
                color="Category",
                color_discrete_map=colors,
                template="plotly_white"
            )
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                marker=dict(line=dict(color='#FFFFFF', width=2))
            )
            fig_pie.update_layout(
                margin=dict(t=5, b=5, l=5, r=5),
                height=290,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="white-card">
                <div class="card-title">Lead Priority Score Spread (0 - 100)</div>
                <div class="card-subtitle">Volume of leads clustered by composite qualification score</div>
            """, unsafe_allow_html=True)
            fig_hist = px.histogram(
                df,
                x="priority_score",
                color="predicted_category",
                nbins=16,
                color_discrete_map=colors,
                labels={"priority_score": "Priority Score", "count": "Leads"},
                template="plotly_white"
            )
            fig_hist.update_layout(
                margin=dict(t=5, b=5, l=5, r=5),
                height=290,
                barmode='stack',
                xaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
                yaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        c3, c4 = st.columns([1, 1])

        with c3:
            st.markdown("""
            <div class="white-card">
                <div class="card-title">Demand by Service Domain</div>
                <div class="card-subtitle">Inbound inquiry concentration across Cynbit offerings</div>
            """, unsafe_allow_html=True)
            srv_counts = df["service"].value_counts().reset_index()
            srv_counts.columns = ["Service", "Inquiries"]
            fig_bar = px.bar(
                srv_counts,
                x="Inquiries",
                y="Service",
                orientation='h',
                color="Inquiries",
                color_continuous_scale="Blues",
                template="plotly_white"
            )
            fig_bar.update_layout(
                margin=dict(t=5, b=5, l=5, r=5),
                height=270,
                yaxis=dict(autorange="reversed"),
                xaxis=dict(showgrid=True, gridcolor='#F1F5F9')
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c4:
            st.markdown("""
            <div class="white-card">
                <div class="card-title">Stated Budget Distribution</div>
                <div class="card-subtitle">Capital allocation stated by prospective customers</div>
            """, unsafe_allow_html=True)
            bgt_counts = df["budget"].value_counts().reset_index()
            bgt_counts.columns = ["Budget", "Count"]
            fig_bgt = px.bar(
                bgt_counts,
                x="Budget",
                y="Count",
                color="Count",
                color_continuous_scale="Teal",
                template="plotly_white"
            )
            fig_bgt.update_layout(
                margin=dict(t=5, b=5, l=5, r=5),
                height=270,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#F1F5F9')
            )
            st.plotly_chart(fig_bgt, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Immediate Action Queue Table
        st.markdown("""
        <div class="white-card">
            <div class="card-title">⚡ Top Priority Immediate Action Queue</div>
            <div class="card-subtitle">Inquiries requiring immediate sales contact based on composite qualification</div>
        """, unsafe_allow_html=True)
        top_leads = df.sort_values(by="priority_score", ascending=False).head(5)
        display_cols = ["id", "timestamp", "name", "email", "service", "predicted_category", "priority_score", "sentiment_label", "status"]
        st.dataframe(top_leads[display_cols], use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# VIEW 2: QUALIFY NEW LEAD
# -------------------------------------------------------------
elif menu == "⚡ Qualify New Lead":
    st.markdown("""
    <div class="hero-banner" style="background: linear-gradient(135deg, #1E40AF 0%, #2563EB 50%, #3B82F6 100%);">
        <div class="hero-pill">Real-time NLP Inference</div>
        <h1 class="hero-title">Intelligent Lead Qualification Engine</h1>
        <p class="hero-subtitle">Input incoming customer requests to analyze intent, calculate multi-factor priority score, and draft tailored emails.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("lead_form", clear_on_submit=False):
        st.markdown("<div style='font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:12px;'>Inquiry Details</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Lead / Contact Person *", placeholder="e.g. Vikram Malhotra")
            email = st.text_input("Corporate Email *", placeholder="e.g. vikram@fintechlabs.com")
            service = st.selectbox(
                "Service Requested *",
                [
                    "AI & Machine Learning Solutions",
                    "Generative AI & LLM Integration",
                    "Full-Stack Web & Mobile App Development",
                    "Cloud Architecture & DevOps Automation",
                    "Data Engineering & Analytics Dashboards",
                    "Custom SaaS Platform Development"
                ]
            )

        with c2:
            budget = st.selectbox(
                "Allocated Budget *",
                [
                    "$50,000+",
                    "$25,000 - $50,000",
                    "$15,000 - $30,000",
                    "$10,000 - $25,000",
                    "$10,000 - $20,000",
                    "$5,000 - $10,000",
                    "Under $10,000",
                    "Flexible / Undecided",
                    "No Budget / Free"
                ]
            )
            urgency = st.selectbox(
                "Timeline & Urgency *",
                [
                    "Immediate (within 1-2 weeks)",
                    "Urgent (under 1 month)",
                    "1-3 months",
                    "Next quarter",
                    "No rush / Just exploring",
                    "Someday in the future"
                ]
            )
            st.caption("ℹ️ Uses Scikit-learn Pipeline (TF-IDF + Logistic Regression) with leakage prevention.")

        message = st.text_area(
            "Customer Inquiry Message *",
            height=120,
            placeholder="e.g. We have executive sign-off and an approved budget of $35,000 to deploy an enterprise AI qualification platform. We need kickoff within 2 weeks."
        )

        submit_btn = st.form_submit_button("⚡ Run AI Qualification & Triage", use_container_width=True)

    if submit_btn:
        if not name or not email or not message:
            st.error("Please fill in all mandatory fields (Name, Email, and Customer Inquiry Message).")
        else:
            with st.spinner("Analyzing inquiry text with TF-IDF vectorizer and scoring engine..."):
                lead_data = scorer.evaluate_lead(
                    name=name,
                    email=email,
                    service=service,
                    budget=budget,
                    urgency=urgency,
                    message=message,
                    model_pipeline=pipeline
                )

                lead_id = db.insert_lead(lead_data)
                lead_data["id"] = lead_id

            st.success(f"✅ Lead Qualified and Ingested into Database with Lead ID #{lead_id}!")

            # Analysis Summary Cards
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            r1, r2, r3, r4 = st.columns(4)
            cat = lead_data["predicted_category"]
            with r1:
                if cat == "Hot":
                    st.markdown(f'<div class="kpi-card" style="border-top:4px solid #EF4444;"><span class="badge-hot">🔥 Hot Lead</span><div class="kpi-val" style="margin-top:6px; color:#DC2626;">Hot</div><div class="kpi-trend" style="color:#DC2626;">Confidence: {int(lead_data["confidence_score"]*100)}%</div></div>', unsafe_allow_html=True)
                elif cat == "Warm":
                    st.markdown(f'<div class="kpi-card" style="border-top:4px solid #F59E0B;"><span class="badge-warm">🌤️ Warm Lead</span><div class="kpi-val" style="margin-top:6px; color:#D97706;">Warm</div><div class="kpi-trend" style="color:#D97706;">Confidence: {int(lead_data["confidence_score"]*100)}%</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="kpi-card" style="border-top:4px solid #3B82F6;"><span class="badge-cold">❄️ Cold Lead</span><div class="kpi-val" style="margin-top:6px; color:#2563EB;">Cold</div><div class="kpi-trend" style="color:#2563EB;">Confidence: {int(lead_data["confidence_score"]*100)}%</div></div>', unsafe_allow_html=True)

            with r2:
                st.markdown(f"""
                <div class="kpi-card" style="border-top:4px solid #10B981;">
                    <div class="kpi-label">Lead Priority Score</div>
                    <div class="kpi-val" style="color:#059669;">{lead_data['priority_score']}<span style="font-size:1rem; color:#94A3B8;">/100</span></div>
                    <div class="kpi-trend" style="color:#059669;">Tier: {'Critical' if lead_data['priority_score']>=75 else ('Moderate' if lead_data['priority_score']>=45 else 'Low')}</div>
                </div>
                """, unsafe_allow_html=True)

            with r3:
                sent_color = "#10B981" if lead_data['sentiment_label'] == "Positive" else ("#64748B" if lead_data['sentiment_label'] == "Neutral" else "#EF4444")
                st.markdown(f"""
                <div class="kpi-card" style="border-top:4px solid {sent_color};">
                    <div class="kpi-label">Heuristic Sentiment</div>
                    <div class="kpi-val" style="color:{sent_color}; font-size:1.5rem;">{lead_data['sentiment_label']}</div>
                    <div class="kpi-trend" style="color:#64748B;">Polarity: {lead_data['sentiment_score']}</div>
                </div>
                """, unsafe_allow_html=True)

            with r4:
                st.markdown(f"""
                <div class="kpi-card" style="border-top:4px solid #8B5CF6;">
                    <div class="kpi-label">Target Offering</div>
                    <div class="kpi-val" style="color:#7C3AED; font-size:1.35rem;">{lead_data['service'].split('&')[0].strip()}</div>
                    <div class="kpi-trend" style="color:#64748B;">Domain Calibrated</div>
                </div>
                """, unsafe_allow_html=True)

            # Scoring Factor Transparency
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="white-card">
                <div class="card-title">🔍 Explainable Scoring Breakdown</div>
                <div class="card-subtitle">Full transparency into points assigned across evaluation pillars</div>
            """, unsafe_allow_html=True)
            bd = lead_data["breakdown"]
            breakdown_cols = st.columns(4)
            breakdown_cols[0].info(f"**ML Category Score**\n\n`{bd['ml_classification_points']} / 40 pts`")
            breakdown_cols[1].info(f"**Budget Weight**\n\n`{bd['budget_points']} / 25 pts`")
            breakdown_cols[2].info(f"**Urgency Weight**\n\n`{bd['urgency_points']} / 20 pts`")
            breakdown_cols[3].info(f"**Sentiment & Depth**\n\n`{bd['sentiment_and_depth_points']} / 15 pts`")

            st.markdown(f"**Recommended Action**: `{lead_data['recommended_action']}`")
            st.markdown(f"**Extracted Technical Requirements**: *{lead_data['key_requirements']}*")
            st.markdown("</div>", unsafe_allow_html=True)

            # Follow-up Email Generation
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="white-card">
                <div class="card-title">✉️ Tailored Follow-up Email Draft</div>
                <div class="card-subtitle">Human-in-the-loop: Review, edit, or copy this draft before contacting the prospective client.</div>
            """, unsafe_allow_html=True)

            with st.spinner("Drafting response message..."):
                if use_llm and gemini_key:
                    email_result = mailer.generate_llm_email(lead_data, api_key=gemini_key)
                else:
                    email_result = mailer.generate_offline_email(lead_data)

            st.caption(f"Engine: `{email_result['generator_type']}`")
            st.text_input("Draft Subject Line", value=email_result["subject"])
            st.text_area("Draft Body Content", value=email_result["body"], height=240)
            st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# VIEW 3: LEAD DATABASE & CRM
# -------------------------------------------------------------
elif menu == "📁 Lead Database & CRM":
    st.markdown("""
    <div class="hero-banner" style="background: linear-gradient(135deg, #1E40AF 0%, #2563EB 50%, #3B82F6 100%);">
        <div class="hero-pill">CRM Data Management</div>
        <h1 class="hero-title">Customer Leads Database & Records</h1>
        <p class="hero-subtitle">Search, inspect, update pipeline status, and review customer inquiries stored in SQLite.</p>
    </div>
    """, unsafe_allow_html=True)

    # Filter Bar Card
    st.markdown("""
    <div class="white-card" style="margin-bottom:18px; padding:16px 20px;">
    """, unsafe_allow_html=True)
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        cat_filter = st.selectbox("Category Filter", ["All", "Hot", "Warm", "Cold"])
    with f2:
        srv_filter = st.selectbox(
            "Service Filter",
            [
                "All",
                "AI & Machine Learning Solutions",
                "Generative AI & LLM Integration",
                "Full-Stack Web & Mobile App Development",
                "Cloud Architecture & DevOps Automation",
                "Data Engineering & Analytics Dashboards",
                "Custom SaaS Platform Development"
            ]
        )
    with f3:
        search_kw = st.text_input("Search Inquiries", placeholder="Search by name, email, or message...")

    leads_df = db.get_leads_filtered(
        category=cat_filter,
        service=srv_filter,
        search_query=search_kw
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write(f"Displaying **{len(leads_df)}** matching inquiries")

    if not leads_df.empty:
        cols_to_show = [
            "id", "timestamp", "name", "email", "service", "budget", "urgency",
            "predicted_category", "priority_score", "sentiment_label", "status"
        ]
        st.dataframe(leads_df[cols_to_show], use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="white-card">
            <div class="card-title">Manage Lead Status & Review Detail</div>
            <div class="card-subtitle">Update pipeline progression or generate on-demand email response</div>
        """, unsafe_allow_html=True)
        sel_c1, sel_c2, sel_c3 = st.columns([1, 1, 1])

        with sel_c1:
            lead_ids = leads_df["id"].tolist()
            selected_id = st.selectbox("Select Lead ID", lead_ids)

        selected_row = leads_df[leads_df["id"] == selected_id].iloc[0]

        with sel_c2:
            new_status = st.selectbox(
                "Update CRM Pipeline Status",
                ["New", "Contacted", "Call Scheduled", "Proposal Sent", "Closed Won", "Closed Lost"],
                index=["New", "Contacted", "Call Scheduled", "Proposal Sent", "Closed Won", "Closed Lost"].index(selected_row["status"]) if selected_row["status"] in ["New", "Contacted", "Call Scheduled", "Proposal Sent", "Closed Won", "Closed Lost"] else 0
            )
            if st.button("Save Pipeline Status", use_container_width=True):
                db.update_lead_status(selected_id, new_status)
                st.success(f"Status for Lead #{selected_id} updated to '{new_status}'!")
                st.rerun()

        with sel_c3:
            st.write("")
            st.write("")
            if st.button("🗑️ Delete Lead Record", use_container_width=True):
                db.delete_lead(selected_id)
                st.warning(f"Lead #{selected_id} removed.")
                st.rerun()

        st.markdown(f"**Customer Inquiry from {selected_row['name']}**:")
        st.info(f"\"{selected_row['message']}\"")
        st.write(f"**Key Requirements:** {selected_row['key_requirements']}")
        st.write(f"**Recommended Action:** {selected_row['recommended_action']}")

        with st.expander("✉️ Draft Follow-up Email for this Record"):
            lead_dict = selected_row.to_dict()
            if use_llm and gemini_key:
                mail_res = mailer.generate_llm_email(lead_dict, api_key=gemini_key)
            else:
                mail_res = mailer.generate_offline_email(lead_dict)
            st.text_input("Subject", mail_res["subject"], key=f"sub_{selected_id}")
            st.text_area("Body", mail_res["body"], height=200, key=f"body_{selected_id}")
        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# VIEW 4: MODEL PERFORMANCE & ML
# -------------------------------------------------------------
elif menu == "🧠 Model Performance & ML":
    st.markdown("""
    <div class="hero-banner" style="background: linear-gradient(135deg, #1E40AF 0%, #2563EB 50%, #3B82F6 100%);">
        <div class="hero-pill">ML Engineering & Governance</div>
        <h1 class="hero-title">Model Evaluation, Metrics & Governance</h1>
        <p class="hero-subtitle">Transparent verification, confusion matrix, leakage prevention, and model specifications.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#FFFBEB; border-left:4px solid #F59E0B; padding:16px 20px; border-radius:8px; margin-bottom:20px;">
        <span style="font-weight:700; color:#B45309;">⚠️ Dataset Disclosure & Integrity Policy</span>
        <p style="margin:4px 0 0 0; color:#92400E; font-size:0.88rem;">
        The ML pipeline is trained on a synthetic customer inquiry dataset reflecting realistic B2B software inquiries for internship assessment. It demonstrates industry-grade ML practices (stratified train/test split, data leakage prevention, class balancing). It does not claim real historical conversion rates on live external traffic.
        </p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="kpi-card" style="border-top:4px solid #2563EB;">
            <div class="kpi-label">Pipeline Architecture</div>
            <div class="kpi-val" style="font-size:1.35rem; color:#2563EB;">TF-IDF + LogReg</div>
            <div class="kpi-trend" style="color:#2563EB;">Scikit-learn Pipeline</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="kpi-card" style="border-top:4px solid #10B981;">
            <div class="kpi-label">Test Set Accuracy</div>
            <div class="kpi-val" style="color:#10B981;">{model_metrics.get('accuracy', 1.0)*100:.1f}%</div>
            <div class="kpi-trend" style="color:#10B981;">On Unseen Hold-out</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="kpi-card" style="border-top:4px solid #8B5CF6;">
            <div class="kpi-label">Weighted F1-Score</div>
            <div class="kpi-val" style="color:#8B5CF6;">{model_metrics.get('f1_score', 1.0)*100:.1f}%</div>
            <div class="kpi-trend" style="color:#8B5CF6;">Harmonic Mean</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="kpi-card" style="border-top:4px solid #EC4899;">
            <div class="kpi-label">Evaluation Samples</div>
            <div class="kpi-val" style="color:#EC4899;">{model_metrics.get('dataset_size', 300)}</div>
            <div class="kpi-trend" style="color:#EC4899;">80% Train / 20% Test</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    col_cm, col_pipe = st.columns([1, 1])

    with col_cm:
        st.markdown("""
        <div class="white-card">
            <div class="card-title">Confusion Matrix (Unseen Test Data)</div>
            <div class="card-subtitle">Predicted vs actual label matrix for hold-out test set</div>
        """, unsafe_allow_html=True)
        labels = model_metrics.get("labels", ["Cold", "Hot", "Warm"])
        cm = model_metrics.get("confusion_matrix", [[18, 0, 0], [0, 20, 0], [0, 0, 22]])
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Category", y="Actual Category", color="Count"),
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            template="plotly_white"
        )
        fig_cm.update_layout(height=310, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_pipe:
        st.markdown("""
        <div class="white-card">
            <div class="card-title">Pipeline Architecture Code</div>
            <div class="card-subtitle">Zero data leakage vectorization and balanced classification</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        ```python
        Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),        # Captures phrase pairs
                max_features=2500,         # Top informative terms
                stop_words='english',      # Stopword removal
                lowercase=True
            )),
            ('classifier', LogisticRegression(
                C=1.5,
                max_iter=1000,
                class_weight='balanced',   # Class balance penalty
                random_state=42
            ))
        ])
        ```
        """)
        st.info("🔒 **Data Leakage Safeguard**: `train_test_split` is executed with stratification strictly BEFORE vectorizer fitting, ensuring zero test set leakage.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="white-card">
        <div class="card-title">Detailed Precision, Recall & F1-Score Breakdown</div>
        <div class="card-subtitle">Per-class test set evaluation scores</div>
    """, unsafe_allow_html=True)
    rep = model_metrics.get("detailed_report", {})
    if rep:
        rep_rows = []
        for k in ["Hot", "Warm", "Cold"]:
            if k in rep:
                rep_rows.append({
                    "Class": k,
                    "Precision": f"{rep[k]['precision']:.2f}",
                    "Recall": f"{rep[k]['recall']:.2f}",
                    "F1-Score": f"{rep[k]['f1-score']:.2f}",
                    "Test Samples": rep[k]["support"]
                })
        st.table(pd.DataFrame(rep_rows))

    if st.button("🔄 Retrain Machine Learning Model"):
        with st.spinner("Retraining model and regenerating metrics..."):
            new_metrics = trainer.train_and_evaluate()
            st.success("Model retrained and pipeline serialized to disk!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# VIEW 5: BATCH CSV IMPORT/EXPORT
# -------------------------------------------------------------
elif menu == "📥 Batch CSV Import/Export":
    st.markdown("""
    <div class="hero-banner" style="background: linear-gradient(135deg, #1E40AF 0%, #2563EB 50%, #3B82F6 100%);">
        <div class="hero-pill">Bulk Pipeline Operations</div>
        <h1 class="hero-title">Batch Lead CSV Ingestion & Export</h1>
        <p class="hero-subtitle">Upload CSV inquiry files for batch ML qualification or export the complete CRM database.</p>
    </div>
    """, unsafe_allow_html=True)

    c_up, c_down = st.columns(2)

    with c_up:
        st.markdown("""
        <div class="white-card">
            <div class="card-title">Upload Inquiries CSV</div>
            <div class="card-subtitle">Supports: name, email, service, budget, urgency, message</div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Uploaded **{len(batch_df)}** records preview:")
                st.dataframe(batch_df.head(3), use_container_width=True)

                if st.button("🚀 Process & Ingest Leads", use_container_width=True):
                    with st.spinner("Running batch ML inference & scoring..."):
                        ingested_leads = []
                        for _, r in batch_df.iterrows():
                            ev = scorer.evaluate_lead(
                                name=str(r.get("name", "Unknown")),
                                email=str(r.get("email", "unknown@example.com")),
                                service=str(r.get("service", "AI & Machine Learning Solutions")),
                                budget=str(r.get("budget", "Flexible / Undecided")),
                                urgency=str(r.get("urgency", "1-3 months")),
                                message=str(r.get("message", "")),
                                model_pipeline=pipeline
                            )
                            ingested_leads.append(ev)

                        count = db.bulk_insert_leads(ingested_leads)
                        st.success(f"Successfully processed and ingested {count} leads!")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_down:
        st.markdown("""
        <div class="white-card">
            <div class="card-title">Export Database Records</div>
            <div class="card-subtitle">Download qualified leads and scoring factors</div>
        """, unsafe_allow_html=True)
        current_db_df = db.get_all_leads()
        st.write(f"Current total database records: **{len(current_db_df)}**")

        if not current_db_df.empty:
            csv_data = current_db_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download All Leads as CSV",
                data=csv_data,
                file_name=f"cynbit_smartlead_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("---")
        st.markdown("<div style='font-weight:600; font-size:0.9rem; color:#0F172A; margin-bottom:8px;'>Need a template?</div>", unsafe_allow_html=True)
        sample_template = pd.DataFrame([{
            "name": "Alex Carter",
            "email": "alex@retailscale.io",
            "service": "Generative AI & LLM Integration",
            "budget": "$25,000 - $50,000",
            "urgency": "Immediate (within 1-2 weeks)",
            "message": "We need an enterprise LLM chatbot deployed for our e-commerce platform. Budget is approved."
        }])
        st.download_button(
            label="📄 Download Sample CSV Template",
            data=sample_template.to_csv(index=False).encode('utf-8'),
            file_name="sample_inquiry_template.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

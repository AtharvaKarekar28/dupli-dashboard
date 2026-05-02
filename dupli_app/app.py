import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pickle
import os
import requests

# ── Constants ─────────────────────────────────────────────────────────────────
PRESS_SPEED  = 5217
WORKING_DAYS = 250
PROFIT_RATE  = 0.0025
BEFORE_MEAN  = 24.57
AFTER_MEAN   = 11.38

# Annual production goals (for cumulative chart + reference bar)
ANNUAL_TARGETS = {"15M": 15_000_000, "25M": 25_000_000, "30M": 30_000_000, "39.1M": 39_100_000}
DAILY_TARGETS  = {k: v / WORKING_DAYS for k, v in ANNUAL_TARGETS.items()}

# Full staffing scenario data
# Each entry: label → {s1, s2, daily, annual, profit, gain_annual}
STAFFING_DATA = {
    "Current": {
        "desc":    "S1: 2 ops · 2 machines · 4 productive hrs/machine\nS2: Thomas · M1 only · 6.5 hrs",
        "s1":      41_736,
        "s2":      33_910,
        "daily":   75_646,
        "annual":  18_911_500,
        "profit":  47_279,
        "gain_m":  0,
        "gain_$":  0,
    },
    "A1: +M2 op S2": {
        "desc":    "S1 unchanged. New operator runs M2 in S2. No packing person.",
        "s1":      41_736,
        "s2":      67_820,
        "daily":   109_556,
        "annual":  27_389_000,
        "profit":  68_473,
        "gain_m":  8.5,
        "gain_$":  21_194,
    },
    "A2: +Packer S1": {
        "desc":    "3 people in S1. Downtime 3 hrs → 0.5 hrs. S2 unchanged.",
        "s1":      67_820,
        "s2":      33_910,
        "daily":   101_730,
        "annual":  25_432_500,
        "profit":  63_581,
        "gain_m":  6.5,
        "gain_$":  16_302,
    },
    "A3: +Packer Both": {
        "desc":    "3 people in S1. Shipping person → packing in S2. Zero extra hiring cost.",
        "s1":      67_820,
        "s2":      44_344,
        "daily":   112_164,
        "annual":  28_041_000,
        "profit":  70_103,
        "gain_m":  9.1,
        "gain_$":  22_824,
    },
    "A4: A3 + M2 op S2 ⭐": {
        "desc":    "A3 + new M2 operator in S2. Max output. 4 people across both shifts.",
        "s1":      67_820,
        "s2":      88_688,
        "daily":   156_508,
        "annual":  39_127_000,
        "profit":  97_818,
        "gain_m":  20.2,
        "gain_$":  50_539,
    },
}

FLOOR_SETUP = {500:1.47, 750:2.39, 1000:1.48, 2000:7.00, 2500:2.62,
               2685:1.70, 5000:3.50, 7500:3.00, 10000:7.45}

AFTER_PILOT_OBS = [(750,8.11),(2500,14.40),(2685,15.00),(2500,15.88),
                   (2500,14.90),(1000,7.58),(7500,9.00),(2000,12.53),(1000,5.00)]

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("gb_before.pkl"):
        with open("gb_before.pkl", "rb") as f:
            return pickle.load(f)
    return None

def get_setup(qty):
    if qty in FLOOR_SETUP:
        return FLOOR_SETUP[qty]
    return FLOOR_SETUP[min(FLOOR_SETUP.keys(), key=lambda k: abs(k - qty))]

def predict_before(qty, setup, model):
    if model:
        X = np.array([[qty, setup, np.log(max(qty, 1))]])
        return float(model.predict(X)[0])
    return setup + (qty / PRESS_SPEED) * 60

def lookup_after(qty):
    matches = [ct for q, ct in AFTER_PILOT_OBS if q == qty]
    return float(np.mean(matches)) if matches else None

# ── Supabase REST ─────────────────────────────────────────────────────────────
SUPABASE_URL = "https://pvjkkrvoxecxjiedwexe.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2amtrcnZveGVjeGppZWR3ZXhlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzczMjM2NzYsImV4cCI6MjA5Mjg5OTY3Nn0.ugLv5GWwG5I8eNz2uS_Z00ur0vQPX1s_N5qBvJ92UEQ"

def _headers():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }

def load_log():
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/daily_log?select=*&order=log_date.asc",
        headers=_headers(), timeout=10
    )
    if r.status_code != 200 or not r.json():
        return pd.DataFrame(columns=["id","log_date","m1_output","m2_output","notes","total"])
    df = pd.DataFrame(r.json())
    df["log_date"]  = pd.to_datetime(df["log_date"])
    df["m1_output"] = pd.to_numeric(df["m1_output"], errors="coerce").fillna(0).astype(int)
    df["m2_output"] = pd.to_numeric(df["m2_output"], errors="coerce").fillna(0).astype(int)
    df["total"]     = df["m1_output"] + df["m2_output"]
    return df

def insert_log(date, m1, m2, notes):
    requests.post(
        f"{SUPABASE_URL}/rest/v1/daily_log",
        headers=_headers(),
        json={"log_date": str(date), "m1_output": int(m1),
              "m2_output": int(m2), "notes": notes or ""},
        timeout=10
    )

def delete_log(row_id):
    requests.delete(
        f"{SUPABASE_URL}/rest/v1/daily_log?id=eq.{int(row_id)}",
        headers=_headers(), timeout=10
    )

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dupli Production Dashboard", page_icon="✉️", layout="wide")

# Dupli brand colors + bigger fonts + top strip
st.markdown("""
<style>
    /* ── Global font size ── */
    html, body, [class*="css"] { font-size: 17px !important; }
    h1 { font-size: 2.4rem !important; }
    h2 { font-size: 1.9rem !important; }
    h3 { font-size: 1.5rem !important; }
    .stMetric label  { font-size: 1.05rem !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 2.1rem !important; }
    .stDataFrame { font-size: 1rem !important; }

    /* ── Blue top header strip ── */
    .dupli-header {
        background-color: #4A7BA7;
        padding: 14px 28px;
        display: flex;
        align-items: center;
        gap: 18px;
        margin: -1rem -1rem 1.5rem -1rem;
        border-bottom: 3px solid #2d5f8a;
    }
    .dupli-header img {
        height: 52px;
        background: white;
        padding: 6px 12px;
        border-radius: 6px;
    }
    .dupli-header-text {
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .dupli-header-sub {
        color: #d0e4f5;
        font-size: 0.95rem;
        margin-top: 2px;
    }

    /* ── Sidebar brand color ── */
    section[data-testid="stSidebar"] {
        background-color: #4A7BA7 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 1rem !important;
    }
</style>

<div class="dupli-header">
    <img src="data:image/png;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCANSBCQDASIAAhEBAxEB/8QAHQABAAIDAQEBAQAAAAAAAAAAAAcIBQYJBAEDAv/EAEgQAQABAwMBAwgFCQUHBAMAAAABAgMEBQYRBxIhMQgTF0FRVJPTFDdhcYEVIjJ0dZGxsrMWNTZSoSMzQkNzgsEYYnLRJZLw/8QAGgEBAAMBAQEAAAAAAAAAAAAAAAECAwUEBv/EAC0RAQABAgUDAgUFAQEAAAAAAAABAgMEERJSoSFR4TFBBRNhcZEUIoHB8LEj/9oADAMBAAIRAxEAPwC5YAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA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AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB/9k=" alt="Dupli logo"/>
    <div>
        <div class="dupli-header-text">Dupli Production Dashboard</div>
        <div class="dupli-header-sub">SCM 755 · Lean Six Sigma · Syracuse, NY</div>
    </div>
</div>
""", unsafe_allow_html=True)

PAGES = ["📊 Daily Dashboard", "⏱ Cycle Time Model", "👥 Staffing Assumptions", "📋 Production Log"]

with st.sidebar:
    st.title("✉️ Dupli Dashboard")
    st.caption("SCM 755 LSS · Syracuse NY")
    st.divider()
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("Champion: Steve Moore")
    st.caption("Scheid (S1)  ·  Thomas (S2)")
    st.caption(f"Press speed: {PRESS_SPEED:,} env/hr")

model = load_model()

# ═══════════════════════════════════════════════════════════════════
# PAGE 1 – DAILY DASHBOARD
# ═══════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.header("📊 Daily Production Dashboard")

    with st.expander("➕ Log today's output", expanded=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
        log_date = c1.date_input("Date", value=datetime.date.today())
        m1_out   = c2.number_input("Memjet 1", min_value=0, max_value=500_000, step=1000, value=40_000)
        m2_out   = c3.number_input("Memjet 2", min_value=0, max_value=500_000, step=1000, value=10_000)
        notes    = c4.text_input("Notes")
        if st.button("Log", type="primary"):
            insert_log(str(log_date), m1_out, m2_out, notes)
            st.success(f"Logged {m1_out + m2_out:,} envelopes.")
            st.rerun()

    log_df = load_log()

    if not log_df.empty:
        # KPI strip
        latest = int(log_df.iloc[-1]["total"])
        k1, k2, k3 = st.columns(3)
        k1.metric("Latest Day Output", f"{latest:,}")
        k2.metric("vs 25M goal (100K/day)", f"{latest - 100_000:+,}", delta_color="normal")
        k3.metric("vs 30M goal (120K/day)", f"{latest - 120_000:+,}", delta_color="normal")
        st.divider()

        # ── Daily output trend ──────────────────────────────────────────────
        st.subheader("Daily Output Trend")
        avg_daily = int(log_df["total"].mean())
        fig2 = go.Figure()
        fig2.add_scatter(x=log_df["log_date"], y=log_df["total"],
                         mode="lines+markers", name="Daily Output",
                         line=dict(color="#4A7BA7", width=2),
                         marker=dict(size=6))
        # Average line
        fig2.add_hline(y=avg_daily, line_dash="dash", line_color="#9467bd", line_width=2,
                       annotation_text=f"Avg: {avg_daily:,}",
                       annotation_position="left",
                       annotation_font_size=13)
        fig2.add_hline(y=60_000,  line_dash="dot", line_color="#2ca02c", line_width=1.5,
                       annotation_text="15M (60K/day)", annotation_position="right",
                       annotation_font_size=12)
        fig2.add_hline(y=100_000, line_dash="dot", line_color="#ff7f0e", line_width=1.5,
                       annotation_text="25M (100K/day)", annotation_position="right",
                       annotation_font_size=12)
        fig2.add_hline(y=120_000, line_dash="dot", line_color="#d62728", line_width=1.5,
                       annotation_text="30M (120K/day)", annotation_position="right",
                       annotation_font_size=12)
        fig2.update_layout(
            height=380,
            margin=dict(t=10, r=120, l=60),
            xaxis_title="Date",
            yaxis_title="Envelopes / Day",
            xaxis=dict(
                tickformat="%b %d",
                dtick="D7",          # equal 7-day intervals
                tickangle=-45,
                tickfont=dict(size=12),
            ),
            yaxis=dict(tickfont=dict(size=12)),
            font=dict(size=14),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Cumulative production vs targets ─────────────────────────────────
        st.subheader("Cumulative Production vs Annual Targets")

        log_sorted = log_df.sort_values("log_date").copy()
        log_sorted["cumulative"] = log_sorted["total"].cumsum()
        n_days = len(log_sorted)

        date_range = pd.date_range(
            start=log_sorted["log_date"].min(),
            periods=n_days
        )

        fig3 = go.Figure()
        fig3.add_scatter(
            x=log_sorted["log_date"],
            y=log_sorted["cumulative"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#4A7BA7", width=3),
            marker=dict(size=6),
        )
        target_colours = {"15M": "#2ca02c", "25M": "#ff7f0e", "30M": "#d62728", "39.1M": "#9467bd"}
        for lbl, ann in ANNUAL_TARGETS.items():
            daily_pace = ann / WORKING_DAYS
            target_cum = [daily_pace * d for d in range(1, n_days + 1)]
            fig3.add_scatter(
                x=date_range,
                y=target_cum,
                mode="lines",
                name=f"{lbl} pace",
                line=dict(dash="dot", color=target_colours[lbl], width=2),
            )
        fig3.update_layout(
            height=420,
            xaxis_title="Date",
            yaxis_title="Cumulative Envelopes",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=10, r=20),
            font=dict(size=14),
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ── Monthly stacked bar — MOVED TO BOTTOM ────────────────────────────
        st.subheader("Monthly Output vs Targets")
        log_df["YM"] = log_df["log_date"].dt.to_period("M").astype(str)
        monthly = log_df.groupby("YM")[["m1_output", "m2_output"]].sum().reset_index()
        monthly["total"] = monthly["m1_output"] + monthly["m2_output"]

        fig = go.Figure()
        fig.add_bar(x=monthly["YM"], y=monthly["m1_output"], name="Memjet 1", marker_color="#4A7BA7")
        fig.add_bar(x=monthly["YM"], y=monthly["m2_output"], name="Memjet 2", marker_color="#2d5f8a")
        for lbl, ann in ANNUAL_TARGETS.items():
            fig.add_hline(
                y=ann / 12, line_dash="dot",
                annotation_text=f"{lbl} ({ann/12/1e3:.0f}K/mo)",
                annotation_position="right",
                annotation_font_size=12,
            )
        fig.update_layout(
            barmode="stack",
            height=500,
            margin=dict(r=140, t=20, b=60),
            xaxis_title="Month",
            yaxis_title="",          # removed y-axis label per professor
            xaxis=dict(tickfont=dict(size=13)),
            yaxis=dict(tickfont=dict(size=13)),
            legend=dict(font=dict(size=13)),
            font=dict(size=14),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No data yet — log your first day above.")

    # ── Required daily output reference ──────────────────────────────────────
    st.subheader("Required Daily Output by Annual Goal (reference)")
    req = pd.DataFrame({
        "Goal":    list(ANNUAL_TARGETS.keys()),
        "Env/Day": [v / WORKING_DAYS for v in ANNUAL_TARGETS.values()]
    })
    fig4 = px.bar(req, x="Goal", y="Env/Day", text="Env/Day", color="Goal",
                  color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    fig4.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig4.update_layout(height=280, showlegend=False, margin=dict(t=10))
    st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 – CYCLE TIME MODEL
# ═══════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.header("⏱ Cycle Time Model")
    if not model:
        st.warning("gb_before.pkl not found — run `python train.py` first. Using linear fallback.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Job Estimator")
        qty   = st.number_input("Quantity", min_value=100, max_value=50_000, step=250, value=2500)
        setup = st.number_input("Setup (min) — auto-filled from floor obs",
                                min_value=0.0, max_value=60.0,
                                step=0.1, value=float(get_setup(int(qty))), format="%.2f")
        before = predict_before(qty, setup, model)
        after  = lookup_after(int(qty))

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Before (model)", f"{before:.1f} min")
        m2.metric("After (pilot)", f"{after:.2f} min" if after else "No exact obs")
        if after:
            m3.metric("Reduction", f"{(before - after) / before * 100:.1f}%", delta_color="inverse")

        st.divider()
        st.metric("Overall pilot improvement",
                  f"{BEFORE_MEAN:.2f} → {AFTER_MEAN:.2f} min",
                  delta=f"-{(BEFORE_MEAN - AFTER_MEAN) / BEFORE_MEAN * 100:.1f}%",
                  delta_color="inverse")

    with col2:
        st.subheader("All 9 Pilot Observations")
        rows = []
        for q, act in AFTER_PILOT_OBS:
            s = get_setup(q)
            b = predict_before(q, s, model)
            rows.append({"Qty": q, "Setup": s, "Before": round(b, 2),
                         "After": act, "Δ%": round((b - act) / b * 100, 1)})
        cmp = pd.DataFrame(rows)
        st.dataframe(cmp, use_container_width=True, hide_index=True)

        fig5 = go.Figure()
        fig5.add_scatter(x=cmp["Qty"], y=cmp["Before"], mode="markers+lines",
                         name="Before", marker=dict(color="red", size=8))
        fig5.add_scatter(x=cmp["Qty"], y=cmp["After"], mode="markers",
                         name="After", marker=dict(color="green", size=10, symbol="star"))
        fig5.update_layout(height=300, xaxis_title="Qty",
                           yaxis_title="Cycle Time (min)", margin=dict(t=10))
        st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 – STAFFING ASSUMPTIONS
# ═══════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.header("👥 Staffing Assumptions")
    st.caption(f"Press speed: {PRESS_SPEED:,} env/hr  ·  Working days: {WORKING_DAYS}/yr  ·  Profit rate: ${PROFIT_RATE}/envelope")

    # ── Summary table ─────────────────────────────────────────────
    st.subheader("Scenario Summary")
    summary_rows = []
    for lbl, d in STAFFING_DATA.items():
        gain_str = f"+{d['gain_m']}M · +${d['gain_$']:,}/yr" if d["gain_m"] > 0 else "—"
        summary_rows.append({
            "Scenario":        lbl,
            "S1 Output":       f"{d['s1']:,}",
            "S2 Output":       f"{d['s2']:,}",
            "Daily Total":     f"{d['daily']:,}",
            "Annual":          f"{d['annual']/1e6:.1f}M",
            "Profit/yr":       f"${d['profit']:,}",
            "Gain vs Current": gain_str,
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ── Scenario detail cards ──────────────────────────────────────
    st.subheader("Scenario Breakdown")
    colours = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for (lbl, d), col in zip(STAFFING_DATA.items(), colours):
        with st.expander(f"**{lbl}** — {d['daily']:,}/day · {d['annual']/1e6:.1f}M/yr"):
            st.caption(d["desc"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("S1 Output", f"{d['s1']:,}")
            c2.metric("S2 Output", f"{d['s2']:,}")
            c3.metric("Daily Total", f"{d['daily']:,}")
            c4.metric("Annual", f"{d['annual']/1e6:.1f}M")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Profit/yr", f"${d['profit']:,}")
            if d["gain_m"] > 0:
                cc2.metric("Annual Gain", f"+{d['gain_m']}M envelopes")
                cc3.metric("Profit Gain", f"+${d['gain_$']:,}/yr")

    # ── Daily output comparison bar ────────────────────────────────
    st.subheader("Daily Output by Scenario")
    fig6 = go.Figure()
    for (lbl, d), col in zip(STAFFING_DATA.items(), colours):
        fig6.add_bar(
            x=[lbl], y=[d["daily"]],
            marker_color=col,
            text=f"{d['daily']:,}",
            textposition="outside",
            name=lbl,
        )
    goal_colours = {"15M": "#2ca02c", "25M": "#1f77b4", "30M": "#ff7f0e", "39.1M": "#d62728"}
    for lbl, daily in DAILY_TARGETS.items():
        fig6.add_hline(
            y=daily, line_dash="dot",
            annotation_text=f"{lbl} goal ({daily/1000:.0f}K/day)",
            annotation_position="right",
            line_color=goal_colours[lbl],
        )
    fig6.update_layout(
        showlegend=False, height=460,
        margin=dict(r=150, t=10, b=80),
        yaxis_title="Envelopes / Day",
        xaxis_tickangle=-15,
    )
    st.plotly_chart(fig6, use_container_width=True)

    # ── Goal check table ───────────────────────────────────────────
    st.subheader("Goal Check")
    goal_check = [
        {"Goal": "15M (envelopes.com)", "Target/day": "60,000",  "Achieved by": "✅ All scenarios"},
        {"Goal": "25M (combined)",       "Target/day": "100,000", "Achieved by": "✅ A1, A2, A3, A4"},
        {"Goal": "28M (A3 max)",         "Target/day": "112,000", "Achieved by": "✅ A3 and A4"},
        {"Goal": "30M",                  "Target/day": "120,000", "Achieved by": "✅ A4 only"},
        {"Goal": "39.1M",                "Target/day": "156,508", "Achieved by": "✅ A4 only"},
    ]
    st.dataframe(pd.DataFrame(goal_check), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 4 – PRODUCTION LOG
# ═══════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.header("📋 Production Log")
    log_df = load_log()

    if log_df.empty:
        st.info("No entries yet — log output from the Daily Dashboard.")
    else:
        k1, k2 = st.columns(2)
        k1.metric("Days Logged", len(log_df))
        k2.metric("Total Envelopes", f"{log_df['total'].sum():,}")
        st.divider()

        disp = log_df[["id", "log_date", "m1_output", "m2_output", "total", "notes"]].copy()
        disp.columns = ["ID", "Date", "Memjet 1", "Memjet 2", "Total", "Notes"]
        disp["Date"] = disp["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(disp.sort_values("Date", ascending=False),
                     use_container_width=True, hide_index=True)

        st.divider()
        del_id = st.number_input("Delete entry by ID", min_value=1, step=1, value=1)
        if st.button("🗑 Delete"):
            delete_log(int(del_id))
            st.success(f"Deleted ID {del_id}.")
            st.rerun()

        csv = log_df.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", csv, "dupli_log.csv", "text/csv")
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
# NOTE: keep the inline base64 logo from your existing file — paste it back into
# the two placeholders below (HEADER_LOGO_BASE64 and SIDEBAR_LOGO_BASE64).
HEADER_LOGO_BASE64 = "PASTE_HEADER_LOGO_BASE64_HERE"
SIDEBAR_LOGO_BASE64 = "PASTE_SIDEBAR_LOGO_BASE64_HERE"

st.markdown(f"""
<style>
    /* ── Global font size ── */
    html, body, [class*="css"] {{ font-size: 17px !important; }}
    h1 {{ font-size: 2.4rem !important; }}
    h2 {{ font-size: 1.9rem !important; }}
    h3 {{ font-size: 1.5rem !important; }}
    .stMetric label  {{ font-size: 1.05rem !important; }}
    .stMetric [data-testid="stMetricValue"] {{ font-size: 2.1rem !important; }}
    .stDataFrame {{ font-size: 1rem !important; }}

    /* ── Blue top header strip ── */
    .dupli-header {{
        background-color: #4A7BA7;
        padding: 14px 28px;
        display: flex;
        align-items: center;
        gap: 18px;
        margin: -1rem -1rem 1.5rem -1rem;
        border-bottom: 3px solid #2d5f8a;
    }}
    .dupli-header img {{
        height: 52px;
        background: white;
        padding: 6px 12px;
        border-radius: 6px;
    }}
    .dupli-header-text {{
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}
    .dupli-header-sub {{
        color: #d0e4f5;
        font-size: 0.95rem;
        margin-top: 2px;
    }}

    /* ── Sidebar brand color ── */
    section[data-testid="stSidebar"] {{
        background-color: #4A7BA7 !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: #ffffff !important;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        font-size: 1rem !important;
    }}
</style>

<div class="dupli-header">
    <img src="data:image/png;base64,{HEADER_LOGO_BASE64}" alt="Dupli logo"/>
    <div>
        <div class="dupli-header-text">Production Dashboard</div>
    </div>
</div>
""", unsafe_allow_html=True)

PAGES = ["📊 Daily Dashboard", "⏱ Cycle Time Model", "👥 Staffing Assumptions", "📋 Production Log"]

with st.sidebar:
    st.markdown(f'<img src="data:image/png;base64,{SIDEBAR_LOGO_BASE64}" style="width:100%;max-width:180px;margin-bottom:8px;background:white;padding:8px;border-radius:6px;">', unsafe_allow_html=True)
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

    # ── Required daily output reference — FIRST ──────────────────────────────
    st.subheader("Required Daily Output by Annual Goal")
    CURRENT_AVG = 50_669
    req = pd.DataFrame({
        "Goal":    ["Current Avg"] + list(ANNUAL_TARGETS.keys()),
        "Env/Day": [CURRENT_AVG] + [v / WORKING_DAYS for v in ANNUAL_TARGETS.values()]
    })
    fig4 = px.bar(req, x="Goal", y="Env/Day", text="Env/Day", color="Goal",
                  color_discrete_sequence=["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    fig4.update_traces(texttemplate="%{text:,.0f}", textposition="outside",
                       textfont=dict(size=15))
    fig4.update_layout(height=320, showlegend=False, margin=dict(t=10),
                       font=dict(size=15),
                       xaxis=dict(tickfont=dict(size=14)),
                       yaxis=dict(tickfont=dict(size=14)))
    st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── Log form — SECOND ────────────────────────────────────────────────────
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

        # ── Daily Output Bar Chart vs A2 target ─────────────────────────────
        st.subheader("Daily Output vs A2 Target (101,730/day)")

        A2_TARGET = 101_730
        avg_daily = int(log_df["total"].mean())
        days_hit  = int((log_df["total"] >= A2_TARGET).sum())

        g1, g2 = st.columns(2)
        g1.metric("Avg Daily Output", f"{avg_daily:,}")
        g2.metric("Gap to A2 Target", f"{avg_daily - A2_TARGET:+,}/day", delta_color="normal")

        bar_colors = ["#2ca02c" if v >= A2_TARGET else "#d62728"
                      for v in log_df["total"]]

        fig2 = go.Figure()
        fig2.add_bar(
            x=log_df["log_date"],
            y=log_df["total"],
            name="Daily Output",
            marker_color=bar_colors,
            text=log_df["total"].apply(lambda x: f"{x/1000:.0f}K"),
            textposition="outside",
            textfont=dict(size=13),
        )
        fig2.add_hline(
            y=A2_TARGET, line_dash="dash", line_color="#4A7BA7", line_width=2.5,
            annotation_text=f"{A2_TARGET:,}",
            annotation_position="right",
            annotation_font_size=13,
            annotation_font_color="#4A7BA7",
        )
        fig2.update_layout(height=440,
            margin=dict(t=30, r=140, l=60, b=60),
            xaxis_title="Date",
            yaxis_title="Envelopes / Day",
            xaxis=dict(tickformat="%b %d", tickangle=-30,
                       tickfont=dict(size=14), nticks=20),
            yaxis=dict(tickfont=dict(size=14)),
            font=dict(size=15),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("🟢 Hit target  🔴 Below target")

        # ── Cumulative production vs targets ─────────────────────────────────
        st.subheader("Cumulative Production vs Annual Targets")

        log_sorted = log_df.sort_values("log_date").copy()
        log_sorted["cumulative"] = log_sorted["total"].cumsum()

        # Full year date range Jan 1 → Dec 31 2026
        year_start = pd.Timestamp("2026-01-01")
        year_end   = pd.Timestamp("2026-12-31")

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
            fig3.add_scatter(
                x=[year_start, year_end],
                y=[0, ann],
                mode="lines",
                name=f"{lbl} pace",
                line=dict(dash="dot", color=target_colours[lbl], width=2),
            )
        fig3.update_layout(height=420,
            xaxis_title="Date",
            yaxis_title="Cumulative Envelopes",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=13)),
            margin=dict(t=10, r=20),
            font=dict(size=15),
            xaxis=dict(tickfont=dict(size=13)),
            yaxis=dict(tickfont=dict(size=13)),
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ── Monthly stacked bar — MOVED TO BOTTOM ────────────────────────────
        st.subheader("Monthly Output vs Targets")
        log_df["YM"] = log_df["log_date"].dt.to_period("M").astype(str)
        monthly = log_df.groupby("YM")[["m1_output", "m2_output"]].sum().reset_index()
        monthly["total"] = monthly["m1_output"] + monthly["m2_output"]
        # Format x-axis as "Jan 2026" etc.
        monthly["YM_label"] = pd.to_datetime(monthly["YM"]).dt.strftime("%b %Y")

        fig = go.Figure()
        fig.add_bar(x=monthly["YM_label"], y=monthly["m1_output"],
                    name="Memjet 1", marker_color="#4A7BA7")
        fig.add_bar(x=monthly["YM_label"], y=monthly["m2_output"],
                    name="Memjet 2", marker_color="#7EB6D9")

        target_line_colours = {
            "15M":   "#2ca02c",
            "25M":   "#ff7f0e",
            "30M":   "#d62728",
            "39.1M": "#9467bd",
        }
        for lbl, ann in ANNUAL_TARGETS.items():
            col = target_line_colours[lbl]
            fig.add_hline(
                y=ann / 12,
                line_dash="dot",
                line_color=col,
                line_width=2,
                annotation_text=f"{lbl} ({ann/12/1e3:.0f}K/mo)",
                annotation_position="right",
                annotation_font_size=12,
                annotation_font_color=col,
            )
        fig.update_layout(barmode="stack",
            height=500,
            margin=dict(r=150, t=20, b=60),
            xaxis_title="Month",
            yaxis_title="",
            xaxis=dict(tickfont=dict(size=13), type="category"),
            yaxis=dict(tickfont=dict(size=13)),
            legend=dict(font=dict(size=13)),
            font=dict(size=14),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No data yet — log your first day above.")


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 – CYCLE TIME MODEL  (per report — Figure 3, Appendix A1, A4)
# ═══════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.header("⏱ Cycle Time Model")
    st.caption("Pilot test calculations — based on the pilot order list (16 orders, 50,935 envelopes) and observed pilot timing.")

    # ── Headline metrics ────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Throughput",       "2,741 → 8,650 env/hr",  "+216%")
    m2.metric("Daily Output",     "21,479 → 50,935 env",   "+137%")
    m3.metric("Cycle Time / 500", "10.9 → 3.5 min",        "−67.9%", delta_color="inverse")
    m4.metric("Daily Downtime",   "4 → 2 hr/day",          "−50%",   delta_color="inverse")

    st.divider()

    # ── Key Metrics Comparison ──────────────────────────────────────────
    st.subheader("Key Metrics — Before vs After Pilot")
    key_metrics = [
        {"Key Metric": "Throughput",          "Before Pilot": "2,741 env/hr",   "After Pilot": "8,650 env/hr",   "Improvement": "+5,909 env/hr  ·  +216%"},
        {"Key Metric": "Daily output",         "Before Pilot": "21,479 env/day", "After Pilot": "50,935 env/day", "Improvement": "+29,456 env/day  ·  +137%"},
        {"Key Metric": "Daily machine DT",     "Before Pilot": "4 hr/day",        "After Pilot": "2 hr/day",        "Improvement": "−2 hr/day  ·  −50%"},
        {"Key Metric": "Cycle time / 500 env", "Before Pilot": "10.9 min",        "After Pilot": "3.5 min",         "Improvement": "−7.4 min  ·  −67.9%"},
        {"Key Metric": "Changeover time",      "Before Pilot": "5 min",           "After Pilot": "2 min",           "Improvement": "−3 min  ·  −60%"},
    ]
    st.dataframe(pd.DataFrame(key_metrics), use_container_width=True, hide_index=True)

    st.divider()

    # ── Visual summary ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Throughput (env/hour)**")
        fig_t = go.Figure(go.Bar(
            x=["Before Pilot", "After Pilot"],
            y=[2741, 8650],
            text=["2,741", "8,650"],
            textposition="outside",
            marker_color=["#E74C3C", "#27AE60"],
            textfont=dict(size=14),
        ))
        fig_t.update_layout(height=340, margin=dict(t=20, b=20),
                            yaxis_title="Envelopes / hour",
                            yaxis=dict(range=[0, 10000], tickfont=dict(size=13)),
                            xaxis=dict(tickfont=dict(size=14)),
                            font=dict(size=14))
        st.plotly_chart(fig_t, use_container_width=True)

    with col2:
        st.markdown("**Cycle Time (min / 500 envelopes)**")
        fig_c = go.Figure(go.Bar(
            x=["Before Pilot", "After Pilot"],
            y=[10.9, 3.5],
            text=["10.9 min", "3.5 min"],
            textposition="outside",
            marker_color=["#E74C3C", "#27AE60"],
            textfont=dict(size=14),
        ))
        fig_c.update_layout(height=340, margin=dict(t=20, b=20),
                            yaxis_title="Minutes per 500 envelopes",
                            yaxis=dict(range=[0, 14], tickfont=dict(size=13)),
                            xaxis=dict(tickfont=dict(size=14)),
                            font=dict(size=14))
        st.plotly_chart(fig_c, use_container_width=True)

    st.divider()

    # ── Throughput Calculation (Appendix A1) ───────────────────────────
    st.subheader("Throughput Calculation")
    st.caption("Pilot test result — 16 orders, 50,935 envelopes total")
    throughput_rows = [
        {"Item": "Total pilot quantity",      "Formula": "Pilot order list",          "Result": "50,935 envelopes"},
        {"Item": "Before pilot time",         "Formula": "Dupli est. completion time", "Result": "672 min"},
        {"Item": "After pilot time",          "Formula": "Observed pilot total time",  "Result": "353.5 min"},
        {"Item": "Before pilot throughput",   "Formula": "50,935 ÷ (672 ÷ 60)",        "Result": "4,548 env/hour"},
        {"Item": "After pilot throughput",    "Formula": "50,935 ÷ (353.5 ÷ 60)",      "Result": "8,650 env/hour"},
        {"Item": "Improvement",               "Formula": "8,650 − 4,548",              "Result": "+4,102 env/hour"},
        {"Item": "% Improvement",             "Formula": "4,102 ÷ 4,548",              "Result": "90.2%"},
    ]
    st.dataframe(pd.DataFrame(throughput_rows), use_container_width=True, hide_index=True)
    st.caption("Table A1. Before/after throughput calculations — pilot batch comparison.")

    st.divider()

    # ── Cycle Time per 500 envelopes (Appendix A4) ─────────────────────
    st.subheader("Cycle Time per 500 Envelopes")
    st.markdown("Cycle time and throughput are inversely related: **Cycle Time = 1 ÷ throughput**")
    cycle_rows = [
        {"Item": "Throughput → Cycle Time",                    "Formula": "Cycle Time = 1 ÷ throughput",  "Result": "Cycle Time (min/unit)"},
        {"Item": "Michael before pilot — 2,741 env/hr",        "Formula": "(1 ÷ 2,741) × 500 × 60",       "Result": "10.9 min / 500 envelopes"},
        {"Item": "Michael + Mark after pilot — 8,650 env/hr",  "Formula": "(1 ÷ 8,650) × 500 × 60",       "Result": "3.5 min / 500 envelopes"},
        {"Item": "Improvement",                                "Formula": "10.9 − 3.5",                   "Result": "7.4 min / 500 env  ·  67.9%"},
    ]
    st.dataframe(pd.DataFrame(cycle_rows), use_container_width=True, hide_index=True)
    st.caption("Table A4. Before/after cycle time per 500 envelopes — operator-level (Michael alone vs. Michael + Mark).")

    st.divider()

    # ── Pilot Order List (Figure 3) ─────────────────────────────────────
    with st.expander("📋 Pilot Order List (Figure 3) — 16 orders, 50,935 envelopes"):
        pilot_orders = [
            {"Order #": "14371",    "Qty": 2685,  "Setup (min)": 2.5, "Working (min)": 15,   "Downtime (min)": 2.5, "Total After (min)": 20,   "Before (min)": 82,  "Saved (min)": 62},
            {"Order #": "4049",     "Qty": 1000,  "Setup (min)": 4.5, "Working (min)": 5,    "Downtime (min)": 1,   "Total After (min)": 10.5, "Before (min)": 14,  "Saved (min)": 3.5},
            {"Order #": "61593",    "Qty": 750,   "Setup (min)": 2.5, "Working (min)": 2.5,  "Downtime (min)": 0.5, "Total After (min)": 5.5,  "Before (min)": 30,  "Saved (min)": 24.5},
            {"Order #": "46523",    "Qty": 5000,  "Setup (min)": 4.5, "Working (min)": 20.5, "Downtime (min)": 5.5, "Total After (min)": 30.5, "Before (min)": 32,  "Saved (min)": 1.5},
            {"Order #": "46523",    "Qty": 10000, "Setup (min)": 2,   "Working (min)": 42.5, "Downtime (min)": 10,  "Total After (min)": 54.5, "Before (min)": 95,  "Saved (min)": 40.5},
            {"Order #": "3983",     "Qty": 2500,  "Setup (min)": 3.5, "Working (min)": 10.5, "Downtime (min)": 1,   "Total After (min)": 15,   "Before (min)": 20,  "Saved (min)": 5},
            {"Order #": "3983",     "Qty": 2500,  "Setup (min)": 7.5, "Working (min)": 10.5, "Downtime (min)": 1,   "Total After (min)": 19,   "Before (min)": 37,  "Saved (min)": 18},
            {"Order #": "61595",    "Qty": 1000,  "Setup (min)": 4.5, "Working (min)": 5,    "Downtime (min)": 1,   "Total After (min)": 10.5, "Before (min)": 31,  "Saved (min)": 20.5},
            {"Order #": "C103216",  "Qty": 3500,  "Setup (min)": 5,   "Working (min)": 15,   "Downtime (min)": 3.5, "Total After (min)": 23.5, "Before (min)": 25,  "Saved (min)": 1.5},
            {"Order #": "C103216",  "Qty": 15000, "Setup (min)": 5.5, "Working (min)": 62,   "Downtime (min)": 12,  "Total After (min)": 79.5, "Before (min)": 105, "Saved (min)": 25.5},
            {"Order #": "448526A",  "Qty": 500,   "Setup (min)": 5,   "Working (min)": 3.5,  "Downtime (min)": 0.5, "Total After (min)": 9,    "Before (min)": 24,  "Saved (min)": 15},
            {"Order #": "35532",    "Qty": 1000,  "Setup (min)": 5,   "Working (min)": 5,    "Downtime (min)": 1,   "Total After (min)": 11,   "Before (min)": 31,  "Saved (min)": 20},
            {"Order #": "35532",    "Qty": 2000,  "Setup (min)": 4,   "Working (min)": 9.5,  "Downtime (min)": 2,   "Total After (min)": 15.5, "Before (min)": 49,  "Saved (min)": 33.5},
            {"Order #": "69BW-200", "Qty": 1000,  "Setup (min)": 6,   "Working (min)": 6.5,  "Downtime (min)": 1,   "Total After (min)": 13.5, "Before (min)": 19,  "Saved (min)": 5.5},
            {"Order #": "40313",    "Qty": 500,   "Setup (min)": 6.5, "Working (min)": 4,    "Downtime (min)": 0.5, "Total After (min)": 11,   "Before (min)": 31,  "Saved (min)": 20},
            {"Order #": "1737FSC",  "Qty": 2000,  "Setup (min)": 7,   "Working (min)": 15,   "Downtime (min)": 3,   "Total After (min)": 25,   "Before (min)": 47,  "Saved (min)": 22},
        ]
        df_pilot = pd.DataFrame(pilot_orders)
        total_row = pd.DataFrame([{
            "Order #":           "TOTAL",
            "Qty":               df_pilot["Qty"].sum(),
            "Setup (min)":       df_pilot["Setup (min)"].sum(),
            "Working (min)":     df_pilot["Working (min)"].sum(),
            "Downtime (min)":    df_pilot["Downtime (min)"].sum(),
            "Total After (min)": df_pilot["Total After (min)"].sum(),
            "Before (min)":      df_pilot["Before (min)"].sum(),
            "Saved (min)":       df_pilot["Saved (min)"].sum(),
        }])
        df_pilot_full = pd.concat([df_pilot, total_row], ignore_index=True)
        st.dataframe(df_pilot_full, use_container_width=True, hide_index=True)
        st.caption("Figure 3. Pilot test order list. Column F (Total After) is observed; column G (Before) is Dupli's pre-pilot estimate.")


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
    fig6.update_layout(showlegend=False, height=460,
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
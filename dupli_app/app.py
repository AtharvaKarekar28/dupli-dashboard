import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pickle
import os
import requests

# ── Dupli Logo (base64-encoded JPEG) ─────────────────────────────────────────
DUPLI_LOGO = "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"

# ── Constants ─────────────────────────────────────────────────────────────────
PRESS_SPEED  = 5217
WORKING_DAYS = 250
PROFIT_RATE  = 0.0025
BEFORE_MEAN  = 24.57
AFTER_MEAN   = 11.38

ANNUAL_TARGETS = {"15M": 15_000_000, "25M": 25_000_000, "30M": 30_000_000, "39.1M": 39_100_000}
DAILY_TARGETS  = {k: v / WORKING_DAYS for k, v in ANNUAL_TARGETS.items()}

STAFFING_DATA = {
    "Current": {
        "desc":    "S1: 2 ops · 2 machines · 4 productive hrs/machine\nS2: Thomas · M1 only · 6.5 hrs",
        "s1":      41_736, "s2": 33_910, "daily": 75_646,
        "annual":  18_911_500, "profit": 47_279, "gain_m": 0, "gain_$": 0,
    },
    "A1: +M2 op S2": {
        "desc":    "S1 unchanged. New operator runs M2 in S2. No packing person.",
        "s1":      41_736, "s2": 67_820, "daily": 109_556,
        "annual":  27_389_000, "profit": 68_473, "gain_m": 8.5, "gain_$": 21_194,
    },
    "A2: +Packer S1": {
        "desc":    "3 people in S1. Downtime 3 hrs → 0.5 hrs. S2 unchanged.",
        "s1":      67_820, "s2": 33_910, "daily": 101_730,
        "annual":  25_432_500, "profit": 63_581, "gain_m": 6.5, "gain_$": 16_302,
    },
    "A3: +Packer Both": {
        "desc":    "3 people in S1. Shipping person → packing in S2. Zero extra hiring cost.",
        "s1":      67_820, "s2": 44_344, "daily": 112_164,
        "annual":  28_041_000, "profit": 70_103, "gain_m": 9.1, "gain_$": 22_824,
    },
    "A4: A3 + M2 op S2 ⭐": {
        "desc":    "A3 + new M2 operator in S2. Max output. 4 people across both shifts.",
        "s1":      67_820, "s2": 88_688, "daily": 156_508,
        "annual":  39_127_000, "profit": 97_818, "gain_m": 20.2, "gain_$": 50_539,
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

# ── Styling (CSS, no logo here) ──────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 17px !important; }
    h1 { font-size: 2.4rem !important; }
    h2 { font-size: 1.9rem !important; }
    h3 { font-size: 1.5rem !important; }
    .stMetric label  { font-size: 1.05rem !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 2.1rem !important; }
    .stDataFrame { font-size: 1rem !important; }
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
    section[data-testid="stSidebar"] { background-color: #4A7BA7 !important; }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Header strip with logo ───────────────────────────────────────────────────
st.markdown(
    '<div class="dupli-header">'
    f'<img src="data:image/jpeg;base64,{DUPLI_LOGO}" alt="Dupli logo"/>'
    '<div><div class="dupli-header-text">Production Dashboard</div></div>'
    '</div>',
    unsafe_allow_html=True
)

PAGES = ["📊 Daily Dashboard", "⏱ Cycle Time Model", "👥 Staffing Assumptions", "📋 Production Log"]

with st.sidebar:
    st.markdown(
        f'<img src="data:image/jpeg;base64,{DUPLI_LOGO}" '
        'style="width:100%;max-width:180px;margin-bottom:8px;background:white;padding:8px;border-radius:6px;">',
        unsafe_allow_html=True
    )
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
        latest = int(log_df.iloc[-1]["total"])
        k1, k2, k3 = st.columns(3)
        k1.metric("Latest Day Output", f"{latest:,}")
        k2.metric("vs 25M goal (100K/day)", f"{latest - 100_000:+,}", delta_color="normal")
        k3.metric("vs 30M goal (120K/day)", f"{latest - 120_000:+,}", delta_color="normal")
        st.divider()

        st.subheader("Daily Output vs A2 Target (101,730/day)")

        A2_TARGET = 101_730
        avg_daily = int(log_df["total"].mean())

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

        st.subheader("Cumulative Production vs Annual Targets")

        log_sorted = log_df.sort_values("log_date").copy()
        log_sorted["cumulative"] = log_sorted["total"].cumsum()

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

        st.subheader("Monthly Output vs Targets")
        log_df["YM"] = log_df["log_date"].dt.to_period("M").astype(str)
        monthly = log_df.groupby("YM")[["m1_output", "m2_output"]].sum().reset_index()
        monthly["total"] = monthly["m1_output"] + monthly["m2_output"]
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

    # Headline metrics — Row 1
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Throughput · operator",   "2,741 → 8,650 env/hr",  "+216%")
    r1c2.metric("Throughput Improvement",  "+90.2%",                "4,548 → 8,650 env/hr")
    r1c3.metric("Daily Output",            "21,479 → 50,935 env",   "+137%")

    # Headline metrics — Row 2
    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric("Cycle Time / 500", "10.9 → 3.5 min", "−67.9%", delta_color="inverse")
    r2c2.metric("Daily Downtime",   "4 → 2 hr/day",   "−50%",   delta_color="inverse")
    r2c3.metric("Changeover Time",  "5 → 2 min",      "−60%",   delta_color="inverse")

    st.divider()

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

    st.subheader("🔍 Compare a Specific Order")
    st.caption("Pick an order from the pilot list to see its before/after pilot times.")

    order_labels = [f"{o['Order #']}  ·  {o['Qty']:,} envelopes" for o in pilot_orders]
    label_to_order = dict(zip(order_labels, pilot_orders))
    selected_label = st.selectbox("Order", order_labels, label_visibility="collapsed")
    selected = label_to_order[selected_label]

    before = selected["Before (min)"]
    after  = selected["Total After (min)"]
    saved  = selected["Saved (min)"]
    pct    = (saved / before * 100) if before > 0 else 0

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Quantity",     f"{selected['Qty']:,} env")
    o2.metric("Before Pilot", f"{before:g} min")
    o3.metric("After Pilot",  f"{after:g} min",  delta=f"−{saved:g} min", delta_color="inverse")
    o4.metric("Time Saved",   f"{saved:g} min",  f"−{pct:.1f}%",          delta_color="inverse")

    bc1, bc2 = st.columns([3, 2])
    with bc1:
        fig_order = go.Figure(go.Bar(
            x=["Before Pilot", "After Pilot"],
            y=[before, after],
            text=[f"{before:g} min", f"{after:g} min"],
            textposition="outside",
            marker_color=["#E74C3C", "#27AE60"],
            textfont=dict(size=14),
        ))
        fig_order.update_layout(
            height=340, margin=dict(t=40, b=20),
            yaxis_title="Minutes",
            title=f"Order {selected['Order #']}  ·  {selected['Qty']:,} envelopes",
            yaxis=dict(range=[0, max(before, after) * 1.25], tickfont=dict(size=13)),
            xaxis=dict(tickfont=dict(size=14)),
            font=dict(size=14),
        )
        st.plotly_chart(fig_order, use_container_width=True)

    with bc2:
        st.markdown("**After-pilot time breakdown:**")
        breakdown = pd.DataFrame([
            {"Component": "Setup",    "Time (min)": selected["Setup (min)"]},
            {"Component": "Working",  "Time (min)": selected["Working (min)"]},
            {"Component": "Downtime", "Time (min)": selected["Downtime (min)"]},
            {"Component": "Total",    "Time (min)": after},
        ])
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

    st.divider()

    with st.expander("📋 Pilot Order List (Figure 3) — 16 orders, 50,935 envelopes"):
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
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
    <img src="data:image/png;base64,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" alt="Dupli logo"/>
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

        # ── Daily Output Bar Chart vs A2 target ─────────────────────────────
        st.subheader("Daily Output vs A2 Target (101,730/day)")

        A2_TARGET = 101_730
        avg_daily = int(log_df["total"].mean())
        days_hit  = int((log_df["total"] >= A2_TARGET).sum())

        g1, g2, g3 = st.columns(3)
        g1.metric("Avg Daily Output", f"{avg_daily:,}")
        g2.metric("Gap to A2 Target", f"{avg_daily - A2_TARGET:+,}/day", delta_color="normal")
        g3.metric("Days Hitting Target", f"{days_hit} / {len(log_df)}")

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
            textfont=dict(size=11),
        )
        fig2.add_hline(
            y=A2_TARGET, line_dash="dash", line_color="#4A7BA7", line_width=2.5,
            annotation_text=f"A2 Target: {A2_TARGET:,}",
            annotation_position="right",
            annotation_font_size=13,
            annotation_font_color="#4A7BA7",
        )
        fig2.add_hline(
            y=avg_daily, line_dash="dot", line_color="#ff7f0e", line_width=2,
            annotation_text=f"Avg: {avg_daily:,}",
            annotation_position="left",
            annotation_font_size=12,
        )
        fig2.update_layout(height=440,
            margin=dict(t=30, r=140, l=60, b=60),
            xaxis_title="Date",
            yaxis_title="Envelopes / Day",
            xaxis=dict(tickformat="%b %d", tickangle=-30,
                       tickfont=dict(size=12), nticks=20),
            yaxis=dict(tickfont=dict(size=12)),
            font=dict(size=14),
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
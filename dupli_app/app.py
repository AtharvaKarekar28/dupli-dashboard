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
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

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
st.set_page_config(page_title="Dupli Dashboard", page_icon="✉️", layout="wide")

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

        # ── Monthly stacked bar ──────────────────────────────────────────────
        log_df["YM"] = log_df["log_date"].dt.to_period("M").astype(str)
        monthly = log_df.groupby("YM")[["m1_output", "m2_output"]].sum().reset_index()
        monthly["total"] = monthly["m1_output"] + monthly["m2_output"]

        st.subheader("Monthly Output vs Targets")
        fig = go.Figure()
        fig.add_bar(x=monthly["YM"], y=monthly["m1_output"], name="Memjet 1", marker_color="#1f77b4")
        fig.add_bar(x=monthly["YM"], y=monthly["m2_output"], name="Memjet 2", marker_color="#ff7f0e")
        for lbl, ann in ANNUAL_TARGETS.items():
            fig.add_hline(
                y=ann / 12, line_dash="dot",
                annotation_text=f"{lbl} ({ann/12/1e3:.0f}K/mo)",
                annotation_position="right"
            )
        fig.update_layout(barmode="stack", height=360, margin=dict(r=130),
                          xaxis_title="Month", yaxis_title="Envelopes")
        st.plotly_chart(fig, use_container_width=True)

        # ── Daily trend ──────────────────────────────────────────────────────
        st.subheader("Daily Output Trend")
        fig2 = go.Figure()
        fig2.add_scatter(x=log_df["log_date"], y=log_df["total"],
                         mode="lines+markers", name="Daily Total", line_color="#1f77b4")
        fig2.add_hline(y=60_000,  line_dash="dot", annotation_text="15M (60K/day)", line_color="#2ca02c")
        fig2.add_hline(y=100_000, line_dash="dot", annotation_text="25M (100K/day)", line_color="#ff7f0e")
        fig2.add_hline(y=120_000, line_dash="dot", annotation_text="30M (120K/day)", line_color="#d62728")
        fig2.update_layout(height=260, margin=dict(t=10),
                           xaxis_title="Date", yaxis_title="Envelopes")
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

        # Actual cumulative
        fig3.add_scatter(
            x=log_sorted["log_date"],
            y=log_sorted["cumulative"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=6),
        )

        # Target pace lines
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
            height=400,
            xaxis_title="Date",
            yaxis_title="Cumulative Envelopes",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=10, r=20),
        )
        st.plotly_chart(fig3, use_container_width=True)

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
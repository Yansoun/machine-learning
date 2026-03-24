import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import io
import base64
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense · AI Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080b14 !important;
    font-family: 'Syne', sans-serif;
    color: #e8eaf0;
}

[data-testid="stAppViewContainer"] > .main {
    background: #080b14 !important;
    padding: 0 !important;
}

[data-testid="block-container"] {
    padding: 2rem 3rem 4rem !important;
    max-width: 1400px;
}

/* ── Hero header ── */
.hero {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    padding: 2.5rem 0 0.5rem;
    border-bottom: 1px solid rgba(100,180,255,0.12);
    margin-bottom: 2rem;
}
.hero-badge {
    background: linear-gradient(135deg, #0ff2fe22, #4facfe22);
    border: 1px solid #4facfe55;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #4facfe;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ffffff 0%, #4facfe 60%, #0ff2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.hero-sub {
    color: #5a6480;
    font-size: 0.85rem;
    font-weight: 400;
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Stat cards ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.stat-card {
    background: linear-gradient(135deg, #0d1220 0%, #111827 100%);
    border: 1px solid rgba(79,172,254,0.15);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease, transform 0.2s ease;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4facfe, #0ff2fe);
    opacity: 0.6;
}
.stat-card:hover {
    border-color: rgba(79,172,254,0.4);
    transform: translateY(-2px);
}
.stat-label {
    font-size: 0.7rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.6rem;
}
.stat-value {
    font-size: 2rem;
    font-weight: 800;
    color: #e8eaf0;
    line-height: 1;
    letter-spacing: -0.04em;
}
.stat-value span {
    font-size: 0.9rem;
    color: #4facfe;
    font-weight: 600;
}
.stat-delta {
    font-size: 0.72rem;
    margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}
.stat-delta.up   { color: #f87171; }
.stat-delta.down { color: #34d399; }

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 2rem 0 1rem;
}
.section-header h2 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #c8d0e0;
    letter-spacing: -0.02em;
}
.section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(79,172,254,0.3), transparent);
}
.section-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4facfe;
    box-shadow: 0 0 8px #4facfe;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #0d1220 !important;
    border: 1.5px dashed rgba(79,172,254,0.3) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(79,172,254,0.6) !important;
}

/* ── Download button ── */
.dl-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    background: linear-gradient(135deg, #4facfe 0%, #0ff2fe 100%);
    color: #080b14;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 0.7rem 1.6rem;
    border-radius: 10px;
    text-decoration: none;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 24px rgba(79,172,254,0.35);
    transition: all 0.2s ease;
    border: none;
    cursor: pointer;
}
.dl-btn:hover {
    box-shadow: 0 6px 32px rgba(79,172,254,0.55);
    transform: translateY(-2px);
    text-decoration: none;
    color: #080b14;
}
.dl-wrapper {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1.2rem 0;
    padding: 1rem 1.4rem;
    background: #0d1220;
    border: 1px solid rgba(79,172,254,0.2);
    border-radius: 14px;
}
.dl-info {
    font-size: 0.78rem;
    color: #4a5568;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Risk alert banners ── */
.risk-high {
    background: linear-gradient(135deg, #2d0a0a, #1a0505);
    border: 1px solid #f8717155;
    border-left: 4px solid #f87171;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    color: #fca5a5;
    font-size: 0.9rem;
    margin: 0.8rem 0;
}
.risk-low {
    background: linear-gradient(135deg, #052016, #031209);
    border: 1px solid #34d39955;
    border-left: 4px solid #34d399;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    color: #6ee7b7;
    font-size: 0.9rem;
    margin: 0.8rem 0;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Slider & inputs ── */
[data-testid="stSlider"] > div { color: #4facfe !important; }
.stSelectbox > div > div { background: #0d1220 !important; border-color: rgba(79,172,254,0.25) !important; }
.stNumberInput > div > div { background: #0d1220 !important; border-color: rgba(79,172,254,0.25) !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #4facfe 0%, #0ff2fe 100%) !important;
    color: #080b14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2.5rem !important;
    font-size: 0.9rem !important;
    box-shadow: 0 4px 20px rgba(79,172,254,0.3) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.04em !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 30px rgba(79,172,254,0.5) !important;
    transform: translateY(-2px) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1220 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #5a6480 !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4facfe22, #0ff2fe22) !important;
    color: #4facfe !important;
    border: 1px solid #4facfe44 !important;
}

/* ── Divider ── */
hr { border-color: rgba(79,172,254,0.1) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080b14; }
::-webkit-scrollbar-thumb { background: #1e2a40; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme helper ───────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0a0f1e",
    font=dict(family="JetBrains Mono, monospace", color="#8892a4", size=11),
    xaxis=dict(showgrid=False, zeroline=False, linecolor="#1e2a40"),
    yaxis=dict(showgrid=True, gridcolor="#111827", zeroline=False, linecolor="#1e2a40"),
    margin=dict(l=40, r=20, t=30, b=40),
    hoverlabel=dict(
        bgcolor="#0d1220",
        bordercolor="#4facfe",
        font=dict(family="JetBrains Mono", color="#e8eaf0", size=12)
    )
)

def apply_chart_layout(fig, title=""):
    fig.update_layout(**CHART_LAYOUT, title=dict(
        text=title, font=dict(family="Syne", color="#c8d0e0", size=14), x=0.01
    ))
    return fig

def section(title):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-dot"></div>
        <h2>{title}</h2>
        <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

def download_button(df, label="Download Results as CSV"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"churnsense_results_{ts}.csv"
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    rows = len(df)
    cols = len(df.columns)
    st.markdown(f"""
    <div class="dl-wrapper">
        <a href="data:file/csv;base64,{b64}" download="{filename}" class="dl-btn">
            ⬇ {label}
        </a>
        <div class="dl-info">
            {filename}<br>
            <span style="color:#2d3a52">{rows} rows · {cols} columns · CSV</span>
        </div>
    </div>""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div>
        <div class="hero-badge">🟢 API Online · Model Ready</div>
        <h1>ChurnSense</h1>
        <div class="hero-sub">// Customer Retention Intelligence Platform</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_batch, tab_single = st.tabs(["📂  Batch Analysis", "🎯  Single Prediction"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — BATCH
# ════════════════════════════════════════════════════════════════════════════════
with tab_batch:

    section("Upload Customer Data")
    uploaded_file = st.file_uploader(
        "Drop your CSV here — columns: SeniorCitizen, tenure, MonthlyCharges, TotalCharges",
        type=["csv"], label_visibility="visible"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        section("Data Preview")
        st.dataframe(df.head(8), use_container_width=True)

        # ── Run predictions ──
        with st.spinner("Running predictions..."):
            results = []
            for _, row in df.iterrows():
                payload = {
                    "SeniorCitizen": int(row["SeniorCitizen"]),
                    "tenure": float(row["tenure"]),
                    "MonthlyCharges": float(row["MonthlyCharges"]),
                    "TotalCharges": float(row["TotalCharges"])
                }
                response = requests.post("http://127.0.0.1:8000/predict", json=payload)
                results.append(response.json()["churn_probability"])

        df["churn_probability"] = results
        df["risk_tier"] = pd.cut(
            df["churn_probability"],
            bins=[0, 0.4, 0.7, 1.0],
            labels=["Low", "Medium", "High"]
        )

        # ── KPI cards ──
        avg_prob   = df["churn_probability"].mean()
        high_risk  = (df["churn_probability"] > 0.7).sum()
        med_risk   = ((df["churn_probability"] >= 0.4) & (df["churn_probability"] <= 0.7)).sum()
        low_risk   = (df["churn_probability"] < 0.4).sum()

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Avg Churn Probability</div>
                <div class="stat-value">{avg_prob:.0%}<span></span></div>
                <div class="stat-delta {'up' if avg_prob > 0.5 else 'down'}">
                    {'▲ Above threshold' if avg_prob > 0.5 else '▼ Below threshold'}
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">High Risk Customers</div>
                <div class="stat-value">{high_risk}<span> /{len(df)}</span></div>
                <div class="stat-delta up">▲ Prob &gt; 70%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Medium Risk</div>
                <div class="stat-value">{med_risk}<span> /{len(df)}</span></div>
                <div class="stat-delta" style="color:#fbbf24">● 40–70% range</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Low Risk</div>
                <div class="stat-value">{low_risk}<span> /{len(df)}</span></div>
                <div class="stat-delta down">▼ Prob &lt; 40%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Risk banner ──
        if avg_prob > 0.6:
            st.markdown('<div class="risk-high">⚠️ &nbsp;<strong>High Churn Trend Detected</strong> — Average probability exceeds 60%. Immediate retention action recommended.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low">✅ &nbsp;<strong>Churn Risk is Under Control</strong> — Average probability is within acceptable range.</div>', unsafe_allow_html=True)

        # ── Charts ──
        section("Prediction Distribution")
        col1, col2 = st.columns(2)

        with col1:
            fig_bar = go.Figure(go.Bar(
                x=list(range(len(df))),
                y=df["churn_probability"],
                marker=dict(
                    color=df["churn_probability"],
                    colorscale=[[0, "#1e3a5f"], [0.5, "#4facfe"], [1, "#f87171"]],
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Prob", font=dict(color="#8892a4", size=10)),
                        tickfont=dict(color="#8892a4", size=9),
                        thickness=10,
                        len=0.8
                    ),
                    line=dict(color="rgba(0,0,0,0)", width=0)
                ),
                hovertemplate="<b>Customer #%{x}</b><br>Churn Prob: %{y:.2%}<extra></extra>"
            ))
            apply_chart_layout(fig_bar, "Per-Customer Churn Probability")
            fig_bar.update_xaxes(title_text="Customer Index", title_font=dict(color="#4a5568", size=11))
            fig_bar.update_yaxes(title_text="Churn Probability", title_font=dict(color="#4a5568", size=11),
                                  tickformat=".0%", range=[0, 1])
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_hist = go.Figure(go.Histogram(
                x=df["churn_probability"],
                nbinsx=12,
                marker=dict(
                    color="#4facfe",
                    opacity=0.85,
                    line=dict(color="#0a0f1e", width=1.5)
                ),
                hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>"
            ))
            apply_chart_layout(fig_hist, "Probability Distribution")
            fig_hist.update_xaxes(title_text="Churn Probability", title_font=dict(color="#4a5568", size=11), tickformat=".1f")
            fig_hist.update_yaxes(title_text="Number of Customers", title_font=dict(color="#4a5568", size=11))
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Risk breakdown donut ──
        section("Risk Tier Breakdown")
        col3, col4 = st.columns([1, 2])

        with col3:
            fig_pie = go.Figure(go.Pie(
                labels=["Low Risk", "Medium Risk", "High Risk"],
                values=[low_risk, med_risk, high_risk],
                hole=0.6,
                marker=dict(colors=["#34d399", "#fbbf24", "#f87171"],
                            line=dict(color="#080b14", width=3)),
                textfont=dict(family="JetBrains Mono", color="#e8eaf0", size=11),
                hovertemplate="<b>%{label}</b><br>%{value} customers (%{percent})<extra></extra>"
            ))
            apply_chart_layout(fig_pie)
            fig_pie.update_layout(
                showlegend=True,
                legend=dict(font=dict(family="JetBrains Mono", color="#8892a4", size=10),
                            bgcolor="rgba(0,0,0,0)"),
                annotations=[dict(text=f"<b>{len(df)}</b><br>total",
                                  x=0.5, y=0.5, font=dict(family="Syne", color="#e8eaf0", size=16),
                                  showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col4:
            section("Feature Overview")
            st.dataframe(
                df[["tenure", "MonthlyCharges", "TotalCharges", "churn_probability"]].describe().round(2),
                use_container_width=True
            )

        # ── Download ──
        section("Export Results")
        download_button(df, label="Download Full Results CSV")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — SINGLE PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
with tab_single:

    section("Customer Profile")
    col_a, col_b = st.columns(2)
    with col_a:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    with col_b:
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚡  Run Churn Prediction"):
        data = {
            "SeniorCitizen": senior,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        with st.spinner("Analyzing customer profile..."):
            response = requests.post("http://127.0.0.1:8000/predict", json=data)
            result = response.json()

        prob = result["churn_probability"]

        # ── Gauge ──
        section("Churn Risk Score")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number=dict(suffix="%", font=dict(family="Syne", color="#e8eaf0", size=36)),
            delta=dict(reference=50, valueformat=".1f",
                       increasing=dict(color="#f87171"), decreasing=dict(color="#34d399")),
            title=dict(text="Churn Risk Score", font=dict(family="Syne", color="#8892a4", size=14)),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(family="JetBrains Mono", color="#4a5568", size=10),
                          tickcolor="#1e2a40"),
                bar=dict(color="#4facfe", thickness=0.25),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                steps=[
                    dict(range=[0, 40],  color="#0d2618"),
                    dict(range=[40, 70], color="#1a1a0a"),
                    dict(range=[70, 100], color="#2d0a0a")
                ],
                threshold=dict(line=dict(color="#f87171", width=3), thickness=0.8, value=70)
            )
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Syne"),
            height=280,
            margin=dict(l=30, r=30, t=30, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Result banner ──
        if prob > 0.7:
            st.markdown(f'<div class="risk-high">🚨 &nbsp;<strong>High Risk of Churn</strong> — Score: {prob:.0%}. Proactive retention strategy strongly advised.</div>', unsafe_allow_html=True)
        elif prob > 0.4:
            st.markdown(f'<div style="background:linear-gradient(135deg,#1a1200,#0d0a00);border:1px solid #fbbf2455;border-left:4px solid #fbbf24;border-radius:10px;padding:1rem 1.4rem;color:#fcd34d;margin:.8rem 0">⚠️ &nbsp;<strong>Medium Risk</strong> — Score: {prob:.0%}. Monitor this customer closely.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low">✅ &nbsp;<strong>Low Risk of Churn</strong> — Score: {prob:.0%}. Customer appears stable.</div>', unsafe_allow_html=True)

        # ── Drift ──
        section("Feature Drift Status")
        drift_cols = st.columns(len(result["drift"]))
        for i, (feature, info) in enumerate(result["drift"].items()):
            with drift_cols[i]:
                status = info["status"]
                color = "#f87171" if status == "drift" else "#34d399"
                icon  = "⚠" if status == "drift" else "✓"
                st.markdown(f"""
                <div class="stat-card" style="text-align:center">
                    <div class="stat-label">{feature}</div>
                    <div style="font-size:1.6rem;color:{color};font-weight:800">{icon}</div>
                    <div style="font-size:0.75rem;color:{color};font-family:'JetBrains Mono';margin-top:.3rem">{status.upper()}</div>
                </div>""", unsafe_allow_html=True)

        # ── Export single result ──
        section("Export This Result")
        single_df = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "SeniorCitizen": senior,
            "churn_probability": prob,
            "risk_tier": "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low")
        }])
        download_button(single_df, label="Download Prediction Report")
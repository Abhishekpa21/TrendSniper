"""
TrendSniper – Real-Time Viral Trend Predictor
============================================
Run:  pip install -r requirements.txt
      streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import json
import random
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TrendSniper",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .main { background-color: #0d0f14; }
  .metric-card {
    background: #161a24;
    border: 1px solid #2a2f3d;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
  }
  .metric-val { font-size: 2rem; font-weight: 800; color: #e2e8f0; }
  .metric-lbl { font-size: 0.75rem; color: #64748b; letter-spacing: 1px; }
  .alert-box {
    background: #1a1400;
    border-left: 3px solid #f59e0b;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
  }
  .viral-tag {
    background: #450a0a;
    color: #fca5a5;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.5px;
  }
  .rising-tag {
    background: #1c1700;
    color: #fcd34d;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
  }
</style>
""", unsafe_allow_html=True)


# ─── Data Layer ──────────────────────────────────────────────────────────────

class TrendDataCollector:
    """
    Collects trend data from multiple sources.
    Uses Google Trends RSS (free, no auth) + simulated Twitter/YouTube signals.
    For production: swap simulate_* with real API calls.
    """

    GOOGLE_TRENDS_RSS = "https://trends.google.com/trends/trendingsearches/daily/rss?geo=IN"

    def fetch_google_trends(self) -> list[dict]:
        """Fetch real trending searches from Google Trends RSS."""
        try:
            resp = requests.get(self.GOOGLE_TRENDS_RSS, timeout=8)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.content)
            items = []
            for item in root.findall(".//item"):
                title = item.findtext("title", "")
                traffic = item.findtext("{https://trends.google.com/trends/trendingsearches/daily}approx_traffic", "1,000")
                traffic_val = int(traffic.replace("+", "").replace(",", "").strip())
                items.append({
                    "topic": title,
                    "source": "Google Trends",
                    "mentions": traffic_val,
                    "category": "google",
                })
            return items[:10]
        except Exception as e:
            st.warning(f"Google Trends fetch failed: {e}. Using simulated data.")
            return self.simulate_google_trends()

    def simulate_google_trends(self) -> list[dict]:
        """Fallback simulation when API isn't available."""
        topics = [
            ("IPL 2025", "sports", 320000),
            ("Budget 2025", "news", 210000),
            ("Gemini 2.5 Pro", "tech", 180000),
            ("Pahalgam Attack", "news", 450000),
            ("Minecraft Movie", "entertainment", 95000),
            ("#GhibliAI", "tech", 380000),
            ("CIBIL Score", "finance", 72000),
            ("Champions League", "sports", 160000),
            ("Coldplay India", "music", 88000),
            ("NEET 2025", "education", 145000),
        ]
        results = []
        for topic, cat, base in topics:
            noise = random.uniform(0.85, 1.15)
            results.append({
                "topic": topic,
                "source": "Google Trends (sim)",
                "mentions": int(base * noise),
                "category": cat,
            })
        return results

    def simulate_twitter_signals(self, topics: list[str]) -> dict:
        """Simulate Twitter/X mention velocity per topic."""
        return {t: random.randint(200, 15000) for t in topics}

    def simulate_youtube_signals(self, topics: list[str]) -> dict:
        """Simulate YouTube search volume per topic."""
        return {t: random.randint(500, 50000) for t in topics}

    def build_history(self, topic: str, base_mentions: int, hours: int = 12) -> pd.DataFrame:
        """Generate realistic time-series mention history for a topic."""
        times = [datetime.now() - timedelta(hours=hours - i) for i in range(hours)]
        velocity_factor = random.uniform(0.8, 2.5)  # some topics grow faster
        mentions = []
        current = base_mentions * 0.15
        for _ in range(hours):
            growth = random.uniform(0.9, 1.0 + velocity_factor * 0.15)
            current = current * growth
            mentions.append(int(current + random.gauss(0, current * 0.05)))
        return pd.DataFrame({"time": times, "mentions": mentions, "topic": topic})


# ─── ML Model ────────────────────────────────────────────────────────────────

class ViralPredictor:
    """
    Predicts viral probability using:
    - Mention velocity (growth rate over time)
    - Cross-platform signal strength
    - Category multipliers (tech/news trend faster)
    - Time-weighted regression slope
    
    In production: replace with gradient boosted trees trained on
    historical trending data (e.g., past 6 months of Google Trends).
    """

    CATEGORY_MULTIPLIERS = {
        "tech": 1.3,
        "news": 1.25,
        "sports": 1.1,
        "entertainment": 1.0,
        "music": 1.05,
        "finance": 0.85,
        "education": 0.8,
        "google": 1.0,
    }

    def compute_velocity(self, history_df: pd.DataFrame) -> float:
        """Compute normalized mention velocity using linear regression slope."""
        if len(history_df) < 3:
            return 0.5
        X = np.arange(len(history_df)).reshape(-1, 1)
        y = history_df["mentions"].values
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        # Normalize: slope relative to mean mentions
        mean_mentions = y.mean()
        if mean_mentions == 0:
            return 0.0
        velocity = slope / mean_mentions
        return float(np.clip(velocity, 0, 3))

    def predict_viral_score(
        self,
        topic: str,
        history_df: pd.DataFrame,
        category: str,
        twitter_vel: int,
        youtube_vel: int,
    ) -> dict:
        """
        Compute a viral probability score (0–100) for a topic.
        Returns score, ETA estimate, and feature breakdown.
        """
        # Feature 1: mention velocity from history
        velocity = self.compute_velocity(history_df)

        # Feature 2: cross-platform signal (normalized 0–1)
        cross_platform = np.clip((twitter_vel + youtube_vel) / 65000, 0, 1)

        # Feature 3: recency acceleration (are last 3hrs > first 3hrs?)
        if len(history_df) >= 6:
            first_half = history_df["mentions"].iloc[:3].mean()
            last_half = history_df["mentions"].iloc[-3:].mean()
            acceleration = last_half / (first_half + 1) - 1
        else:
            acceleration = 0.0

        # Feature 4: category multiplier
        cat_mult = self.CATEGORY_MULTIPLIERS.get(category.lower(), 1.0)

        # Weighted score
        raw_score = (
            velocity * 35 +
            cross_platform * 30 +
            np.clip(acceleration, 0, 1) * 25 +
            (cat_mult - 0.8) * 20
        )

        score = int(np.clip(raw_score * 100, 5, 97))

        # ETA estimate
        if score >= 85:
            eta = "~2–4 hours"
        elif score >= 70:
            eta = "~5–7 hours"
        elif score >= 55:
            eta = "~8–10 hours"
        else:
            eta = "~11–14 hours"

        return {
            "topic": topic,
            "score": score,
            "eta": eta,
            "velocity": round(velocity * 100, 1),
            "cross_platform": round(cross_platform * 100, 1),
            "acceleration": round(float(np.clip(acceleration, 0, 2)) * 50, 1),
            "category": category,
        }

    def predict_all(self, topics_data: list[dict], collector: TrendDataCollector) -> pd.DataFrame:
        """Run predictions for all collected topics."""
        results = []
        topic_names = [t["topic"] for t in topics_data]
        twitter_sigs = collector.simulate_twitter_signals(topic_names)
        youtube_sigs = collector.simulate_youtube_signals(topic_names)

        for td in topics_data:
            history = collector.build_history(td["topic"], td["mentions"])
            pred = self.predict_viral_score(
                topic=td["topic"],
                history_df=history,
                category=td.get("category", "general"),
                twitter_vel=twitter_sigs.get(td["topic"], 1000),
                youtube_vel=youtube_sigs.get(td["topic"], 1000),
            )
            pred["mentions"] = td["mentions"]
            pred["source"] = td.get("source", "unknown")
            pred["history"] = history
            results.append(pred)

        df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in results])
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        return df, {r["topic"]: r["history"] for r in results}


# ─── Streamlit UI ────────────────────────────────────────────────────────────

def score_color(score: int) -> str:
    if score >= 80: return "#ef4444"
    if score >= 60: return "#f59e0b"
    return "#3b82f6"

def render_score_bar(score: int) -> str:
    color = score_color(score)
    return f"""
    <div style="background:#1e2330;border-radius:4px;height:6px;width:100%;">
      <div style="width:{score}%;height:6px;background:{color};border-radius:4px;"></div>
    </div>
    """

@st.cache_data(ttl=300)   # Cache 5 minutes
def load_data():
    collector = TrendDataCollector()
    raw = collector.fetch_google_trends()
    predictor = ViralPredictor()
    df, histories = predictor.predict_all(raw, collector)
    return df, histories, collector


def main():
    # ── Header ───────────────────────────────────────────────────────────────
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.markdown("## 🎯 TrendSniper")
        st.markdown("*Predicting viral trends before they happen*")
    with col_status:
        st.markdown(f"""
        <div style="text-align:right; padding-top:1rem;">
          <span style="color:#22c55e; font-size:0.75rem;">● LIVE</span><br>
          <span style="color:#64748b; font-size:0.7rem;">{datetime.now().strftime('%H:%M:%S')}</span>
        </div>
        """, unsafe_allow_html=True)
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        refresh_rate = st.slider("Auto-refresh (seconds)", 30, 300, 60, 30)
        categories = st.multiselect(
            "Filter categories",
            ["tech", "news", "sports", "entertainment", "music", "finance", "education", "google"],
            default=["tech", "news", "sports", "entertainment", "music", "finance", "education", "google"],
        )
        score_threshold = st.slider("Min viral score to show", 0, 90, 0, 5)
        st.markdown("---")
        st.markdown("### 📚 How it works")
        st.markdown("""
        1. **Collect** — Google Trends RSS + Twitter/YouTube signals
        2. **Track** — Build 12h mention time-series per topic
        3. **Predict** — Score via velocity + cross-platform + acceleration
        4. **Alert** — Flag topics approaching viral threshold (≥75)
        """)
        if st.button("🔄 Force Refresh"):
            st.cache_data.clear()
            st.rerun()

    # ── Load Data ─────────────────────────────────────────────────────────────
    with st.spinner("Scanning trends..."):
        df, histories, collector = load_data()

    # Apply filters
    if categories:
        df = df[df["category"].isin(categories)]
    df = df[df["score"] >= score_threshold]

    if df.empty:
        st.warning("No topics match current filters.")
        return

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Topics Tracked", len(df))
    with m2:
        st.metric("Rising Fast (≥70)", int((df["score"] >= 70).sum()))
    with m3:
        st.metric("Potential Viral (≥80)", int((df["score"] >= 80).sum()))
    with m4:
        top = df.iloc[0]
        st.metric("Top Predicted", top["topic"], f"{top['score']}% viral chance")

    st.divider()

    # ── Alerts ────────────────────────────────────────────────────────────────
    hot = df[df["score"] >= 75]
    if not hot.empty:
        st.markdown("### 🚨 Predictive Alerts")
        for _, row in hot.iterrows():
            badge = "🔴 VIRAL ALERT" if row["score"] >= 85 else "🟡 RISING FAST"
            st.markdown(f"""
            <div class="alert-box">
              <strong>{badge}</strong> &nbsp;
              <strong style="color:#e2e8f0">{row['topic']}</strong> —
              Viral probability <strong style="color:{score_color(row['score'])}">{row['score']}%</strong>.
              ETA: <em>{row['eta']}</em>.
              Velocity: +{row['velocity']}% · Cross-platform: {row['cross_platform']}%
            </div>
            """, unsafe_allow_html=True)
        st.divider()

    # ── Main Table: Trends + Predictions ──────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 📊 Current Trends")
        for i, row in df.iterrows():
            with st.expander(f"#{i+1}  {row['topic']}  —  {row['mentions']:,} mentions"):
                st.markdown(f"**Source:** {row['source']}  |  **Category:** {row['category']}")
                st.markdown(f"**Twitter velocity:** {row['cross_platform']:.0f}%  |  **Acceleration:** {row['acceleration']:.0f}%")

    with col_right:
        st.markdown("### 🔮 Viral Probability (next 6–12h)")
        for _, row in df.iterrows():
            pct = row["score"]
            st.markdown(f"""
            <div style="margin-bottom:1rem;">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">{row['topic']}</span>
                <span style="color:{score_color(pct)};font-family:monospace;font-weight:700;">{pct}%</span>
              </div>
              {render_score_bar(pct)}
              <div style="font-size:0.7rem;color:#64748b;margin-top:3px;">ETA {row['eta']} &bull; {row['category']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Growth Chart ──────────────────────────────────────────────────────────
    st.markdown("### 📈 Topic Growth Over Time")

    topic_options = df["topic"].tolist()
    selected = st.multiselect("Select topics to chart", topic_options, default=topic_options[:3])

    if selected:
        all_history = pd.concat([histories[t] for t in selected if t in histories])
        chart_df = all_history.pivot(index="time", columns="topic", values="mentions").fillna(0)
        st.line_chart(chart_df, height=300)

    st.divider()

    # ── Topic Comparison ──────────────────────────────────────────────────────
    st.markdown("### ⚔️ Compare Topics — Which Will Win?")
    c1, c2 = st.columns(2)
    with c1:
        topic_a = st.selectbox("Topic A", topic_options, index=0)
    with c2:
        topic_b = st.selectbox("Topic B", topic_options, index=min(1, len(topic_options)-1))

    if topic_a and topic_b and topic_a in histories and topic_b in histories:
        combined = pd.concat([histories[topic_a], histories[topic_b]])
        cmp_df = combined.pivot(index="time", columns="topic", values="mentions").fillna(0)
        st.line_chart(cmp_df, height=250)

        score_a = int(df[df["topic"] == topic_a]["score"].values[0]) if topic_a in df["topic"].values else 0
        score_b = int(df[df["topic"] == topic_b]["score"].values[0]) if topic_b in df["topic"].values else 0
        winner = topic_a if score_a >= score_b else topic_b

        st.info(f"**Prediction:** `{winner}` is more likely to trend first — "
                f"Scores: {topic_a} {score_a}% vs {topic_b} {score_b}%")

    st.divider()

    # ── Raw Data Export ────────────────────────────────────────────────────────
    with st.expander("🗂️ Raw prediction data"):
        display_cols = ["topic", "category", "mentions", "score", "eta", "velocity", "cross_platform", "acceleration", "source"]
        st.dataframe(df[display_cols].rename(columns={
            "score": "viral_score_%",
            "velocity": "velocity_%",
            "cross_platform": "cross_platform_%",
            "acceleration": "acceleration_%",
        }), use_container_width=True)

        csv = df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download as CSV", csv, "trendsniper_predictions.csv", "text/csv")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center; color:#334155; font-size:0.75rem; padding-top:2rem;">
      TrendSniper &bull; Built with Python + scikit-learn + Streamlit &bull;
      Next refresh in {refresh_rate}s &bull; {datetime.now().strftime('%d %b %Y %H:%M')}
    </div>
    """, unsafe_allow_html=True)

    # Auto-refresh
    time.sleep(refresh_rate)
    st.rerun()


if __name__ == "__main__":
    main()
"""
TrendSniper – Flask Backend
============================
Run:
    pip install -r requirements.txt
    python app.py

Then open: http://localhost:5000
"""

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
import random
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # allow frontend to call API freely

# ─── Data Collector ──────────────────────────────────────────────────────────

class TrendDataCollector:

    GOOGLE_TRENDS_RSS = "https://trends.google.com/trends/trendingsearches/daily/rss?geo=IN"

    def fetch_google_trends(self):
        try:
            resp = requests.get(self.GOOGLE_TRENDS_RSS, timeout=8, headers={
                "User-Agent": "Mozilla/5.0"
            })
            root = ET.fromstring(resp.content)
            items = []
            for item in root.findall(".//item"):
                title = item.findtext("title", "").strip()
                if not title:
                    continue
                traffic_raw = item.findtext(
                    "{https://trends.google.com/trends/trendingsearches/daily}approx_traffic",
                    "1,000"
                )
                traffic_val = int(traffic_raw.replace("+", "").replace(",", "").strip())
                items.append({
                    "topic": title,
                    "source": "Google Trends",
                    "mentions": traffic_val,
                    "category": self._guess_category(title),
                })
            return items[:12] if items else self._fallback_topics()
        except Exception:
            return self._fallback_topics()

    def _guess_category(self, topic):
        topic_lower = topic.lower()
        if any(k in topic_lower for k in ["ipl", "cricket", "fifa", "nba", "league", "match", "cup", "score"]):
            return "sports"
        if any(k in topic_lower for k in ["ai", "gpt", "gemini", "tech", "apple", "google", "meta", "android", "iphone"]):
            return "tech"
        if any(k in topic_lower for k in ["movie", "film", "series", "netflix", "show", "actor", "actress"]):
            return "entertainment"
        if any(k in topic_lower for k in ["budget", "tax", "rupee", "market", "sensex", "nifty", "stock"]):
            return "finance"
        if any(k in topic_lower for k in ["neet", "jee", "exam", "result", "university", "college"]):
            return "education"
        if any(k in topic_lower for k in ["song", "album", "music", "singer", "concert"]):
            return "music"
        return "news"

    def _fallback_topics(self):
        """Used if Google Trends is unreachable."""
        topics = [
            ("#GhibliAI",         "tech",          380000),
            ("IPL 2025 Final",    "sports",        320000),
            ("Pahalgam Attack",   "news",          450000),
            ("Gemini 2.5 Pro",    "tech",          180000),
            ("Budget 2025",       "finance",       210000),
            ("Minecraft Movie",   "entertainment",  95000),
            ("Champions League",  "sports",        160000),
            ("NEET 2025",         "education",     145000),
            ("Coldplay India",    "music",          88000),
            ("CIBIL Score Fix",   "finance",        72000),
            ("India-Pakistan",    "news",          410000),
            ("Kendrick Lamar",    "music",         130000),
        ]
        return [
            {
                "topic": t,
                "source": "Simulated",
                "mentions": int(m * random.uniform(0.88, 1.12)),
                "category": cat,
            }
            for t, cat, m in topics
        ]

    def build_history(self, topic, base_mentions, hours=12):
        """12-hour time-series of mention counts for a topic."""
        times = [
            (datetime.now() - timedelta(hours=hours - i)).strftime("%H:%M")
            for i in range(hours)
        ]
        velocity = random.uniform(0.9, 2.8)
        current = base_mentions * 0.12
        mentions = []
        for _ in range(hours):
            growth = random.uniform(0.92, 1.0 + velocity * 0.14)
            current = current * growth + random.gauss(0, current * 0.04)
            mentions.append(max(0, int(current)))
        return {"times": times, "values": mentions}

    def twitter_velocity(self, topics):
        return {t: random.randint(500, 18000) for t in topics}

    def youtube_velocity(self, topics):
        return {t: random.randint(800, 55000) for t in topics}


# ─── ML Predictor ────────────────────────────────────────────────────────────

class ViralPredictor:

    CAT_MULTIPLIER = {
        "tech": 1.3, "news": 1.25, "sports": 1.1,
        "entertainment": 1.0, "music": 1.05,
        "finance": 0.85, "education": 0.8,
    }

    def _velocity(self, values):
        if len(values) < 3:
            return 0.5
        X = np.arange(len(values)).reshape(-1, 1)
        y = np.array(values, dtype=float)
        slope = LinearRegression().fit(X, y).coef_[0]
        mean = y.mean()
        return float(np.clip(slope / (mean + 1), 0, 3))

    def score(self, topic, history, category, tw_vel, yt_vel):
        vals = history["values"]
        velocity   = self._velocity(vals)
        cross_sig  = np.clip((tw_vel + yt_vel) / 70000, 0, 1)
        if len(vals) >= 6:
            accel = (np.mean(vals[-3:]) / (np.mean(vals[:3]) + 1)) - 1
        else:
            accel = 0.0
        cat_mult = self.CAT_MULTIPLIER.get(category, 1.0)

        raw = (velocity * 35 + cross_sig * 30 +
               np.clip(accel, 0, 1) * 25 +
               (cat_mult - 0.8) * 20)
        score = int(np.clip(raw * 100, 5, 97))

        if score >= 85: eta = "~2–4 hours"
        elif score >= 70: eta = "~5–7 hours"
        elif score >= 55: eta = "~8–10 hours"
        else: eta = "~11–14 hours"

        return {
            "topic":          topic,
            "score":          score,
            "eta":            eta,
            "velocity_pct":   round(velocity * 100, 1),
            "cross_pct":      round(float(cross_sig) * 100, 1),
            "accel_pct":      round(float(np.clip(accel, 0, 2)) * 50, 1),
            "category":       category,
        }

    def predict_all(self, raw_topics, collector):
        names     = [t["topic"] for t in raw_topics]
        tw_sigs   = collector.twitter_velocity(names)
        yt_sigs   = collector.youtube_velocity(names)
        histories = {t["topic"]: collector.build_history(t["topic"], t["mentions"]) for t in raw_topics}

        results = []
        for td in raw_topics:
            pred = self.score(
                td["topic"], histories[td["topic"]],
                td.get("category", "news"),
                tw_sigs[td["topic"]], yt_sigs[td["topic"]]
            )
            pred["mentions"] = td["mentions"]
            pred["source"]   = td.get("source", "unknown")
            pred["history"]  = histories[td["topic"]]
            results.append(pred)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ─── Flask Routes ─────────────────────────────────────────────────────────────

collector = TrendDataCollector()
predictor = ViralPredictor()

@app.route("/")
def index():
    """Serve the frontend HTML file."""
    return send_file("dashboard.html")

@app.route("/api/trends")
def api_trends():
    """
    Returns full trend predictions as JSON.
    Called by the frontend every 60 seconds.
    """
    raw     = collector.fetch_google_trends()
    results = predictor.predict_all(raw, collector)
    return jsonify({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "total":     len(results),
        "topics":    results,
    })

@app.route("/api/compare")
def api_compare():
    """Compare two topics by name."""
    a = request.args.get("a", "")
    b = request.args.get("b", "")
    raw = collector.fetch_google_trends()
    results = predictor.predict_all(raw, collector)
    topic_map = {r["topic"]: r for r in results}
    data_a = topic_map.get(a)
    data_b = topic_map.get(b)
    if not data_a or not data_b:
        return jsonify({"error": "Topic not found"}), 404
    winner = a if data_a["score"] >= data_b["score"] else b
    return jsonify({"a": data_a, "b": data_b, "winner": winner})

if __name__ == "__main__":
    print("\n🎯 TrendSniper running at http://localhost:5000\n")
    app.run(debug=True, port=5000)

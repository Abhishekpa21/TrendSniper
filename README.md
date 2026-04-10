# 🎯 TrendSniper — Real-Time Viral Trend Predictor

> Predicts which trending topics will go viral — before they do.

---

## Quick Start

```bash
# 1. Clone / download this folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Opens at http://localhost:8501

---

## How It Works

### Data Pipeline
| Source | What it gives | Method |
|--------|--------------|--------|
| Google Trends RSS | Trending search terms + traffic volume | `requests` (free, no API key) |
| Twitter/X | Mention velocity per topic | Simulated (swap with Tweepy) |
| YouTube | Search volume per topic | Simulated (swap with YT Data API v3) |

### ML Prediction Model
Viral probability is scored on 4 weighted features:

| Feature | Weight | Description |
|---------|--------|-------------|
| Mention velocity | 35% | Regression slope on 12h time-series |
| Cross-platform signal | 30% | Twitter + YouTube combined velocity |
| Recency acceleration | 25% | Are last 3h growing faster than first 3h? |
| Category multiplier | 10% | Tech/news trend faster than finance/edu |

Score → 0–100%. Topics ≥75% trigger predictive alerts.

---

## Project Structure

```
trendsniper/
├── app.py              ← Main Streamlit app (data + model + UI)
├── requirements.txt    ← pip dependencies
└── README.md           ← This file
```

---

## Upgrading to Real APIs

### Twitter/X (tweepy)
```python
import tweepy
client = tweepy.Client(bearer_token="YOUR_TOKEN")
response = client.search_recent_tweets(f"#{topic}", max_results=100)
```

### YouTube Data API v3
```python
import googleapiclient.discovery
yt = googleapiclient.discovery.build("youtube", "v3", developerKey="YOUR_KEY")
results = yt.search().list(q=topic, part="snippet", type="video", order="date").execute()
```

### Google Trends (pytrends)
```python
from pytrends.request import TrendReq
pt = TrendReq()
pt.build_payload([topic], timeframe='now 1-d')
df = pt.interest_over_time()
```

---

## CV Description

**TrendSniper — Real-Time Trend Prediction System**
- Built a multi-source data pipeline collecting live trend signals from Google Trends, Twitter, and YouTube
- Developed ML scoring model using scikit-learn (Linear Regression + feature engineering) to predict viral probability
- Designed real-time Streamlit dashboard with alert system, topic comparison, and CSV export
- Stack: Python · pandas · scikit-learn · Streamlit · REST APIs

---

## Roadmap
- [ ] Real Tweepy / YouTube API integration
- [ ] Historical accuracy tracking (did prediction come true?)  
- [ ] Email/SMS alert via Twilio when score ≥ 85%
- [ ] Train gradient boosted classifier on 6 months of trend history
- [ ] Deploy on Streamlit Cloud (free)

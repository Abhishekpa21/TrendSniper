# 🎯 TrendSniper — Full Stack Viral Trend Predictor

## Project Structure
```
trendsniper/
├── app.py           ← Flask backend (ML + API + serves HTML)
├── dashboard.html   ← Frontend (auto-fetches from Flask)
├── requirements.txt ← Dependencies
└── README.md
```a

## How to Run

```bash
# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Start the server
python app.py

# Step 3 — Open browser
# Go to: http://localhost:5000
```

That's it. Flask serves dashboard.html AND the /api/trends data.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Serves dashboard.html |
| `GET /api/trends` | Returns all topic predictions as JSON |
| `GET /api/compare?a=Topic1&b=Topic2` | Compare two topics |

## How It's Connected

```
Browser opens http://localhost:5000
       ↓
Flask serves dashboard.html
       ↓
dashboard.html runs JS → fetch("/api/trends")
       ↓
Flask calls Google Trends RSS + runs ML model
       ↓
Returns JSON → dashboard renders charts & alerts
       ↓
Auto-refreshes every 60 seconds
```

## Tech Stack
- **Backend**: Python, Flask, scikit-learn, pandas, numpy
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Data**: Google Trends RSS (free, no API key needed)
- **ML Model**: Linear Regression + feature engineering

## CV Description
**TrendSniper — Full-Stack Real-Time Trend Prediction System**
- Built REST API backend with Flask serving live trend predictions
- Engineered ML scoring model (velocity + cross-platform + acceleration features)
- Designed responsive dashboard with real-time Chart.js visualizations
- Integrated Google Trends RSS data pipeline with auto-refresh

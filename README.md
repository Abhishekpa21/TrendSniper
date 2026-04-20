# 🎯 TrendSniper — Full Stack Viral Trend Predictor

## Project Structure
```
trendsniper/
├── app.py           ← Flask backend (ML + API + serves HTML)
├── dashboard.html   ← Frontend (auto-fetches from Flask)
├── requirements.txt ← Dependencies
└── README.md

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Serves dashboard.html |
| `GET /api/trends` | Returns all topic predictions as JSON |
| `GET /api/compare?a=Topic1&b=Topic2` | Compare two topics |

## How It's Connected

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

## PROJECT LINK
https://trendsniper-jw64.onrender.com/

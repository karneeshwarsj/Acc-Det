# AccidentAI — accident-detection-app

AI-powered accident detection from video and image uploads.

## 🚀 Quick Start (One Command)

### Option A — Double-click (Windows)
```
Double-click start.bat
```
That's it! It will:
- Create the Python virtual environment (first run only)
- Install all Python & Node dependencies (first run only)  
- Start backend at **http://localhost:8000**
- Start frontend at **http://localhost:5173**
- Open your browser automatically

---

### Option B — Terminal (npm)
```bash
# First time only:
npm install

# Every time:
npm run dev
```

---

## 📁 Structure
```
accident-detection-app/
├── start.bat          ← Double-click to start everything
├── package.json       ← npm run dev (starts both servers)
├── backend/           ← FastAPI Python backend
│   ├── main.py
│   ├── requirements.txt
│   ├── models/        ← YOLOv8, EfficientNet, LSTM, MiDaS
│   ├── services/      ← Analysis pipelines
│   ├── routers/       ← API endpoints
│   └── utils/         ← Physics, tracking, frame utils
├── frontend/          ← React + Vite UI
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       └── components/
└── training/          ← Model training scripts
```

## 🔗 Endpoints
| URL | Description |
|-----|-------------|
| http://localhost:5173 | Frontend UI |
| http://localhost:8000/docs | API Documentation |
| POST /api/video/analyze | Auto-detect & analyze video |
| POST /api/image/analyze | Auto-detect & analyze image |

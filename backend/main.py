"""
AccidentAI Backend — FastAPI Application Entry Point
"""

import os
import shutil
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routers.video_router import router as video_router
from routers.image_router import router as image_router
from models.yolo_detector import YOLODetector

load_dotenv()

# Global model references (populated in lifespan)
yolo_detector: YOLODetector = None
_models_ready = False


def _load_classifiers_background(app):
    """
    Load ViT classifiers in a background thread so server starts instantly.
    Models become available after the download/load completes.
    """
    global _models_ready
    try:
        print("[MODEL LOADER] 🧠 Loading ViT accident classifier...")
        print("[MODEL LOADER]    (First run: ~330 MB from HuggingFace — cached forever after)")
        from models.accident_classifier import ImageAccidentClassifier
        image_clf = ImageAccidentClassifier()
        app.state.image_classifier = image_clf
        print(f"[MODEL LOADER] ✅ Image classifier ready  [{image_clf.model_status}]")

        print("[MODEL LOADER] 🎬 Initialising video classifier...")
        from models.video_classifier import VideoAccidentClassifier
        video_clf = VideoAccidentClassifier()
        # Share the already-loaded ViT — no double download
        video_clf._classifier = image_clf
        video_clf.model_status = image_clf.model_status
        app.state.video_classifier = video_clf
        print(f"[MODEL LOADER] ✅ Video classifier ready  [{video_clf.model_status}]")

        _models_ready = True
        print("[MODEL LOADER] 🎉 All models ready — full analysis available!\n")

    except Exception as e:
        print(f"[MODEL LOADER] ❌ Classifier load failed: {e}")
        print("[MODEL LOADER]    Physics-based fallback will be used for analysis.")
        app.state.image_classifier = None
        app.state.video_classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start server instantly; load heavy models in background thread."""
    global yolo_detector

    os.makedirs("temp", exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    # ── YOLO loads fast (~2 sec) — do it synchronously ──────────────────────────
    yolo_model = os.getenv("YOLO_MODEL", "yolov8s.pt")
    print(f"🚀 Loading YOLO detector ({yolo_model})...")
    yolo_detector = YOLODetector(model_name=yolo_model)
    app.state.yolo_detector = yolo_detector
    app.state.image_classifier = None   # populated by background thread
    app.state.video_classifier = None   # populated by background thread
    print("✅ YOLO loaded — server is READY.")
    print("⏳ ViT classifier loading in background (first upload may take a moment)...\n")

    # ── ViT loads slowly (~5-30 sec depending on cache) — background thread ─────
    loader = threading.Thread(
        target=_load_classifiers_background,
        args=(app,),
        daemon=True,
        name="ModelLoader",
    )
    loader.start()

    yield

    # Cleanup
    if os.path.exists("temp"):
        shutil.rmtree("temp", ignore_errors=True)
        os.makedirs("temp", exist_ok=True)
    print("🛑 AccidentAI backend shut down.")


# ── App setup ────────────────────────────────────────────────────────────────────

raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
allowed_origins = [o.strip() for o in raw_origins.split(",")]

app = FastAPI(
    title="AccidentAI",
    description="Accident Detection & Probability Analysis API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video_router, prefix="/api/video", tags=["Video Analysis"])
app.include_router(image_router, prefix="/api/image", tags=["Image Analysis"])


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check — shows loaded model status."""
    image_clf = getattr(app.state, "image_classifier", None)
    video_clf = getattr(app.state, "video_classifier", None)
    return {
        "status": "ok",
        "models_ready": _models_ready,
        "models": {
            "yolo":             app.state.yolo_detector is not None,
            "image_classifier": getattr(image_clf, "model_status", "loading..."),
            "video_classifier": getattr(video_clf, "model_status", "loading..."),
        },
        "version": "1.0.0",
    }


@app.get("/", tags=["Root"])
async def root():
    return {"message": "AccidentAI API — visit /docs for interactive documentation."}

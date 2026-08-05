"""Video analysis API router."""

import os
import uuid
import shutil
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse

from services.video_service import classify_complete_video
from services.video_prob_service import estimate_video_probability
from services.video_analyze_service import analyze_video

router = APIRouter()

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", "100"))


def _validate_video(file: UploadFile):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}",
        )


async def _save_upload(file: UploadFile, prefix: str = "video") -> str:
    """Save uploaded file to temp directory and return path."""
    os.makedirs("temp", exist_ok=True)
    ext = os.path.splitext(file.filename or "video.mp4")[1].lower()
    tmp_path = f"temp/{prefix}_{uuid.uuid4().hex}{ext}"

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_VIDEO_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_VIDEO_SIZE_MB} MB.",
        )

    with open(tmp_path, "wb") as f:
        f.write(contents)

    return tmp_path


@router.post("/classify")
async def classify_video(
    file: UploadFile = File(..., description="Complete video file for accident classification"),
):
    """
    Classify a **complete** video as Accident or No Accident.

    - Upload a full video clip (dashcam, CCTV, etc.)
    - Returns: label, confidence score, annotated preview frames
    """
    _validate_video(file)
    tmp_path = await _save_upload(file, prefix="complete")

    try:
        result = classify_complete_video(tmp_path)
        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/probability")
async def video_probability(
    file: UploadFile = File(..., description="Partial/incomplete video clip for probability estimation"),
):
    """
    Estimate accident probability from an **incomplete** video clip.

    - Upload a partial clip (e.g., dashcam segment, surveillance snippet)
    - Returns: probability %, risk level, TTC, risk-over-time heatmap
    """
    _validate_video(file)
    tmp_path = await _save_upload(file, prefix="incomplete")

    try:
        result = estimate_video_probability(tmp_path)
        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/analyze")
async def analyze_video_auto(
    file: UploadFile = File(..., description="Any video — type is auto-detected"),
):
    """
    **Unified video analysis endpoint** — auto-detects complete vs incomplete.

    - Upload any dashcam or surveillance video
    - The model automatically determines if an accident has already occurred
      or if this is an ongoing risk scenario
    - Returns: analysis_mode, label/probability, confidence/risk_level, annotated frames
    """
    _validate_video(file)
    tmp_path = await _save_upload(file, prefix="auto")

    try:
        result = analyze_video(tmp_path)
        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-analysis failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

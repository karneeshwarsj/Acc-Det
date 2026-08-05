"""Image analysis API router."""

import os
import uuid

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from services.image_service import classify_accident_image
from services.image_prob_service import estimate_proximity_probability
from services.image_analyze_service import analyze_image

router = APIRouter()

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "20"))


def _validate_image(file: UploadFile):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{ext}'. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}",
        )


async def _save_upload(file: UploadFile, prefix: str = "image") -> str:
    os.makedirs("temp", exist_ok=True)
    ext = os.path.splitext(file.filename or "image.jpg")[1].lower()
    tmp_path = f"temp/{prefix}_{uuid.uuid4().hex}{ext}"

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_IMAGE_SIZE_MB} MB.",
        )

    with open(tmp_path, "wb") as f:
        f.write(contents)

    return tmp_path


@router.post("/classify")
async def classify_image(
    file: UploadFile = File(..., description="Accident scene image (collision or deformed vehicle)"),
):
    """
    Classify a **complete accident image**.

    - Detects: vehicle collisions, deformed/damaged vehicles
    - Returns: accident label, confidence, vehicle count, annotated image
    """
    _validate_image(file)
    tmp_path = await _save_upload(file, prefix="classify")

    try:
        result = classify_accident_image(tmp_path)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/probability")
async def image_probability(
    file: UploadFile = File(..., description="Near-miss image with vehicles close but not collided"),
):
    """
    Estimate accident probability for a **near-miss image**.

    - Analyzes vehicle proximity and estimates real-world distance
    - Returns: probability table across speed ranges (0–30, 30–60 km/h, etc.)
    """
    _validate_image(file)
    tmp_path = await _save_upload(file, prefix="prob")

    try:
        result = estimate_proximity_probability(tmp_path)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Probability estimation failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/analyze")
async def analyze_image_auto(
    file: UploadFile = File(..., description="Any image — type is auto-detected"),
):
    """
    **Unified image analysis endpoint** — auto-detects accident scene vs near-miss.

    - If vehicles are overlapping or a deformed vehicle is detected → accident classification
    - If multiple vehicles are close but not colliding → proximity probability table
    - Returns: analysis_mode, result data, annotated image
    """
    _validate_image(file)
    tmp_path = await _save_upload(file, prefix="auto")

    try:
        result = analyze_image(tmp_path)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-analysis failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

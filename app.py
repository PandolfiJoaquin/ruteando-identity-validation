
from google import genai
from google.genai import types
import base64
import io
import os
from typing import Literal

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Depends, Request
from google import genai
from google.genai import types

CACHE_MODEL = "gemini-2.5-flash"
CACHE_TTL_SECONDS = 3600
SYSTEM_INSTRUCTION = (
    "You are a validator. Compare the selfie to the DNI photo and answer in JSON: "
    "{match: boolean, confidence: 0..1, notes: string[]}."
)
BOOTSTRAP_PROMPT = "Session bootstrap"

# --- helpers ---------------------------------------------------------------

def create_client() -> genai.Client:
    return genai.Client(http_options=types.HttpOptions(api_version="v1"))



# --- lifespan: create once, clean up once ---------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    client = create_client()

    # stash in app.state
    app.state.genai_client = client
    try:
        yield
    finally:
        pass

app = FastAPI(title="ID/Selfie Verification API", version="1.0", lifespan=lifespan)

class VerifyResponse(BaseModel):
    decision: Literal["ACCEPT", "REJECT", "IMAGE_UNCLEAR"]
    reason: str
    score: float | None = None  # similarity score when engine="local"

BASE_PATH = "/home/joaco/Desktop/tutoria-piñeiro/ruteando/ruteando-validator/dataset/"
PROMPT = """
    Act as a border agent.
    You will receive two images: they may be (a) an ID (Argentine DNI) and a selfie,
    (b) both sides of an ID, or (c) unrelated.
    Rules:
    1) If BOTH images are ID/DNI photos (front/back or duplicate), output: 
    Decision: REJECT | Reason: Both are DNI images (no selfie).
    2) Else, if there is a DNI and a selfie, compare faces. If same person: 
    Decision: ACCEPT | Reason: short justification. If not: 
    Decision: REJECT | Reason: short justification.
    Return ONLY:
    Decision: <ACCEPT|REJECT>
    Reason: <one sentence>
"""
# ---------- utilities ---------

# def pil_from_bucket(uri: str) -> Image.Image:
# TODO

def pil_from_local_storage(path: str) -> Image.Image:
    with open(BASE_PATH + path, "rb") as f:
        data = f.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image ({path.filename}): {e}")
    return img

def to_base64_data_url(img: Image.Image) -> str:
    """Encode PIL image -> data URL for OpenAI vision input."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def verify_with_gemini(dni_uri: str, face_uri: str) -> VerifyResponse:
    
    with open(BASE_PATH + dni_uri, "rb") as f:
        dni_data = f.read()
    with open(BASE_PATH + face_uri, "rb") as f:
        face_data = f.read()

    dni_part = types.Part.from_bytes(data=dni_data, mime_type="image/jpeg")
    face_part = types.Part.from_bytes(data=face_data, mime_type="image/jpeg")
    try:
        response = app.state.genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{
                "role": "user",
                "parts": [
                    {"text": PROMPT},
                    face_part,
                    dni_part
                ]
            }]
        )

        text_response = response.text
        for line in text_response.strip().splitlines():
            if line.upper().startswith("DECISION:"):
                val = line.split(":", 1)[1].strip().upper()
                decision = "ACCEPT" if "ACCEPT" in val else "REJECT"
            if line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        return VerifyResponse(decision=decision, reason=reason, score=None)
    except Exception as e:
        print(f"Gemini API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. issue with gemini api")



def verify_with_openai(img1: Image.Image, img2: Image.Image) -> VerifyResponse:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)


    # You can switch to a bigger model if you like (e.g., 'gpt-4o' or 'gpt-4.1')
    model_name = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

    msg = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": to_base64_data_url(img1)}},
                {"type": "image_url", "image_url": {"url": to_base64_data_url(img2)}},
            ],
        },
    ]

    try:
        resp = client.chat.completions.create(model=model_name, messages=msg, temperature=0)
        content = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

    # Simple parse (expects the two-line schema)
    decision = "REJECT"
    reason = "Unparseable response"
    for line in content.splitlines():
        if line.upper().startswith("DECISION:"):
            val = line.split(":", 1)[1].strip().upper()
            decision = "ACCEPT" if "ACCEPT" in val else "REJECT"
        if line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return VerifyResponse(decision=decision, reason=reason, score=None)

# ---------- Local engine (embeddings + cosine) ----------
def verify_with_deepface(img1: Image.Image, img2: Image.Image, threshold: float) -> VerifyResponse:
    """
    Local face embeddings + cosine similarity using DeepFace (facenet512 default).
    This engine DOES NOT detect 'both are DNI'—it just compares faces.
    Callers should ensure one is a selfie and one is an ID, or accept that
    two ID crops of the same photo will 'match'.
    """
    import numpy as np
    from deepface import DeepFace  # pip install deepface==0.0.93

    # Extract embeddings; enforce detect_faces = True to crop the dominant face.
    try:
        emb1 = DeepFace.represent(img_path=np.array(img1), model_name="Facenet512", enforce_detection=True)[0]["embedding"]
        emb2 = DeepFace.represent(img_path=np.array(img2), model_name="Facenet512", enforce_detection=True)[0]["embedding"]
    except Exception as e:
        return VerifyResponse(decision="IMAGE_UNCLEAR", reason=f"Face not found / local engine error: {e}", score=None)

    import numpy as np
    v1 = np.array(emb1, dtype="float32")
    v2 = np.array(emb2, dtype="float32")
    # cosine similarity -> [-1..1], higher is more similar
    sim = float(v1.dot(v2) / ((np.linalg.norm(v1) + 1e-8) * (np.linalg.norm(v2) + 1e-8)))

    decision = "ACCEPT" if sim >= threshold else "REJECT"
    reason = f"Cosine similarity = {sim:.3f} (threshold {threshold:.2f})"
    return VerifyResponse(decision=decision, reason=reason, score=sim)


class Request(BaseModel):
    dni_uri: str
    facepic_uri: str
    storage: Literal["bucket", "local"] = "local"
    local_threshold: float = 0.40  # only used if storage=localA

@app.post("/verify", response_model=VerifyResponse)
async def verify(req: Request):
    """
    Verify whether the selfie corresponds to the DNI.
    - engine='openai'  -> uses OpenAI vision reasoning (also rejects if both are IDs)
    - engine='local'   -> uses local embeddings (threshold-based). Does NOT detect 'two IDs' automatically.

    Returns: {decision, reason, engine, score?}
    """

    if req.storage == "bucket":
        raise HTTPException(status_code=501, detail="Bucket storage not implemented")
    elif req.storage == "local":
        return _verify(req.dni_uri, req.facepic_uri)
    else:
        raise HTTPException(status_code=400, detail="Unknown storage. storage should be one of [bucket, local]")

def _verify(dni_uri: str, facepic_uri: str) -> VerifyResponse:
    return verify_with_deepface(pil_from_local_storage(dni_uri), pil_from_local_storage(facepic_uri), 0.40)
    # return verify_with_gemini(dni_uri, facepic_uri)
    # local_result = verify_with_local(dni, facepic, threshold=float(req.local_threshold))
    # openai_result = verify_with_openai(dni, facepic)

    # if local_result and openai_result:
    #     if local_result.decision == "ACCEPT" and openai_result.decision == "ACCEPT":
    #         return VerifyResponse(decision="ACCEPT", reason="Both engines ACCEPT", score=local_result.score)
    #     elif local_result.decision == "REJECT" and openai_result.decision == "REJECT":
    #         return VerifyResponse(decision="REJECT", reason="Both engines REJECT", score=local_result.score)
    #     elif local_result.decision == "ACCEPT" and openai_result.decision == "REJECT":
    #         return VerifyResponse(decision="REJECT", reason="OpenAI REJECTs, Local ACCEPTs", score=local_result.score)
    #     else:
    #         return VerifyResponse(decision="REJECT", reason="Local REJECTs, OpenAI ACCEPTs", score=local_result.score)

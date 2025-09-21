
from google import genai
from google.genai import types
import base64
import io
import os
from typing import Literal, List, Dict

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Depends, Request
from google import genai
from google.genai import types

from storage_utils import init_firebase_admin

CACHE_MODEL = "gemini-2.5-flash"
CACHE_TTL_SECONDS = 3600
SYSTEM_INSTRUCTION = (
    "You are a validator. Compare the selfie to the DNI photo and answer in JSON: "
    "{match: boolean, confidence: 0..1, notes: string[]}."
)
BOOTSTRAP_PROMPT = "Session bootstrap"
THRESHOLD = 0.35



def create_client() -> genai.Client:
    """
    This requires GOOGLE env vars to be set:
        GEMINI_API_KEY
        GOOGLE_GENAI_USE_VERTEXAI=false
        GOOGLE_CLOUD_PROJECT
        GOOGLE_CLOUD_LOCATION
    """
    return genai.Client(http_options=types.HttpOptions(api_version="v1"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    bucket_name = "FIREBASE_BUCKET_NAME"
    init_firebase_admin(
        service_account_json="serviceAccount.json",
        bucket_name=bucket_name,
    )
    client = create_client()
    app.state.genai_client = client
    try:
        yield
    finally:
        pass

app = FastAPI(title="ID/Selfie Verification API", version="1.0", lifespan=lifespan)

class VerifyResponse(BaseModel):
    result: Literal["ACCEPT", "REJECT", "IMAGE_UNCLEAR"]
    reason: str
    scores: List[float] = []
    models: str 

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
        raise HTTPException(status_code=400, detail=f"Invalid image ({path}): {e}")
    return img

def to_base64_data_url(img: Image.Image) -> str:
    """Encode PIL image -> data URL for OpenAI vision input."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def flatmap(lst : List[List]) -> List:
    return [item for sublist in lst for item in sublist]


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

        return VerifyResponse(result=decision, reason=reason, models='gemini')
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

    return VerifyResponse(result=decision, reason=reason)

# ---------- Local engine (embeddings + cosine) ----------
def verify_with_deepface(img1: Image.Image, img2: Image.Image, threshold: float) -> List[VerifyResponse]:
    """
    Local face embeddings + cosine similarity using DeepFace (facenet512 default).
    This engine DOES NOT detect 'both are DNI'—it just compares faces.
    Callers should ensure one is a selfie and one is an ID, or accept that
    two ID crops of the same photo will 'match'.
    """
    import numpy as np
    from deepface import DeepFace  # pip install deepface==0.0.93
    responses = []
    models = [
        "VGG-Face", "Facenet", "Facenet512", 
        "ArcFace", "GhostFaceNet"
    ]
    for model in models:
        # Extract embeddings; enforce detect_faces = True to crop the dominant face.
        try:
            emb1 = DeepFace.represent(img_path=np.array(img1), model_name=model, enforce_detection=True)[0]["embedding"]
            emb2 = DeepFace.represent(img_path=np.array(img2), model_name=model, enforce_detection=True)[0]["embedding"]
        except Exception as e:
            responses.append(VerifyResponse(result="IMAGE_UNCLEAR", reason=f"Face not found / local engine error: {e}", models=model))
            continue
        import numpy as np
        v1 = np.array(emb1, dtype="float32")
        v2 = np.array(emb2, dtype="float32")
        # va entre -1 y 1
        sim = float(v1.dot(v2) / ((np.linalg.norm(v1) + 1e-8) * (np.linalg.norm(v2) + 1e-8)))

        decision = "ACCEPT" if sim >= THRESHOLD else "REJECT"
        reason = f"Cosine similarity = {sim:.3f} (threshold {THRESHOLD:.2f})"
        responses.append(VerifyResponse(result=decision, reason=reason, scores=[sim], models=model))
    return responses


class Request(BaseModel):
    dni_uri: str
    facepic_uri: str
    storage: Literal["bucket", "local"] = "local"

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

def agg_responses(responses: List[VerifyResponse]) -> VerifyResponse: 
    images_unclear = list(filter(lambda r: r.result == "IMAGE_UNCLEAR", responses))
    scores = flatmap([r.scores for r in responses])
    if len(images_unclear) > 1:
        reasons = "; ".join([f"{r.models}: {r.reason}" for r in images_unclear])
        return VerifyResponse(
            result="IMAGE_UNCLEAR",
            reason=f"Image unclear for the following models: {reasons}",
            scores=scores,
            models=",".join([r.models for r in images_unclear]),
        )
    
    # rejects = list(filter(lambda r: r.decision == "REJECT", responses))
    # if len(rejects) > 2:
    #     reasons = "; ".join([f"{r.model}: {r.reason}" for r in rejects])
    #     return VerifyResponse(
    #         decision="REJECT",
    #         reason=f"Too many rejections: {reasons}",
    #         scores=scores,
    #         model=",".join([r.model for r in rejects]),
    #     )
    
    if sum(scores)/len(scores) < THRESHOLD:
        return VerifyResponse(
            result="REJECT",
            reason=f"Low average similarity ({sum(scores)/len(scores):.3f})",
            scores=scores,
            models=",".join([r.models for r in responses]),
        )

    return VerifyResponse(
        result="ACCEPT",
        reason="Good enough average",
        scores=scores,
        models=",".join([r.models for r in responses]),
    )


def _verify(dni_uri: str, facepic_uri: str) -> VerifyResponse:
    try:
        dni_image = pil_from_local_storage(dni_uri)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="dni picture not found")

    try:
        face_image = pil_from_local_storage(facepic_uri)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="face picture not found")

    responses = verify_with_deepface(dni_image, face_image, 0.40)

    return agg_responses(responses)
    # return verify_with_gemini(dni_uri, facepic_uri)
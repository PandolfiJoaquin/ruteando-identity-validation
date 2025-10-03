
from google import genai
from google.genai import types
import base64
import io
import os
from typing import Literal, List, Dict

import numpy as np
from deepface import DeepFace

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from openai import InternalServerError
from pydantic import BaseModel
from PIL import Image

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Depends, Request
from google import genai
from google.genai import types

from storage_utils import init_firebase_admin, load_image_from_firebase_admin

CACHE_MODEL = "gemini-2.5-flash"
CACHE_TTL_SECONDS = 3600
SYSTEM_INSTRUCTION = (
    "You are a validator. Compare the selfie to the DNI photo and answer in JSON: "
    "{match: boolean, confidence: 0..1, notes: string[]}."
)
BOOTSTRAP_PROMPT = "Session bootstrap"
THRESHOLD = 0.35
DEEPFACE_MODELS = ["Facenet512"]


def preload_deepface_models():
    """Preload DeepFace models to avoid delays on first request"""
    try:
        # Create a small dummy image to trigger model loading
        dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Use configured models to preload
        models_to_preload = DEEPFACE_MODELS
        
        for model in models_to_preload:
            try:
                print(f"Loading {model} model...")
                DeepFace.represent(img_path=dummy_img, model_name=model, enforce_detection=False)
                print(f"{model} model loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to preload {model} model: {e}")
                
        try:
            print("Preloading face detection backend...")
            DeepFace.extract_faces(img_path=dummy_img, enforce_detection=False)
            print("Face detection backend loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to preload face detection backend: {e}")
                
    except Exception as e:
        print(f"Warning: Failed to preload DeepFace models: {e}")


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
    bucket_name = "test"
    init_firebase_admin(
        service_account_json="serviceAccount.json",
        bucket_name=bucket_name,
    )
    client = create_client()
    app.state.genai_client = client
    
    # Preload DeepFace models to avoid first-request delays
    print("Preloading DeepFace models...")
    try:
        preload_deepface_models()
        print("DeepFace models preloaded successfully")
    except Exception as e:
        print(f"Warning: Failed to preload models during startup: {e}")
    
    try:
        yield
    finally:
        pass

app = FastAPI(title="ID/Selfie Verification API", version="1.0", lifespan=lifespan)

############################################################
# Identity
############################################################
class VerifyResponse(BaseModel):
    result: Literal["ACCEPT", "REJECT", "IMAGE_UNCLEAR"]
    reason: str
    scores: List[float] = []
    models: str 

BASE_PATH = "/home/joaco/Desktop/tutoria-piñeiro/ruteando/ruteando-validator/dataset/"
IDENTITY_VALIDATION_PROMPT = """
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

VEHICLE_VALIDATION_PROMPT = """
    You are a vehicle registration validator.
    You will receive an image of a vehicle and a text description of the vehicle from its registration.
    Your task is to determine if the vehicle in the image matches the description provided.
    The description may include details such as make, model, color, and any distinctive features.
    You don't need to be super rigorous, but you should be able to identify clear mismatches, pictures that 
    don't show a vehicle, or images that are too unclear to make a judgment (less than 30% of the vehicle visible, extreme blur, etc).

    Rules:
    1) If the vehicle in the image clearly matches the description, output:
    Decision: ACCEPT | Reason: Matches.
    2) If the vehicle in the image does not match the description, output:
    Decision: REJECT | Reason: short justification.
    3) If the image is unclear or does not show a vehicle, output:
    Decision: IMAGE_UNCLEAR | Reason: short justification.
    Return ONLY:
    Decision: <ACCEPT|REJECT|IMAGE_UNCLEAR>
    Reason: <one sentence>"""

# ---------- utilities ---------

def pil_from_local_storage(path: str) -> Image.Image:
    with open(BASE_PATH + path, "rb") as f:
        data = f.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image ({path}): {e}")
    return img

def to_base64_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def flatmap(lst : List[List]) -> List:
    return [item for sublist in lst for item in sublist]

def verify_identity_with_gemini(dni_uri: str, face_uri: str) -> VerifyResponse:
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
                    {"text": IDENTITY_VALIDATION_PROMPT},
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


# ---------- Local engine (embeddings + cosine) ----------
def verify_with_deepface(dni_img: Image.Image, face_img: Image.Image) -> List[VerifyResponse]:
    """
    Local face embeddings + cosine similarity using DeepFace (facenet512 default).
    This engine DOES NOT detect 'both are DNI'—it just compares faces.
    Callers should ensure one is a selfie and one is an ID, or accept that
    two ID crops of the same photo will 'match'.
    """
    
    responses = []
    models = DEEPFACE_MODELS
    for model in models:
        # Extract embeddings; enforce detect_faces = True to crop the dominant face.
        try:
            emb1 = DeepFace.represent(img_path=np.array(dni_img), model_name=model, enforce_detection=True)[0]["embedding"]
            emb2 = DeepFace.represent(img_path=np.array(face_img), model_name=model, enforce_detection=True, anti_spoofing=False)[0]["embedding"]
        except Exception as e:
            responses.append(VerifyResponse(result="IMAGE_UNCLEAR", reason=f"Face not found / spoofing detected: {e}", models=model))
            continue
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
    bucket: str

@app.post("/verify", response_model=VerifyResponse)
async def verify(req: Request):
    """
    Verify whether the selfie corresponds to the DNI.
    - engine='openai'  -> uses OpenAI vision reasoning (also rejects if both are IDs)
    - engine='local'   -> uses local embeddings (threshold-based). Does NOT detect 'two IDs' automatically.

    Returns: {decision, reason, engine, score?}
    """

    result = _verify(req.dni_uri, req.facepic_uri, req.bucket)
    print(result)
    return result

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


def _verify(dni_uri: str, facepic_uri: str, bucket: str) -> VerifyResponse:
    if bucket == "local":
        try:
            dni_image = pil_from_local_storage(dni_uri)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail="dni picture not found")

        try:
            face_image = pil_from_local_storage(facepic_uri)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail="face picture not found")
    else:
        try:
            dni_image = load_image_from_firebase_admin(dni_uri, bucket)
            face_image = load_image_from_firebase_admin(facepic_uri, bucket)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Error loading images: {e}")

    responses = verify_with_deepface(dni_image, face_image)

    return agg_responses(responses)
############################################################
# VEHICLE
############################################################

class VehicleRequest(BaseModel):
    img_uri: str
    description: str
    bucket: str

class VehicleResponse(BaseModel):
    result: Literal["ACCEPT", "REJECT", "IMAGE_UNCLEAR"]
    reason: str



def verify_vehicle_with_gemini(image: Image, description: str) -> VehicleRequest:
    try:
        print("about to save to JPEG")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        print("saved to JPEG")
        print("About to open as PIL")
        image = Image.open(buffer)
        print("Opened as PIL")
    except Exception as e1:
        print("Error trying to load as JPEG, trying PNG:", e1)
        try:
            print("about to save to PNG")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            print("saved to PNG")
            print("About to open as PIL")
            image = Image.open(buffer)
            print("Opened as PIL")
        except Exception as e2:
            print("Error trying to load as PNG:", e2)
            raise HTTPException(status_code=400, detail=f"Error processing image: {e1}; {e2}")
        
        

    vehicle_part = types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")
    print("bytes part")
    try:
        response = app.state.genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{
                "role": "user",
                "parts": [
                    {"text": VEHICLE_VALIDATION_PROMPT + "\nVehicle description: " + description},
                    vehicle_part
                ]
            }]
        )

        text_response = response.text
        for line in text_response.strip().splitlines():
            if line.upper().startswith("DECISION:"):
                val = line.split(":", 1)[1].strip().upper()
                if val in ["ACCEPT", "REJECT", "IMAGE_UNCLEAR"]:
                    decision = val
                else:
                    raise InternalServerError("Invalid decision")
            if line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        return VehicleResponse(result=decision, reason=reason)
    except Exception as e:
        print(f"Gemini API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. issue with gemini api")
@app.post("/verify/vehicle", response_model=VehicleResponse)
async def verify_vehicle(req: VehicleRequest) -> VehicleResponse:
    """
    Verify whether the vehicle in the selfie corresponds to the vehicle in the DNI.
    Uses OpenAI vision reasoning.

    Returns: {decision, reason, engine, score?}
    """
    try:
        vehicle_image = load_image_from_firebase_admin(req.img_uri, req.bucket)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error loading images: {e}")

    return verify_vehicle_with_gemini(vehicle_image, req.description)
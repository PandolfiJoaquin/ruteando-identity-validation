# pip install firebase-admin pillow
from io import BytesIO
from typing import Optional
from PIL import Image

import firebase_admin
from firebase_admin import credentials, storage


def init_firebase_admin(
    *, 
    service_account_json: Optional[str] = None, 
    bucket_name: Optional[str] = None
):
    """
    Initialize firebase_admin once per process. 
    """
    if not service_account_json:
        raise ValueError("service_account_json must be provided.")

    cred = credentials.Certificate(service_account_json)
    firebase_admin.initialize_app(cred, {"storageBucket": bucket_name} if bucket_name else None)


def _parse_gs_uri(gs_uri: str) -> tuple[Optional[str], str]:
    """
    Accepts either:
      - 'gs://my-bucket/path/to/file.png'
      - 'path/to/file.png'  (uses default bucket from init)
    Returns (bucket_name_or_None, blob_path)
    """
    if gs_uri.startswith("gs://"):
        # gs://bucket/path...
        without_scheme = gs_uri[5:]
        bucket, _, blob_path = without_scheme.partition("/")
        if not bucket or not blob_path:
            raise ValueError(f"Invalid gs URI: {gs_uri}")
        return bucket, blob_path
    return None, gs_uri


def load_image_from_firebase_admin(gs_uri: str, bucket_name: Optional[str] = None) -> Image.Image:
    """
    Downloads an image from Firebase Storage using firebase-admin and returns it as a PIL.Image.
    
    Args:
        gs_uri: Either a full 'gs://bucket/path/to/file.ext' or just 'path/to/file.ext'
                (the latter uses the default bucket set at init).
    """
    bucket_from_uri, blob_path = _parse_gs_uri(gs_uri)

    if bucket_from_uri:
        bucket = storage.bucket(bucket_from_uri)
        print(f"Using bucket from URI: {bucket_from_uri}")
    elif bucket_name:
        bucket = storage.bucket(bucket_name)
        print(f"Using provided bucket name: {bucket_name}")
    else:
        bucket = storage.bucket()
        print("Using default bucket from firebase_admin initialization")

    blob = bucket.blob(blob_path)
    data = blob.download_as_bytes()
    img = Image.open(BytesIO(data))
    img.load()
    return img


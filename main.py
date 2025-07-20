import traceback

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from keras.api.applications.mobilenet_v2 import preprocess_input
from keras.api.preprocessing import image
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import time
import os
import uuid
from rembg import remove
from starlette.responses import StreamingResponse

app = FastAPI()

# /ml 접두사를 가진 라우터 생성
ml_router = APIRouter(
    prefix="/ml",
    tags=["ml"],
)

@app.get("/health")
def health_check():
    return {"message": "thatzfit-image-worker is running"}


# 모델 정보
MODEL_PATH = "./models/mobilenet_v2_fashion_classifier.h5"
IMAGE_SIZE = (224, 224)

# 모델 로딩
print("Loading model")
start_time = time.time()
model = None
try:
    # 모델 파일 존재 확인
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    # 모델 로딩 시도
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Model loaded successfully")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")
    print(f"Full error details: {traceback.format_exc()}")

load_time = time.time() - start_time
print(f"Model loading took {load_time:.2f} seconds")

# 모델이 로드되지 않은 경우 경고
if model is None:
    print("WARNING: Model failed to load. The /predict endpoint will return errors.")

class_dictionary = {
    0: 'ACCESSORY',
    1: 'LONG_PANTS',
    2: 'LONG_SLEEVE',
    3: 'OUTWEAR',
    4: 'SHOES',
    5: 'SHORT_PANTS',
    6: 'SHORT_SLEEVE',
    7: 'SLEEVELESS'
}


# 전처리 함수
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
        return img_array
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        raise e


# 예측
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 모델 상태 확인
    if model is None:
        return JSONResponse(
            content={
                "error": "Model is not available", 
                "details": "The model failed to load during startup. Please check the server logs."
            }, 
            status_code=503
        )

    try:
        print(f"[INFO] Received file: {file.filename}")
        
        # 파일 타입 검증
        if not file.content_type or not file.content_type.startswith('image/'):
            return JSONResponse(
                content={"error": "Invalid file type. Please upload an image file."}, 
                status_code=400
            )
        
        # 이미지 전처리
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            return JSONResponse(
                content={"error": "Empty file received"}, 
                status_code=400
            )
            
        processed_image = preprocess_image(image_bytes)

        # 예측 수행
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = int(np.argmax(predictions[0]))
        predicted_class = class_dictionary[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        print(f"[INFO] Predicted class: {predicted_class}, confidence: {confidence:.2f}")

        return JSONResponse(content={
            "class_idx": predicted_class_idx,
            "class_name": predicted_class,
            "confidence": confidence
        }, status_code=200)

    except Exception as e:
        # 예외가 발생한 경우 상세한 오류 정보 반환
        error_details = traceback.format_exc()
        print(f"[ERROR] Prediction failed: {error_details}")
        return JSONResponse(
            content={
                "error": f"Prediction failed: {str(e)}", 
                "details": error_details
            }, 
            status_code=500
        )




@app.post("/remove-background")
async def remove_background(file: bytes = File(...)):
    try:
        output_image = remove(file)
        return StreamingResponse(io.BytesIO(output_image), media_type="image/png")  # type: ignore

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error removing background: {error_details}")
        return JSONResponse(content={"error": str(e), "details": error_details}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

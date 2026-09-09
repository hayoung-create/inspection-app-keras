import os
import numpy as np
from PIL import Image
import openvino as ov
from tensorflow import keras

# ── 설정 ─────────────────────────────────────────────────────────
MODEL_XML_PATH  = "./weights/leather_model.xml"
TEST_IMAGE_PATH = "sample.png"
INPUT_IMG_SIZE  = (224, 224)
CLASSES         = ["정상", "불량"]


# 1. OpenVINO 모델 로드 및 컴파일
def load_ov_model():
    if not os.path.exists(MODEL_XML_PATH):
        raise FileNotFoundError(f"OpenVINO 모델 파일이 없습니다: {MODEL_XML_PATH}")
    
    core = ov.Core()
    # xml 모델 파일 읽기
    model = core.read_model(MODEL_XML_PATH)
    # CPU 디바이스용으로 컴파일
    compiled_model = core.compile_model(model, "CPU")
    print(f"[1] OpenVINO 모델 로드 완료 → {MODEL_XML_PATH}")
    return compiled_model


# 2. 이미지 전처리 (기존 VGG16 전처리 유지)
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = keras.applications.vgg16.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


# 3. OpenVINO 추론
def predict(compiled_model, pil_img):
    arr = preprocess(pil_img)
    # OpenVINO 추론 실행 (출력 레이어 결과 가져오기)
    output = compiled_model(arr)[compiled_model.output(0)]
    prob = float(output[0][0])
    label = CLASSES[1 if prob > 0.5 else 0]
    return label, prob


# 메인 실행
def main():
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"테스트 이미지가 없습니다: {TEST_IMAGE_PATH}")
        return

    # 모델 로드
    compiled_model = load_ov_model()
    
    # 이미지 준비
    pil_img = Image.open(TEST_IMAGE_PATH)
    
    # 추론 실행
    print("[2] OpenVINO 추론 중...")
    label, prob = predict(compiled_model, pil_img)
    
    # 결과 출력
    print("─" * 40)
    print(f"  예측 결과 : {label}")
    print(f"  불량 확률 : {prob:.1%}")
    print(f"  정상 확률 : {(1 - prob):.1%}")
    print("─" * 40)


if __name__ == "__main__":
    main()
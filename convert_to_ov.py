import os
import numpy as np
import openvino as ov
import tensorflow as tf

MODEL_PATH = "./weights/leather_model.keras"
SAVE_DIR = "./weights"
OV_MODEL_PATH = os.path.join(SAVE_DIR, "leather_model.xml")

# 1. Keras 모델 로드
model = tf.keras.models.load_model(MODEL_PATH)

# 2. OpenVINO 변환을 위한 가짜 입력 데이터 생성 (배치1, 높이224, 너비224, 채널3)
example_input = np.zeros((1, 224, 224, 3), dtype=np.float32)

# 3. OpenVINO 모델로 변환 (example_input 전달)
ov_model = ov.convert_model(model, example_input=example_input)

# 4. OpenVINO IR (.xml, .bin) 형식으로 저장
ov.save_model(ov_model, OV_MODEL_PATH)
print(f"OpenVINO 모델 변환 성공! -> {OV_MODEL_PATH}")
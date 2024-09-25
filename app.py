import streamlit as st
from tensorflow.keras.models import model_from_json
from pathlib import Path
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array # Import img_to_array


# กำหนดชื่อไฟล์ JSON และ H5
model_structure_file = 'model_structure.json()'
model_weights_file = 'model.weights.h5'

# โหลดโมเดลจากไฟล์ JSON และ H5
with open(model_structure_file, 'r') as f:
    model_structure = f.read()
model = model_from_json(model_structure)
model.load_weights(model_weights_file)

# โหลดโมเดล EfficientNetB0 สำหรับการสกัดคุณสมบัติ
feature_extraction_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def process_image(img):
    img = img.resize((224, 224))  # ปรับขนาดรูปภาพ
    img_array = img_to_array(img) # Use img_to_array to convert PIL Image to NumPy array
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # ใช้ preprocess_input จาก EfficientNetB0

    # สกัดคุณสมบัติโดยใช้ EfficientNetB0
    features = feature_extraction_model.predict(img_array)

    # ทำนายผลโดยใช้โมเดลของคุณ
    results = model.predict(features)
    predicted_class = np.argmax(results)

    # แปลผลการทำนายเป็นชื่อคลาส
    class_names = {
        0: 'This is Ant bite คุณโดนมดกัด',
        1: 'This is Bedbug bite คุณโดนตัวเรือดกัด',
        2: 'This is Chigger bite คุณโดนเห็บลมกัด',
        3: 'This is fleas bite คุณโดนหมัดกัด',
        4: 'This is mosquito bite คุณโดนยุงกัด',
        5: 'This is no bite คุณไม่ได้โดนแมลงกัด',
        6: 'This is spider bite คุณโดนแมงมุมกัด',
        7: 'This is tick bite คุณโดนเห็บกัด'
    }
    predicted_name = class_names.get(predicted_class, 'ไม่สามารถระบุได้')

    return predicted_name, results[0][predicted_class]

# ส่วนของ Streamlit UI
st.title('แอปพลิเคชันตรวจสอบรอยกัดจากแมลง')

uploaded_file = st.file_uploader("เลือกรูปภาพ", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพที่อัปโหลด', use_column_width=True)

    if st.button('ประมวลผล'):
        result, confidence = process_image(image)
        st.write(f"ผลการทำนาย: {result}")
        st.write(f"ความมั่นใจ: {confidence*100:.2f}%")
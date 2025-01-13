from tensorflow.keras.models import model_from_json
import streamlit as st
from tensorflow.keras.models import model_from_json
from pathlib import Path
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# เพิ่มรูปภาพหัวข้อด้านบนของแอป
st.image("header.jpg", width=700, use_column_width=False)

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

def process_image(img, symptoms):
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # สกัดคุณสมบัติโดยใช้ EfficientNetB0
    features = feature_extraction_model.predict(img_array)

    # ทำนายผลโดยใช้โมเดลของคุณ
    results = model.predict(features)
    predicted_class = np.argmax(results)

    # แปลผลการทำนายเป็นชื่อคลาส
    class_names = {
        0: 'โดนมดกัด',
        1: 'โดนตัวเรือดกัด',
        2: 'โดนเห็บลมกัด',
        3: 'โดนหมัดกัด',
        4: 'โดนยุงกัด',
        5: 'ไม่ได้โดนแมลงกัด',
        6: 'โดนแมงมุมกัด',
        7: 'โดนเห็บกัด'
    }

    advice = {
        0: 'ล้างบริเวณที่ถูกกัดด้วยสบู่และน้ำสะอาด ใช้ยาหม่องหรือคาลาไมน์เพื่อลดอาการคัน.',
        1: 'รักษาความสะอาด ใช้คาลาไมน์หรือครีมเพื่อลดอาการคัน.',
        2: 'อาบน้ำและเปลี่ยนเสื้อผ้าที่สะอาด ใช้คาลาไมน์เพื่อลดการระคายเคือง.',
        3: 'ล้างด้วยสบู่และน้ำสะอาด ใช้ยาฆ่าเชื้อหรือคาลาไมน์เพื่อลดอาการ.',
        4: 'ใช้คาลาไมน์หรือครีมลดอาการคัน อาจใช้ยาทากันยุงเพื่อป้องกันในอนาคต.',
        5: 'ไม่จำเป็นต้องทำการรักษา.',
        6: 'ล้างแผลด้วยสบู่และน้ำสะอาด หลีกเลี่ยงการเกาและพบแพทย์หากมีอาการบวม.',
        7: 'ล้างบริเวณที่ถูกกัดทันที ใช้ยาฆ่าเชื้อและปรึกษาแพทย์หากจำเป็น.'
    }

    symptoms_info = "\n".join([f"- {symptom}" for symptom in symptoms])

    predicted_name = class_names.get(predicted_class, 'ไม่สามารถระบุได้')
    treatment = advice.get(predicted_class, '')

    return predicted_name, results[0][predicted_class], treatment, symptoms_info

# เพิ่ม CSS สำหรับปรับสีพื้นหลังและข้อความ
st.markdown(
    """
    <style>
    .main {
        background-color: #63a66a;
    }
    .stTextInput, .stFileUploader {
        background-color: #f9e8c6;
    }
    .stMarkdown {
        background-color: #f9e8c6;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ใช้ HTML และ CSS เพื่อกำหนดสีน้ำเงินให้กับข้อความ
st.markdown(
    """
    <style>
    .blue-text {
        color: #3e4a61; /* สีน้ำเงิน */
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ใช้ HTML และ CSS ในข้อความของ st.file_uploader
st.markdown("<p class='blue-text'>กรุณาเลือกภาพแผลที่ถูกแมลงกัด</p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    # เพิ่มส่วนเลือกอาการ
    st.markdown("<p class='blue-text'>**โปรดเลือกอาการที่คุณมี (ถ้ามี):**</p>", unsafe_allow_html=True)
    symptoms = []
    if st.checkbox("คัน"):
        symptoms.append("คัน")
    if st.checkbox("แสบร้อน"):
        symptoms.append("แสบร้อน")
    if st.checkbox("อ่อนเพลีย"):
        symptoms.append("อ่อนเพลีย")
    if st.checkbox("บวม"):
        symptoms.append("บวม")

    if st.button('วิเคราะห์'):
        result, confidence, advice, symptoms_info = process_image(image, symptoms)

      # ผลการวิเคราะห์ (ข้อความสีน้ำเงินเข้ม)
        st.markdown(
            f"""
            <p style='font-size:18px; padding: 15px; border: 2px solid #3e4a61; background-color: #f9e8c6; border-radius: 10px; color: #000080;'>
                <strong>ผลการวิเคราะห์:</strong> {result} &nbsp;&nbsp;&nbsp;&nbsp;
                <strong>ความมั่นใจ:</strong> {confidence*100:.2f}%
            </p>
            """, unsafe_allow_html=True
        )

        # รายงานอาการที่เลือก
        if symptoms_info:
            st.markdown(
                f"""
                <div style="border: 2px solid #3e4a61; background-color: #f9e8c6; border-radius: 10px; padding: 20px; margin-top: 20px; font-size: 18px; color: #000080;">
                    <strong>อาการที่รายงาน:</strong><br>{symptoms_info}
                </div>
                """, unsafe_allow_html=True
            )

        # คำแนะนำ (ข้อความสีน้ำเงินเข้ม)
        if advice:
            st.markdown(
                f"""
                <div style="border: 2px solid #3e4a61; background-color: #f9e8c6; border-radius: 10px; padding: 20px; margin-top: 20px; font-size: 20px; font-weight: bold; color: #000080; text-align: center;">
                    คำแนะนำเบื้องต้น: {advice}
                </div>
                """, unsafe_allow_html=True
            )

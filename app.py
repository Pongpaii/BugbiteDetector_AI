from tensorflow.keras.models import model_from_json
import streamlit as st
from tensorflow.keras.models import model_from_json
from pathlib import Path
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array # Import img_to_array

# Add the header image at the top of the app
st.image("header.jpeg", width=700,  use_column_width=False)

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
    img_array = img_to_array(img)  # แปลงรูปภาพจาก PIL Image เป็น NumPy array
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # ใช้ preprocess_input จาก EfficientNetB0

    # สกัดคุณสมบัติโดยใช้ EfficientNetB0
    features = feature_extraction_model.predict(img_array)

    # ทำนายผลโดยใช้โมเดลของคุณ
    results = model.predict(features)
    predicted_class = np.argmax(results)

    # แปลผลการทำนายเป็นชื่อคลาส
    class_names = {
        0: '1This is Ant bite คุณโดนมดกัด',
        1: '2This is Bedbug bite คุณโดนตัวเรือดกัด',
        2: '3This is Chigger bite คุณโดนเห็บลมกัด',
        3: '4This is fleas bite คุณโดนหมัดกัด',
        4: '5This is mosquito bite คุณโดนยุงกัด',
        5: '6This is no bite คุณไม่ได้โดนแมลงกัด',
        6: '7This is spider bite คุณโดนแมงมุมกัด',
        7: '8This is tick bite คุณโดนเห็บกัด'
    }
    
    # คำแนะนำการรักษาเบื้องต้น
    treatment_advice = {
        0: 'ล้างบริเวณที่ถูกกัดด้วยสบู่และน้ำสะอาด ใช้ยาหม่องหรือคาลาไมน์เพื่อลดอาการคัน',
        1: 'รักษาความสะอาด ใช้คาลาไมน์หรือครีมเพื่อบรรเทาอาการคัน และหลีกเลี่ยงการเกาบริเวณที่ถูกกัด',
        2: 'อาบน้ำทันทีและเปลี่ยนเสื้อผ้าที่สะอาด ใช้ครีมหรือคาลาไมน์เพื่อลดการระคายเคือง',
        3: 'ล้างด้วยสบู่และน้ำสะอาด ใช้ยาฆ่าเชื้อหรือคาลาไมน์เพื่อบรรเทาอาการ',
        4: 'ใช้คาลาไมน์หรือครีมลดอาการคัน อาจใช้ยาทากันยุงเพื่อป้องกันยุงกัดในอนาคต',
        5: '',  # No treatment for no bite
        6: 'ล้างแผลด้วยน้ำสะอาดและสบู่ หลีกเลี่ยงการเกาและไปพบแพทย์หากมีอาการบวมแดงหรือปวดมากขึ้น',
        7: 'ล้างบริเวณที่ถูกกัดทันที ใช้ยาฆ่าเชื้อและไปพบแพทย์เพื่อรับคำแนะนำเพิ่มเติมหากจำเป็น'
    }

    predicted_name = class_names.get(predicted_class, 'Unable to specify')
    advice = treatment_advice.get(predicted_class, '')

    return predicted_name, results[0][predicted_class], advice

# เพิ่ม CSS สำหรับปรับสีพื้นหลังและกล่องข้อความ
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


uploaded_file = st.file_uploader("Select a photo of your insect bite wound.", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    if st.button('Processing'):
        # เรียกใช้ฟังก์ชัน process_image และรับค่า 3 ตัวคือ ผลการทำนาย, ความมั่นใจ, และคำแนะนำการรักษา
        result, confidence, advice = process_image(image)

            # Replace st.write() for result and confidence with st.markdown() using inline HTML
        st.markdown(
            f"""
            <p style='font-size:18px; padding: 15px; border: 2px solid #3e4a61; background-color: #f9e8c6; border-radius: 10px;'>
                <strong>Prediction results:</strong> {result} &nbsp;&nbsp;&nbsp;&nbsp;
                <strong>Confidence:</strong> {confidence*100:.2f}%
            </p>
            """, unsafe_allow_html=True
        )

        
            # Display the basic treatment advice in a styled box
        if advice:
            st.markdown(
                
                f"""
                <div style="
                    border: 2px solid #3e4a61; 
                    background-color: #f9e8c6;
                    border-radius: 10px;
                    padding: 20px;
                    margin-top: 20px;
                    font-size: 20px;
                    font-weight: bold;
                    color: #3e4a61;
                    text-align: center;">
                    Basic treatment advice: {advice}
                </div>
                """, 
                unsafe_allow_html=True
            )

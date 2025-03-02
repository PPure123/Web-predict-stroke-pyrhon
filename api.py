from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

# โหลดโมเดลและ scaler
model_path = "D:/Project/jammyhomework/gradient_boosting_model.pkl"
scaler_path = "D:/Project/jammyhomework/scaler.pkl"
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# รายการคอลัมน์ที่โมเดลคาดหวัง
expected_columns = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'avg_glucose_level', 'bmi'
]

# สร้างแอป Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # หน้าเว็บที่ให้กรอกข้อมูล

@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูลจากฟอร์ม
    try:
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['married'])  # ปรับจาก 'ever_married' เป็น 'married' ให้ตรงกับ HTML
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])

        # สร้าง dictionary สำหรับข้อมูลดิบ
        input_dict = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi
        }

        # แปลงข้อมูลเป็น DataFrame
        input_df = pd.DataFrame([input_dict])

        # ปรับให้คอลัมน์ตรงกับที่โมเดลคาดหวัง
        input_df = input_df[expected_columns]

        # ปรับขนาดข้อมูลด้วย scaler
        input_scaled = scaler.transform(input_df)

        # ทำนายผล
        prediction = model.predict(input_scaled)[0]

        # ส่งผลการทำนาย
        if prediction == 1:
            result = 'มีความเสี่ยงเป็นโรคหลอดเลือดสมอง'
        else:
            result = 'ไม่มีความเสี่ยงเป็นโรคหลอดเลือดสมอง'

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"เกิดข้อผิดพลาด: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# สร้างแอป Flask
app = Flask(__name__)
CORS(app) 

# โหลดโมเดลและ Scaler ที่บันทึกไว้
model = joblib.load('models/fifa_player_rating_model_projectV2.pkl')
scaler = joblib.load('models/scalerV2.pkl')

# สร้าง API สำหรับทำนายคะแนนรวมของผู้เล่น FIFA
@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูลฟีเจอร์จากผู้ใช้
    data = request.get_json()

    # ฟีเจอร์ที่โมเดลต้องการ
    features = ['crossing', 'finishing', 'short_passing', 'dribbling', 
                'ball_control', 'acceleration', 'sprint_speed', 'agility', 
                'shot_power', 'stamina', 'vision', 'reactions', 'composure',
                'strength', 'interceptions', 'standing_tackle', 'sliding_tackle', 
                'heading_accuracy', 'marking', 'aggression'
                ]
    
    # แปลงข้อมูลให้อยู่ในรูปแบบที่โมเดลต้องการ
    input_data = np.array([data[feature] for feature in features]).reshape(1, -1)

    # ปรับสเกลข้อมูลให้เข้ากับโมเดลที่ใช้ StandardScaler
    input_data_scaled = scaler.transform(input_data)

    # ทำนายผลลัพธ์
    prediction = model.predict(input_data_scaled)[0]

    # ส่งผลลัพธ์กลับไปในรูปแบบ JSON
    return jsonify({'overall_rating': prediction})

if __name__ == '__main__':
    app.run(debug=True)

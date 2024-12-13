from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime

app = Flask(__name__)

# Load model dan scaler
try:
    model = tf.keras.models.load_model('model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    scaler = joblib.load('scaler_x.pkl')  # Load scaler untuk fitur input
    y_scaler = joblib.load('scaler_y.pkl')  # Load scaler untuk output
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None  # Set to None jika loading gagal

@app.route('/predict', methods=['POST'])
def predict():
    # Mengecek apakah model berhasil dimuat
    if model is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    # Parse data request dari client (format JSON)
    data = request.get_json()

    # Memeriksa apakah field 'tasks' ada dalam request
    if 'tasks' not in data:
        return jsonify({'error': 'Missing required "tasks" field'}), 400

    try:
        tasks = data['tasks']  # Ambil daftar tugas
        features_list = []
        task_ids = []

        # Iterasi melalui setiap tugas dan proses fitur input
        for task in tasks:
            # Ambil task_id jika ada
            task_id = task.get('task_id', None)
            task_ids.append(task_id)

            # Memetakan 'difficulty_level'
            difficulty_mapping = {
                'Easy': 1,
                'Medium': 2,
                'Hard': 3
            }
            difficulty_level = difficulty_mapping.get(task.get('difficulty_level', ''), 0)  # Default 0

            # Memetakan 'duration'
            duration_mapping = {
                "Kurang dari 1 Jam": 1,
                "1-3 Jam": 3,
                "4-6 Jam": 5,
                "Lebih dari 6 Jam": 7
            }
            duration = duration_mapping.get(task.get('duration', ''), 0)  # Default 0

            # Menghitung days_until_deadline
            deadline = datetime.strptime(task.get('deadline', ''), '%Y-%m-%d %H:%M:%S')
            current_time = datetime.now()
            days_until_deadline = (deadline - current_time).days
            hour_of_day = deadline.hour  # Mendapatkan jam dari deadline
            day_of_week = deadline.weekday()  # Mendapatkan hari dalam minggu

            # Menambahkan fitur ke list
            features_list.append([days_until_deadline, difficulty_level, duration, hour_of_day, day_of_week])

        # Konversi fitur menjadi array NumPy
        features = np.array(features_list)

        # Preprocessing: Normalisasi fitur input menggunakan scaler
        scaled_features = scaler.transform(features)

        # Melakukan prediksi batch dengan model
        predictions_scaled = model.predict(scaled_features)

        # Mengembalikan hasil prediksi ke skala aslinya menggunakan y_scaler
        predictions = y_scaler.inverse_transform(predictions_scaled).flatten()

        # Menyiapkan response dalam format JSON dengan task_id
        response = [
            {'task_id': task_id, 'priority_level': float(pred)}
            for task_id, pred in zip(task_ids, predictions)
        ]
        return jsonify(response)

    except Exception as e:
        # Menangani error yang terjadi selama prediksi
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

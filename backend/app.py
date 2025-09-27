# backend/app.py
import os
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # kurangi log TF

app = Flask(__name__, template_folder="../templates", static_folder="../static")
MODEL = None

# URUTAN KELAS harus cocok dengan yang model latih — repo menyebut kelas:
CLASS_NAMES = ['trash','plastic','metal','paper','cardboard']  # pastikan urutan sesuai model. :contentReference[oaicite:4]{index=4}

def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    candidates = [
        os.path.join('models','MobileNetV2'),
        os.path.join('models','model.h5'),
        os.path.join('..','trash_repo','saved_models','MobileNetV2')
    ]
    for p in candidates:
        if os.path.exists(p):
            MODEL = tf.keras.models.load_model(p)
            print("Loaded model from", p)
            return MODEL
    raise RuntimeError("Model tidak ditemukan. Letakkan model di backend/models/MobileNetV2 atau sesuaikan path.")

def prepare_image(pil_image, target_size=(224,224)):
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize(target_size)
    arr = np.array(pil_image).astype('float32')
    arr = preprocess_input(arr)  # MobileNetV2 preprocessing
    arr = np.expand_dims(arr, 0)
    return arr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error":"File 'image' tidak ditemukan"}), 400
        f = request.files['image']
        img = Image.open(f.stream)
        x = prepare_image(img)
        model = load_model()
        preds = model.predict(x)

        # aman untuk berbagai bentuk output
        probs = tf.nn.softmax(preds[0]).numpy() if preds.ndim == 2 else tf.nn.softmax(preds).numpy()
        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        confidence = float(probs[idx])

        # mapping ke kategori buangan + fakta singkat (ubah sesuai preferensi lokal)
        mapping = {
            'plastic': {
                'category': 'Anorganik — Plastik (daur ulang bila bisa, mis. PET)',
                'fact': 'Botol PET bisa butuh ratusan tahun untuk terurai, namun bisa didaur ulang menjadi benang poliester.'
            },
            'metal': {
                'category': 'Anorganik — Logam (daur ulang)',
                'fact': 'Logam seperti aluminium sangat bernilai untuk didaur ulang dan bisa diproses berulang kali.'
            },
            'paper': {
                'category': 'Kertas — Daur ulang (pisahkan dari sampah basah)',
                'fact': 'Kertas mudah didaur ulang; hindari kertas yang basah/minyak.'
            },
            'cardboard': {
                'category': 'Kardus/Karton — Daur ulang',
                'fact': 'Kardus dapat didaur ulang menjadi karton baru setelah dipisahkan dari kontaminan.'
            },
            'trash': {
                'category': 'Sampah residu / Anorganik (tidak dapat didaur ulang)',
                'fact': 'Beberapa sampah bercampur/terkontaminasi tidak bisa didaur ulang—masuk ke residu.'
            }
        }
        info = mapping.get(label, {'category':'Tidak diketahui','fact':''})

        return jsonify({
            'label': label,
            'category': info['category'],
            'fact': info['fact'],
            'confidence': round(confidence, 3)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # cara simpel menjalankan: python backend/app.py
    app.run(host='127.0.0.1', port=5000, debug=True)
                                                                             
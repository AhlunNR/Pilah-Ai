# backend/predict_cli.py
import sys, os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

CLASS_NAMES = ['trash','plastic','metal','paper','cardboard']

model = tf.keras.models.load_model('models/MobileNetV2')  # sesuaikan path

def predict_path(p):
    img = Image.open(p).convert('RGB').resize((224,224))
    x = preprocess_input(np.array(img).astype('float32'))
    x = np.expand_dims(x,0)
    preds = model.predict(x)
    probs = tf.nn.softmax(preds[0]).numpy()
    idx = int(np.argmax(probs))
    print(p, CLASS_NAMES[idx], probs[idx])

if __name__ == '__main__':
    for path in sys.argv[1:]:
        predict_path(path)

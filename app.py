import io
import os
import gdown
import numpy as np
from ai_edge_litert.interpreter import Interpreter

from flask import Flask, request, render_template, url_for
from PIL import Image, ImageChops, ImageEnhance
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ── Load Model TFLite (auto-download dari Drive kalau belum ada) ───────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'model_cnn_ela_qris_quant.tflite')
FILE_ID    = "1Tne2au-bjRO8Z9oxV-N-quZlKuJW8JTV"

if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}&export=download&confirm=t", MODEL_PATH, quiet=False, fuzzy=True)

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = 128

# ── ELA ───────────────────────────────────────────────────────────────
def convert_to_ela_image(path, quality=90):
    im  = Image.open(path).convert('RGB')
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    resaved_im = Image.open(buf)

    ela_im   = ImageChops.difference(im, resaved_im)
    extrema  = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    ela_im = ImageEnhance.Brightness(ela_im).enhance(255.0 / max_diff)
    return ela_im

# ── Routes ────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='Tidak ada file yang diupload!')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='Pilih file terlebih dahulu!')

    allowed = {'jpg', 'jpeg', 'png'}
    if not ('.' in file.filename and
            file.filename.rsplit('.', 1)[1].lower() in allowed):
        return render_template('index.html', error='Format file harus JPG atau PNG!')

    filename    = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    # ELA preprocessing
    ela_img   = convert_to_ela_image(upload_path)
    ela_img   = ela_img.resize((IMG_SIZE, IMG_SIZE))
    ela_arr   = np.array(ela_img, dtype=np.float32) / 255.0
    img_array = ela_arr.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    # Prediksi pakai TFLite interpreter
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    class_idx  = np.argmax(prediction[0])
    confidence = float(prediction[0][class_idx]) * 100
    label      = 'ASLI' if class_idx == 1 else 'PALSU'

    return render_template('result.html',
        label        = label,
        confidence   = f'{confidence:.2f}',
        original_img = url_for('static', filename=f'uploads/{filename}'),
    )

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
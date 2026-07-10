import io
import os
import gdown
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm


from flask import Flask, request, render_template, url_for
from PIL import Image, ImageChops, ImageEnhance
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ── Load Model ────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'model_cnn_ela_qris_compressed.h5')
FILE_ID    = "1BdkhFhHN_3_o28gkJXABc6YM70nVXIeM"

if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

model    = tf.keras.models.load_model(MODEL_PATH)
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

# ── Grad-CAM ──────────────────────────────────────────────────────────
def generate_gradcam(img_array, model, last_conv_layer='conv2d_1'):
    conv_layer  = model.get_layer(last_conv_layer)
    conv_model  = tf.keras.Model(inputs=model.inputs,
                                  outputs=conv_layer.output)

    classifier_input = tf.keras.Input(shape=conv_layer.output.shape[1:])
    x     = classifier_input
    found = False
    for layer in model.layers:
        if found:
            x = layer(x)
        if layer.name == last_conv_layer:
            found = True
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)
        predictions  = classifier_model(conv_outputs)
        pred_index   = tf.argmax(predictions[0])
        loss         = predictions[:, pred_index]

    grads   = tape.gradient(loss, conv_outputs)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def save_gradcam_overlay(original_path, heatmap, save_path):
    img         = Image.open(original_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    colormap    = cm.get_cmap('jet')
    heatmap_col = np.uint8(colormap(heatmap)[:, :, :3] * 255)
    heatmap_img = Image.fromarray(heatmap_col).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    overlay     = Image.blend(img, heatmap_img, alpha=0.8)
    overlay.save(save_path)

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

    # Prediksi
    prediction = model.predict(img_array)
    class_idx  = np.argmax(prediction[0])
    confidence = float(prediction[0][class_idx]) * 100
    label      = 'ASLI' if class_idx == 1 else 'PALSU'

    # Grad-CAM
    heatmap      = generate_gradcam(img_array, model)
    gradcam_name = 'gradcam_' + filename
    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_name)
    save_gradcam_overlay(upload_path, heatmap, gradcam_path)

    return render_template('result.html',
        label        = label,
        confidence   = f'{confidence:.2f}',
        original_img = url_for('static', filename=f'uploads/{filename}'),
        gradcam_img  = url_for('static', filename=f'uploads/{gradcam_name}'),
    )

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
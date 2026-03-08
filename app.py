"""
Skin Disease Classifier - Flask App
OOD detection: Softmax Confidence + Entropy (no distance metrics)
Run add_ood_to_existing_model.py first to generate ood_params.npz
"""

import os
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
from tensorflow import keras

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ================== GLOBALS ==================
model              = None
CLASS_NAMES        = None
CONF_THRESHOLD     = None
ENTROPY_THRESHOLD  = None
IMG_SIZE           = (380, 380)
NUM_CLASSES        = 3

# ================== LOADER ==================
def load_resources():
    global model, CLASS_NAMES, CONF_THRESHOLD, ENTROPY_THRESHOLD

    if not os.path.exists('best_production_model.h5'):
        raise FileNotFoundError("best_production_model.h5 not found")

    model = keras.models.load_model('best_production_model.h5', compile=False)
    dummy = np.zeros((1,) + IMG_SIZE + (3,), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)
    print("Model ready")

    ood_path = 'ood_params.npz'
    if not os.path.exists(ood_path):
        raise FileNotFoundError(
            "ood_params.npz not found. Run add_ood_to_existing_model.py first!"
        )

    data              = np.load(ood_path, allow_pickle=True)
    CONF_THRESHOLD    = float(data['conf_threshold'][0])
    ENTROPY_THRESHOLD = float(data['entropy_threshold'][0])
    CLASS_NAMES       = list(data['class_names']) if 'class_names' in data \
                        else ['acne', 'eczema', 'ringworm']

    print(f"OOD params loaded")
    print(f"  Classes            : {CLASS_NAMES}")
    print(f"  Confidence thresh  : {CONF_THRESHOLD:.4f}  (reject if below)")
    print(f"  Entropy thresh     : {ENTROPY_THRESHOLD:.4f}  (reject if above)")

# ================== IMAGE PREP ==================
def prepare_image(stream):
    img = Image.open(stream).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

# ================== PREDICT + OOD ==================
def predict_with_ood(img_array):
    probs      = model.predict(img_array, verbose=0)[0]
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    entropy    = float(-np.sum(probs * np.log(probs + 1e-8)))

    print(f"[OOD] Probs      : { {CLASS_NAMES[i]: round(float(probs[i]),3) for i in range(NUM_CLASSES)} }")
    print(f"[OOD] Confidence : {confidence:.4f} (threshold: >{CONF_THRESHOLD:.4f})", end="")
    print(f"  |  Entropy: {entropy:.4f} (threshold: <{ENTROPY_THRESHOLD:.4f})", end=" -> ")

    # Gate 1: low confidence
    if confidence < CONF_THRESHOLD:
        print(f"REJECTED (confidence {confidence:.3f} < {CONF_THRESHOLD:.3f})")
        return "Unknown / Not Recognized", 0.0, True, confidence, entropy, list(probs)

    # Gate 2: high entropy (uncertain prediction)
    if entropy > ENTROPY_THRESHOLD:
        print(f"REJECTED (entropy {entropy:.3f} > {ENTROPY_THRESHOLD:.3f})")
        return "Unknown / Not Recognized", 0.0, True, confidence, entropy, list(probs)

    print(f"ACCEPTED -> {CLASS_NAMES[pred_idx]} ({confidence*100:.1f}%)")
    return CLASS_NAMES[pred_idx], confidence, False, confidence, entropy, list(probs)

# ================== HTML ==================
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Skin Disease Classifier</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            font-family:'Segoe UI',sans-serif;
            background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
            min-height:100vh; padding:20px; color:#333;
        }
        .container { max-width:900px; margin:0 auto; }
        header {
            text-align:center; color:white; margin-bottom:30px;
            padding:28px 20px; background:rgba(255,255,255,.13);
            border-radius:16px;
        }
        header h1 { font-size:2.3em; margin-bottom:8px; text-shadow:2px 2px 6px rgba(0,0,0,.25); }
        header p  { opacity:.9; font-size:1.05em; }
        main {
            background:white; border-radius:20px;
            padding:40px; box-shadow:0 12px 45px rgba(0,0,0,.22);
        }
        .upload-box {
            border:3px dashed #667eea; border-radius:15px;
            padding:55px 40px; text-align:center; background:#f8f9ff;
            cursor:pointer; transition:all .3s;
        }
        .upload-box:hover { border-color:#764ba2; background:#eff1ff; transform:translateY(-2px); }
        .upload-box .icon { font-size:3.8em; margin-bottom:14px; }
        .upload-box h2 { font-size:1.35em; color:#555; margin-bottom:6px; }
        .upload-box p  { color:#999; }

        .preview-section { text-align:center; margin-top:28px; display:none; }
        #imagePreview {
            max-width:100%; max-height:370px;
            border-radius:12px; box-shadow:0 6px 22px rgba(0,0,0,.12);
        }
        .btn-row { margin-top:18px; }
        .btn {
            padding:12px 30px; border:none; border-radius:25px;
            font-size:1em; font-weight:600; cursor:pointer;
            margin:0 8px; transition:all .2s;
        }
        .btn-primary { background:linear-gradient(135deg,#667eea,#764ba2); color:white; }
        .btn-primary:hover { transform:translateY(-2px); box-shadow:0 4px 14px rgba(102,126,234,.5); }
        .btn-primary:disabled { opacity:.6; cursor:not-allowed; transform:none; }
        .btn-secondary { background:#e8e8e8; color:#444; }
        .btn-secondary:hover { background:#d8d8d8; }

        .loading { text-align:center; padding:40px; display:none; }
        .spinner {
            border:5px solid #eee; border-top:5px solid #667eea;
            border-radius:50%; width:58px; height:58px;
            animation:spin .9s linear infinite; margin:0 auto 14px;
        }
        @keyframes spin { to { transform:rotate(360deg); } }

        .results-section {
            margin-top:34px; padding:28px; border-radius:14px;
            display:none; text-align:center;
        }
        .results-section.accepted { background:#f0faf4; border:2px solid #28a745; }
        .results-section.rejected { background:#fff5f5; border:2px solid #dc3545; }

        .result-label {
            font-size:.82em; text-transform:uppercase;
            letter-spacing:1.5px; color:#aaa; margin-bottom:10px;
        }
        .disease-name {
            font-size:2.3em; font-weight:700;
            text-transform:capitalize; margin-bottom:16px;
        }
        .accepted .disease-name { color:#28a745; }
        .rejected .disease-name { color:#dc3545; }

        .confidence-badge {
            display:inline-block; padding:10px 26px;
            border-radius:25px; font-size:1.35em;
            font-weight:700; color:white; margin-bottom:20px;
        }
        .badge-high   { background:#28a745; }
        .badge-medium { background:#fd7e14; }
        .badge-low    { background:#dc3545; }
        .badge-ood    { background:#6c757d; }

        /* Probability bars */
        .prob-breakdown {
            margin-top:16px; display:inline-block;
            background:rgba(0,0,0,.04); padding:18px 24px;
            border-radius:12px; min-width:300px; text-align:left;
        }
        .prob-breakdown h4 {
            color:#aaa; margin-bottom:14px; font-size:.78em;
            text-transform:uppercase; letter-spacing:.8px;
        }
        .prob-row { margin-bottom:10px; }
        .prob-label {
            display:flex; justify-content:space-between;
            font-size:.9em; margin-bottom:4px;
        }
        .prob-label .cls { color:#555; text-transform:capitalize; }
        .prob-label .pct { font-weight:700; color:#333; }
        .prob-bar-bg {
            background:#e8e8e8; border-radius:6px; height:8px; overflow:hidden;
        }
        .prob-bar-fill {
            height:100%; border-radius:6px;
            background:linear-gradient(90deg,#667eea,#764ba2);
            transition:width .4s ease;
        }
        .ood-signals {
            margin-top:12px; padding-top:12px;
            border-top:1px solid #ddd; font-size:.85em; color:#888;
        }
        .signal-row { display:flex; justify-content:space-between; margin-bottom:4px; }
        .signal-row .rejected-signal { color:#dc3545; font-weight:600; }
        .signal-row .ok-signal { color:#28a745; font-weight:600; }

        .ood-note { margin-top:14px; font-size:.92em; color:#999; line-height:1.5; }
        #errorMsg {
            color:#dc3545; margin-top:18px; display:none;
            font-size:.94em; padding:12px 16px;
            background:#fff5f5; border-radius:8px;
            border:1px solid #f5c6cb;
        }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>&#128298; Skin Disease Classifier</h1>
        <p>Detects Acne &middot; Eczema &middot; Ringworm &nbsp;|&nbsp; Rejects unrelated images</p>
    </header>
    <main>
        <div class="upload-box" id="uploadBox">
            <div class="icon">&#128247;</div>
            <h2>Upload a Skin Image</h2>
            <p>Drag &amp; drop or click to select</p>
            <input type="file" id="fileInput" accept="image/*" hidden>
        </div>

        <div class="preview-section" id="previewSection">
            <img id="imagePreview" src="" alt="Preview">
            <div class="btn-row">
                <button class="btn btn-primary"   id="analyzeBtn">Analyze</button>
                <button class="btn btn-secondary" id="clearBtn">Clear</button>
            </div>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="color:#aaa">Analyzing image...</p>
        </div>

        <div class="results-section" id="resultsSection">
            <div class="result-label"     id="resultLabel">Diagnosis</div>
            <div class="disease-name"     id="diseaseName">-</div>
            <div class="confidence-badge" id="confBadge">-</div>
            <div class="prob-breakdown"   id="probBreakdown" style="display:none"></div>
            <div class="ood-note"         id="oodNote"       style="display:none"></div>
        </div>

        <div id="errorMsg"></div>
    </main>
</div>

<script>
const uploadBox   = document.getElementById('uploadBox');
const fileInput   = document.getElementById('fileInput');
const prevSec     = document.getElementById('previewSection');
const imgPrev     = document.getElementById('imagePreview');
const analyzeBtn  = document.getElementById('analyzeBtn');
const clearBtn    = document.getElementById('clearBtn');
const loading     = document.getElementById('loading');
const resultsSec  = document.getElementById('resultsSection');
const resultLabel = document.getElementById('resultLabel');
const diseaseName = document.getElementById('diseaseName');
const confBadge   = document.getElementById('confBadge');
const probDiv     = document.getElementById('probBreakdown');
const oodNote     = document.getElementById('oodNote');
const errorMsg    = document.getElementById('errorMsg');

let selectedFile = null;

uploadBox.onclick = () => fileInput.click();
fileInput.onchange = e => handleFile(e.target.files[0]);
uploadBox.addEventListener('dragover',  e => { e.preventDefault(); uploadBox.style.borderColor='#764ba2'; });
uploadBox.addEventListener('dragleave', ()=> { uploadBox.style.borderColor='#667eea'; });
uploadBox.addEventListener('drop', e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); });

function handleFile(file) {
    if (!file) return;
    selectedFile = file;
    const r = new FileReader();
    r.onload = e => {
        imgPrev.src = e.target.result;
        uploadBox.style.display  = 'none';
        prevSec.style.display    = 'block';
        resultsSec.style.display = 'none';
        errorMsg.style.display   = 'none';
    };
    r.readAsDataURL(file);
}

clearBtn.onclick = () => {
    selectedFile = null; fileInput.value = ''; imgPrev.src = '';
    uploadBox.style.display  = 'block';
    prevSec.style.display    = 'none';
    resultsSec.style.display = 'none';
    errorMsg.style.display   = 'none';
};

analyzeBtn.onclick = async () => {
    if (!selectedFile) return;
    loading.style.display    = 'block';
    resultsSec.style.display = 'none';
    errorMsg.style.display   = 'none';
    analyzeBtn.disabled      = true;
    const fd = new FormData();
    fd.append('file', selectedFile);
    try {
        const res  = await fetch('/predict', { method:'POST', body:fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Server error');
        showResult(data);
    } catch(err) {
        errorMsg.textContent   = 'Error: ' + err.message;
        errorMsg.style.display = 'block';
    } finally {
        loading.style.display = 'none';
        analyzeBtn.disabled   = false;
    }
};

function showResult(data) {
    resultsSec.className     = 'results-section ' + (data.is_ood ? 'rejected' : 'accepted');
    resultsSec.style.display = 'block';

    if (data.is_ood) {
        resultLabel.textContent = 'Result';
        diseaseName.textContent = 'Not Recognized';
        confBadge.textContent   = 'Out of Distribution';
        confBadge.className     = 'confidence-badge badge-ood';
        oodNote.textContent     = 'This image does not appear to show acne, eczema, or ringworm. Please upload a clear, close-up skin photo.';
        oodNote.style.display   = 'block';
    } else {
        resultLabel.textContent = 'Detected Condition';
        diseaseName.textContent = data.predicted_class;
        const c = data.confidence;
        confBadge.textContent   = (c*100).toFixed(1) + '% Confidence';
        confBadge.className     = 'confidence-badge ' + (c>=0.75?'badge-high':c>=0.55?'badge-medium':'badge-low');
        oodNote.style.display   = 'none';
    }

    // Probability bars + OOD signal breakdown
    if (data.probs && data.class_names) {
        let h = '<h4>Class Probabilities</h4>';
        data.class_names.forEach((cls, i) => {
            const pct = (data.probs[i] * 100).toFixed(1);
            h += `<div class="prob-row">
                    <div class="prob-label">
                        <span class="cls">${cls}</span>
                        <span class="pct">${pct}%</span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:${pct}%"></div>
                    </div>
                  </div>`;
        });

        // OOD signal rows
        const confOk    = data.confidence >= data.conf_threshold;
        const entropyOk = data.entropy    <= data.entropy_threshold;
        h += `<div class="ood-signals">
                <div class="signal-row">
                    <span>Confidence (${(data.confidence*100).toFixed(1)}% vs min ${(data.conf_threshold*100).toFixed(1)}%)</span>
                    <span class="${confOk ? 'ok-signal':'rejected-signal'}">${confOk ? 'PASS' : 'FAIL'}</span>
                </div>
                <div class="signal-row">
                    <span>Entropy (${data.entropy.toFixed(3)} vs max ${data.entropy_threshold.toFixed(3)})</span>
                    <span class="${entropyOk ? 'ok-signal':'rejected-signal'}">${entropyOk ? 'PASS' : 'FAIL'}</span>
                </div>
              </div>`;

        probDiv.innerHTML      = h;
        probDiv.style.display  = 'inline-block';
    } else {
        probDiv.style.display = 'none';
    }
}
</script>
</body>
</html>"""

# ================== ROUTES ==================
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    try:
        img_array = prepare_image(file.stream)
        pred_class, confidence, is_ood, conf_val, entropy_val, probs = predict_with_ood(img_array)
        return jsonify({
            'predicted_class'   : pred_class,
            'confidence'        : confidence,
            'is_ood'            : is_ood,
            'probs'             : [round(float(p), 4) for p in probs],
            'class_names'       : CLASS_NAMES,
            'entropy'           : round(entropy_val, 4),
            'conf_threshold'    : round(CONF_THRESHOLD, 4),
            'entropy_threshold' : round(ENTROPY_THRESHOLD, 4)
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ================== MAIN ==================
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  Skin Disease Classifier")
    print("="*55)
    load_resources()
    print("\nReady at http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
import os, base64, numpy as np
import cv2, mediapipe as mp
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
app.config['SECRET_KEY'] = 'asl_secret'
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024


class ModelAdapter:
    LABELS = [chr(c) for c in range(ord('A'), ord('Z') + 1)] + ['SPACE', 'DELETE', 'NOTHING']

    def __init__(self):
        self.model_path = os.path.join(BASE_DIR, 'model', 'asl_model.h5')
        self.model  = None
        self.ready  = False

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            print(f"⚠️  Model not found: {self.model_path}")
            return
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            self.model = tf.keras.models.load_model(self.model_path)
            self.model.predict(np.zeros((1, 42), dtype='float32'), verbose=0)
            self.ready = True
            print(f"✅  Model loaded — {self.model.input_shape} → {self.model.output_shape}")
        except Exception as e:
            print(f"❌  Model load failed: {e}")

    def predict(self, pil_img):
        if not self.ready:
            return {'hand_detected': False,
                    'error': 'Model not loaded — check model/asl_model.h5 exists'}

        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        res = self.hands.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        if not res.multi_hand_landmarks:
            return {'hand_detected': False, 'letter': None, 'confidence': 0}

        coords = []
        for lm in res.multi_hand_landmarks[0].landmark:
            coords.extend([lm.x, lm.y])
        inp = np.array(coords, dtype='float32').reshape(1, -1)

        preds    = self.model.predict(inp, verbose=0)[0]
        top3_idx = np.argsort(preds)[::-1][:3]
        best     = top3_idx[0]

        return {
            'hand_detected': True,
            'letter':     self.LABELS[best],
            'confidence': round(float(preds[best]), 3),
            'top3': [
                {'letter': self.LABELS[i], 'conf': round(float(preds[i]), 3)}
                for i in top3_idx
            ]
        }


adapter = ModelAdapter()


@app.route('/')
def index():
    return render_template('index.html',
                           model_ready=adapter.ready,
                           model_name=os.path.basename(adapter.model_path))

@app.route('/health')
def health():
    return jsonify({'model_ready': adapter.ready, 'model_path': adapter.model_path})

@app.post('/api/recognize')
def recognize():
    try:
        data_url = request.get_json(force=True).get('frame', '')
        if not data_url.startswith('data:image/'):
            return jsonify({'error': 'Invalid image data'}), 400
        _, encoded = data_url.split(',', 1)
        pil_img = Image.open(BytesIO(base64.b64decode(encoded))).convert('RGB')
        return jsonify(adapter.predict(pil_img))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"\n🤟  ASL → Text  |  http://localhost:5000")
    print(f"    Model ready: {adapter.ready}\n")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

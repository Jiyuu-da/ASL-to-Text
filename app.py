"""
ASL → Text Converter  (Letter + Word mode)
==========================================
- Letter mode: asl_model.h5  (static pose, A-Z)
- Word mode:   word_model.h5  (LSTM on 30-frame sequences)
"""

import os, base64, pickle, collections
import numpy as np, cv2, mediapipe as mp
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
app.config['SECRET_KEY'] = 'asl_secret'
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024

# ── Word dictionary for autocomplete ──────────────────────────────────────
WORD_DICTIONARY = [
    "hello","hi","hey","bye","goodbye","please","thanks","thankyou",
    "sorry","yes","no","okay","ok","help","stop","go","come","wait",
    "i","me","you","we","they","he","she","my","your","name",
    "friend","family","mother","father","sister","brother","baby",
    "want","need","like","love","have","good","bad","happy","sad",
    "hungry","thirsty","tired","sick","hot","cold","water","food",
    "home","school","work","time","today","tomorrow","now","more",
    "again","different","same","big","small","new","old","fast","slow",
    "eat","drink","sleep","walk","run","sit","stand","read","write",
    "talk","listen","see","hear","feel","think","know","learn","play",
    "here","there","where","what","when","why","how","which",
    "cat","dog","book","car","house","money","phone","computer",
    "one","two","three","four","five","many","few","all","some",
    "can","will","not","and","but","with","from","about","very",
]

def get_suggestions(prefix, n=6):
    if not prefix: return []
    p = prefix.lower()
    matches = [w for w in WORD_DICTIONARY if w.startswith(p)]
    return sorted(matches, key=lambda w: (len(w), w))[:n]


# ── MediaPipe (shared) ─────────────────────────────────────────────────────
mp_hands  = mp.solutions.hands
hands_det = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_landmarks_xy(pil_img):
    """Returns (42,) x,y flat array or None."""
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    res = hands_det.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return None
    coords = []
    for lm in res.multi_hand_landmarks[0].landmark:
        coords.extend([lm.x, lm.y])
    return np.array(coords, dtype='float32')


# ── Letter Model ───────────────────────────────────────────────────────────
class LetterModel:
    LABELS = [chr(c) for c in range(ord('A'), ord('Z')+1)] + ['SPACE','DELETE','NOTHING']

    def __init__(self):
        self.model = None
        self.ready = False
        path = os.path.join(BASE_DIR, 'model', 'asl_model.h5')
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            self.model = tf.keras.models.load_model(path)
            self.model.predict(np.zeros((1,42), dtype='float32'), verbose=0)
            self.ready = True
            print(f"✅  Letter model loaded")
        except Exception as e:
            print(f"⚠️  Letter model: {e}")

    def predict(self, pil_img):
        if not self.ready:
            return {'hand_detected': False, 'error': 'Letter model not loaded'}
        lms = get_landmarks_xy(pil_img)
        if lms is None:
            return {'hand_detected': False, 'letter': None, 'confidence': 0}
        preds    = self.model.predict(lms.reshape(1,-1), verbose=0)[0]
        top3_idx = np.argsort(preds)[::-1][:3]
        best     = top3_idx[0]
        return {
            'hand_detected': True,
            'letter':     self.LABELS[best],
            'confidence': round(float(preds[best]), 3),
            'top3': [{'letter': self.LABELS[i], 'conf': round(float(preds[i]),3)} for i in top3_idx]
        }


# ── Word Model (LSTM) ──────────────────────────────────────────────────────
class WordModel:
    SEQ_LEN = 30

    def __init__(self):
        self.model  = None
        self.labels = []
        self.ready  = False
        # Rolling buffer of last SEQ_LEN frames
        self.buffer = collections.deque(maxlen=self.SEQ_LEN)

        model_path  = os.path.join(BASE_DIR, 'model', 'word_model.h5')
        labels_path = os.path.join(BASE_DIR, 'model', 'word_labels.pkl')
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            self.model  = tf.keras.models.load_model(model_path)
            with open(labels_path, 'rb') as f:
                self.labels = pickle.load(f)
            # Warm up
            dummy = np.zeros((1, self.SEQ_LEN, 42), dtype='float32')
            self.model.predict(dummy, verbose=0)
            self.ready = True
            print(f"✅  Word model loaded — {self.labels}")
        except Exception as e:
            print(f"⚠️  Word model not ready: {e}")
            print("    Run: python model/collect_words.py  then  python model/train_words.py")

    def predict(self, pil_img):
        if not self.ready:
            return {'hand_detected': False,
                    'error': 'Word model not trained yet — run collect_words.py + train_words.py'}

        lms = get_landmarks_xy(pil_img)

        if lms is None:
            # Keep buffer but don't add empty frame
            return {'hand_detected': False, 'word': None, 'confidence': 0,
                    'buffer_fill': len(self.buffer), 'buffer_needed': self.SEQ_LEN}

        self.buffer.append(lms)

        if len(self.buffer) < self.SEQ_LEN:
            return {
                'hand_detected': True,
                'word': None,
                'confidence': 0,
                'buffer_fill': len(self.buffer),
                'buffer_needed': self.SEQ_LEN,
                'collecting': True   # tells UI to show "collecting frames" state
            }

        seq   = np.array(self.buffer, dtype='float32').reshape(1, self.SEQ_LEN, 42)
        preds = self.model.predict(seq, verbose=0)[0]
        top3_idx = np.argsort(preds)[::-1][:3]
        best  = top3_idx[0]
        conf  = float(preds[best])

        return {
            'hand_detected': True,
            'word':       self.labels[best],
            'confidence': round(conf, 3),
            'buffer_fill': self.SEQ_LEN,
            'buffer_needed': self.SEQ_LEN,
            'top3': [{'word': self.labels[i], 'conf': round(float(preds[i]),3)} for i in top3_idx]
        }

    def clear_buffer(self):
        self.buffer.clear()


# ── Initialise models ─────────────────────────────────────────────────────
letter_model = LetterModel()
word_model   = WordModel()


# ── Routes ────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           letter_ready=letter_model.ready,
                           word_ready=word_model.ready)

@app.route('/health')
def health():
    return jsonify({
        'letter_model': letter_model.ready,
        'word_model':   word_model.ready,
        'word_labels':  word_model.labels
    })

@app.post('/api/recognize')
def recognize():
    try:
        body    = request.get_json(force=True)
        mode    = body.get('mode', 'letter')
        data_url= body.get('frame', '')
        if not data_url.startswith('data:image/'):
            return jsonify({'error': 'Invalid image'}), 400
        _, encoded = data_url.split(',', 1)
        pil_img = Image.open(BytesIO(base64.b64decode(encoded))).convert('RGB')

        if mode == 'word':
            return jsonify(word_model.predict(pil_img))
        else:
            return jsonify(letter_model.predict(pil_img))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.post('/api/clear_buffer')
def clear_buffer():
    word_model.clear_buffer()
    return jsonify({'ok': True})

@app.get('/api/suggest')
def suggest():
    prefix = request.args.get('q','').strip()
    return jsonify({'suggestions': get_suggestions(prefix)})


if __name__ == '__main__':
    print(f"\n🤟  ASL → Text  |  http://localhost:5000")
    print(f"    Letter model: {letter_model.ready}")
    print(f"    Word model:   {word_model.ready}\n")
    # app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)

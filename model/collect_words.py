"""
ASL Word Data Collector
========================
Collects 30 sequences per word for LSTM training.
Each sequence = 30 frames of MediaPipe landmarks.

Usage:
  python model/collect_words.py

Controls:
  SPACE  — start recording a sequence
  Q      — quit and save
  R      — redo last sequence

Reference signs (learn from https://www.handspeak.com):
  HELLO  — open hand wave from forehead
  THANKS — flat hand from chin moving forward
  SORRY  — fist circular motion on chest
  YES    — fist nodding up and down
  NO     — index+middle finger snap to thumb
  PLEASE — flat hand circular on chest
  HELP   — fist on palm, lift up
  GOOD   — flat hand from chin forward and down
  BAD    — flat hand from chin, flip down
  LOVE   — arms crossed on chest
"""

import cv2, os, numpy as np, time
import mediapipe as mp

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, 'model', 'word_data')
WORDS       = ['hello', 'thanks', 'sorry', 'yes', 'no', 'please', 'help', 'good', 'bad', 'love']
SEQ_LEN     = 30   # frames per sequence
NUM_SEQ     = 30   # sequences per word
os.makedirs(DATA_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.6)

def get_landmarks(frame_rgb):
    res = hands.process(frame_rgb)
    if not res.multi_hand_landmarks:
        return np.zeros(42)   # 21 × x,y
    lms = res.multi_hand_landmarks[0].landmark
    return np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32).flatten()

def count_existing(word):
    d = os.path.join(DATA_DIR, word)
    if not os.path.exists(d): return 0
    return len([f for f in os.listdir(d) if f.endswith('.npy')])

cap = cv2.VideoCapture(0)
word_idx = 0

print("\n🤟  ASL Word Data Collector")
print(f"   Words: {WORDS}")
print("   SPACE = record  |  N = next word  |  R = redo  |  Q = quit\n")

while word_idx < len(WORDS):
    word    = WORDS[word_idx]
    word_dir = os.path.join(DATA_DIR, word)
    os.makedirs(word_dir, exist_ok=True)
    collected = count_existing(word)

    print(f"\n👉  Current word: '{word.upper()}'  ({collected}/{NUM_SEQ} sequences)")
    print(f"   Learn the sign at: https://www.handspeak.com/word/search/index.php?id={word}")

    state = 'idle'   # idle | countdown | recording | done
    seq_frames = []
    countdown_start = None

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = hands.process(rgb)

        # Draw landmarks
        if res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        collected = count_existing(word)
        h, w = frame.shape[:2]

        # ── State machine ──
        if state == 'idle':
            cv2.putText(frame, f"Word: {word.upper()}  [{collected}/{NUM_SEQ}]",
                        (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(frame, "Press SPACE to record",
                        (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,120), 2)
            if not res.multi_hand_landmarks:
                cv2.putText(frame, "Show your hand first",
                            (10,115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,100,255), 2)

        elif state == 'countdown':
            elapsed = time.time() - countdown_start
            remaining = 3 - int(elapsed)
            cv2.putText(frame, f"GET READY: {remaining}",
                        (w//2-120, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,220,255), 4)
            if elapsed >= 3:
                state = 'recording'
                seq_frames = []

        elif state == 'recording':
            lms = get_landmarks(rgb)
            seq_frames.append(lms)
            progress = len(seq_frames) / SEQ_LEN
            bar_x = int(w * progress)
            cv2.rectangle(frame, (0, h-8), (bar_x, h), (0,220,120), -1)
            cv2.putText(frame, f"RECORDING... {len(seq_frames)}/{SEQ_LEN}",
                        (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,220,120), 2)

            if len(seq_frames) == SEQ_LEN:
                seq = np.array(seq_frames)
                idx = count_existing(word)
                np.save(os.path.join(word_dir, f"{idx:03d}.npy"), seq)
                print(f"  ✅  '{word}' sequence {idx+1}/{NUM_SEQ} saved")
                state = 'idle'
                seq_frames = []

                if count_existing(word) >= NUM_SEQ:
                    state = 'done'

        elif state == 'done':
            cv2.putText(frame, f"'{word.upper()}' COMPLETE! ✓",
                        (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,120), 2)
            cv2.putText(frame, "Press N for next word",
                        (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Progress grid
        for i, w_label in enumerate(WORDS):
            n = count_existing(w_label)
            done = n >= NUM_SEQ
            col = (0,200,80) if done else (80,80,80)
            x = 10 + (i % 5) * 110
            y = frame.shape[0] - 55 + (i // 5) * 28
            cv2.rectangle(frame, (x,y), (x+100,y+22), col, -1)
            cv2.putText(frame, f"{w_label[:6]} {n}", (x+3,y+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0) if done else (200,200,200), 1)

        cv2.imshow('ASL Word Collector — Q to quit', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cap.release(); cv2.destroyAllWindows()
            print("\n✅  Data saved. Run:  python model/train_words.py")
            exit()
        elif key == ord('n') or (state == 'done' and key != 255):
            word_idx += 1; break
        elif key == 32 and state == 'idle' and res.multi_hand_landmarks:  # SPACE
            state = 'countdown'; countdown_start = time.time()
        elif key == ord('r') and state == 'idle':
            # Redo: delete last saved sequence
            idx = count_existing(word) - 1
            f   = os.path.join(word_dir, f"{idx:03d}.npy")
            if os.path.exists(f):
                os.remove(f)
                print(f"  🗑️  Removed last sequence for '{word}'")

cap.release()
cv2.destroyAllWindows()
print("\n✅  All words collected!")
print("Next step:  python model/train_words.py")

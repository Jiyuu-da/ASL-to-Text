"""
ASL Hand Gesture Recognition - Training Script
===============================================
Uses MediaPipe to extract hand landmarks from images, then trains
a Random Forest classifier on those landmark features.

Dataset Options (choose one):
  1. Kaggle ASL Alphabet: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
     - 87,000 images, 29 classes (A-Z + space + delete + nothing)
     - Best accuracy (~99%)
  
  2. Kaggle ASL Dataset: https://www.kaggle.com/datasets/ayuraj/asl-dataset
     - Smaller, faster to train
  
  3. Demo Mode (no dataset needed):
     - Generates synthetic data to demonstrate the pipeline
     - ~60-70% accuracy (for testing only)

Usage:
  # Demo mode (no dataset):
  python model/train.py --demo

  # With Kaggle dataset (after downloading and extracting):
  python model/train.py --data_dir /path/to/asl_alphabet_train
"""

import os
import sys
import pickle
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def generate_demo_data():
    """Generate synthetic landmark data for demo/testing"""
    print("🔧 Generating synthetic demo data...")
    
    labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['space', 'del', 'nothing']
    n_samples_per_class = 200
    n_features = 63  # 21 landmarks × 3 (x, y, z)
    
    X, y = [], []
    np.random.seed(42)
    
    for i, label in enumerate(labels):
        # Each class gets a distinct "center" in feature space
        center = np.zeros(n_features)
        # Simulate distinct hand positions per letter
        center[i % n_features] = 1.0
        center[(i * 3) % n_features] = 0.5
        
        samples = center + np.random.randn(n_samples_per_class, n_features) * 0.15
        X.extend(samples)
        y.extend([label] * n_samples_per_class)
    
    return np.array(X), np.array(y)

def load_dataset_with_mediapipe(data_dir):
    """Load real dataset using MediaPipe landmark extraction"""
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        print("❌ OpenCV or MediaPipe not installed. Run: pip install opencv-python mediapipe")
        sys.exit(1)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    X, y = [], []
    classes = sorted(os.listdir(data_dir))
    print(f"📂 Found {len(classes)} classes: {classes}")
    
    for label in classes:
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        
        images = [f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Limit to 1000 images per class for speed
        images = images[:1000]
        
        count = 0
        for img_file in images:
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                lms = results.multi_hand_landmarks[0]
                wrist = lms.landmark[0]
                features = []
                for lm in lms.landmark:
                    features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                X.append(features)
                y.append(label.upper())
                count += 1
        
        print(f"  ✅ {label}: {count} samples")
    
    hands.close()
    return np.array(X), np.array(y)

def train(X, y, output_path):
    """Train Random Forest classifier"""
    print(f"\n🏋️  Training on {len(X)} samples across {len(set(y))} classes...")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Test Accuracy: {acc:.2%}")
    
    if acc > 0.5:
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({'model': clf, 'label_encoder': le}, f)
    
    print(f"\n✅ Model saved to: {output_path}")
    return clf, le

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ASL Recognition Model')
    parser.add_argument('--demo', action='store_true', help='Use synthetic demo data')
    parser.add_argument('--data_dir', type=str, help='Path to ASL dataset directory')
    parser.add_argument('--output', type=str, default='model/asl_model.pkl', help='Output model path')
    args = parser.parse_args()
    
    if args.demo:
        X, y = generate_demo_data()
    elif args.data_dir:
        X, y = load_dataset_with_mediapipe(args.data_dir)
    else:
        print("ℹ️  No arguments provided. Running in demo mode.")
        print("    For real dataset: python model/train.py --data_dir /path/to/dataset")
        print("    For demo:         python model/train.py --demo\n")
        X, y = generate_demo_data()
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.output)
    train(X, y, model_path)
    print("\n🚀 Done! Now run: python app.py")

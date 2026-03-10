"""
ASL Word LSTM Trainer — Anti-Overfitting Edition
=================================================
Fixes 100% training accuracy (overfitting) using:
  1. Heavy data augmentation  — turns 30 sequences into 300+
  2. High dropout             — forces generalization
  3. Smaller model            — less capacity to memorize
  4. L2 regularization        — penalizes large weights
  5. Label smoothing          — stops overconfident predictions

No new data collection needed.
"""

import os, sys, pickle
import numpy as np

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, 'model', 'word_data')
OUT_MODEL = os.path.join(BASE_DIR, 'model', 'word_model.h5')
OUT_LABELS= os.path.join(BASE_DIR, 'model', 'word_labels.pkl')
SEQ_LEN   = 30


def augment_sequence(seq, n=9):
    """
    Augment one (30, 42) sequence into n+1 variations.
    Simulates natural hand variation: slight position shifts,
    scale changes, speed changes, noise — all things that happen
    in real live signing.
    """
    augmented = [seq]
    for _ in range(n):
        s = seq.copy()

        # 1. Gaussian noise — simulates landmark jitter
        s += np.random.randn(*s.shape).astype(np.float32) * 0.01

        # 2. Scale — simulates hand closer/further from camera
        scale = np.random.uniform(0.85, 1.15)
        s *= scale

        # 3. Translation — simulates hand in different part of frame
        shift = np.random.uniform(-0.05, 0.05, size=(1, 42)).astype(np.float32)
        s += shift

        # 4. Time warp — simulates signing faster or slower
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            new_len = max(10, int(SEQ_LEN * factor))
            indices = np.linspace(0, SEQ_LEN-1, new_len).astype(int)
            s_warped = s[indices]
            final_indices = np.linspace(0, len(s_warped)-1, SEQ_LEN).astype(int)
            s = s_warped[final_indices]

        # 5. Random frame dropout — simulates missed detections
        if np.random.rand() > 0.7:
            drop_idx = np.random.randint(0, SEQ_LEN)
            s[drop_idx] = s[max(0, drop_idx-1)]

        augmented.append(s.astype(np.float32))

    return augmented


def load_data():
    words = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])
    if not words:
        print(f"❌  No data in {DATA_DIR} — run collect_words.py first")
        sys.exit(1)

    print(f"📂  Words: {words}\n")
    X_raw, y_raw = [], []

    for idx, word in enumerate(words):
        word_dir = os.path.join(DATA_DIR, word)
        seqs = sorted([f for f in os.listdir(word_dir) if f.endswith('.npy')])
        count = 0
        for sf in seqs:
            seq = np.load(os.path.join(word_dir, sf))
            if seq.shape == (SEQ_LEN, 42):
                X_raw.append(seq)
                y_raw.append(idx)
                count += 1
        print(f"  {word:12s}: {count} raw sequences")

    X_raw = np.array(X_raw, dtype=np.float32)
    y_raw = np.array(y_raw, dtype=np.int32)

    # Augment
    print(f"\n🔀  Augmenting {len(X_raw)} → ", end='')
    X_aug, y_aug = [], []
    for seq, label in zip(X_raw, y_raw):
        for aug_seq in augment_sequence(seq, n=9):
            X_aug.append(aug_seq)
            y_aug.append(label)

    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug, dtype=np.int32)
    print(f"{len(X_aug)} sequences (10× augmentation)")

    return X_aug, y_aug, words


def train(X, y, words):
    try:
        import tensorflow as tf
        from tensorflow import keras
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        print("❌  pip install tensorflow")
        sys.exit(1)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    n_classes = len(words)
    print(f"\n📊  {len(X)} sequences | {n_classes} classes | {X.shape[1]} frames × {X.shape[2]} features")

    y_cat = tf.keras.utils.to_categorical(y, n_classes)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_cat, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    Train: {len(X_tr)}  Test: {len(X_te)}\n")

    reg = keras.regularizers.l2(1e-4)

    model = keras.Sequential([
        keras.layers.Input(shape=(SEQ_LEN, 42)),
        keras.layers.Bidirectional(
            keras.layers.LSTM(32, return_sequences=True,
                              kernel_regularizer=reg, recurrent_regularizer=reg)
        ),
        keras.layers.Dropout(0.5),
        keras.layers.Bidirectional(
            keras.layers.LSTM(32, return_sequences=False,
                              kernel_regularizer=reg, recurrent_regularizer=reg)
        ),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(n_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=25, restore_best_weights=True,
            monitor='val_accuracy', verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=10, factor=0.5, min_lr=1e-5, verbose=1
        ),
    ]

    print("\n🏋️  Training...")
    model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=300,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    y_true = np.argmax(y_te, axis=1)
    acc = accuracy_score(y_true, y_pred)

    print(f"\n🎯  Test Accuracy: {acc:.2%}")
    if acc == 1.0:
        print("    ⚠️  Still 100% — consider collecting more varied data")
    elif acc >= 0.85:
        print("    ✅  Good — should generalise well to live signing")
    else:
        print("    ⚠️  Low — try collecting more sequences")

    print(classification_report(y_true, y_pred, target_names=words))

    model.save(OUT_MODEL)
    with open(OUT_LABELS, 'wb') as f:
        pickle.dump(words, f)

    print(f"\n✅  Model → {OUT_MODEL}")
    print(f"✅  Labels → {OUT_LABELS}")
    print("\n🚀  Run:  python app.py")


if __name__ == '__main__':
    np.random.seed(42)
    X, y, words = load_data()
    train(X, y, words)

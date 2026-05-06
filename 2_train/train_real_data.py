import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import os
import time

print("=" * 60)
print("  TEST KONSEP: SPLIT CNN + LSTM Waste Classifier")
print("=" * 60)

# ============================================================
# KONFIGURASI
# ============================================================
IMG_HEIGHT = 48
IMG_WIDTH = 48
NUM_CHANNELS = 3
SEQUENCE_LENGTH = 3  # Dikurangi menjadi 3 (awal, tengah, akhir) untuk real-time
NUM_CLASSES = 3
CLASS_NAMES = ["kertas", "plastik", "organik"]

BATCH_SIZE = 8
EPOCHS = 20

# ============================================================
# STEP 1: Load Data + Augmentasi (Stratified Split + Balanced)
# ============================================================
def augment_sequence(seq):
    """Augmentasi satu sequence (3 frame) dengan transformasi yang sama."""
    from scipy.ndimage import rotate, shift, zoom as ndzoom

    # Random horizontal flip
    if np.random.random() > 0.5:
        seq = seq[:, :, ::-1, :]

    # Random rotation (-15 to +15 degrees)
    if np.random.random() > 0.3:
        angle = np.random.uniform(-15, 15)
        for i in range(seq.shape[0]):
            seq[i] = rotate(seq[i], angle, axes=(0,1), reshape=False, mode='reflect')

    # Random zoom (0.85 to 1.15)
    if np.random.random() > 0.3:
        z = np.random.uniform(0.85, 1.15)
        for i in range(seq.shape[0]):
            h, w, c = seq[i].shape
            zoomed = ndzoom(seq[i], (z, z, 1), mode='reflect')
            zh, zw = zoomed.shape[:2]
            # Center crop/pad back to original size
            if zh >= h and zw >= w:
                sh = (zh - h) // 2
                sw = (zw - w) // 2
                seq[i] = zoomed[sh:sh+h, sw:sw+w, :]
            else:
                pad_h = (h - zh) // 2
                pad_w = (w - zw) // 2
                result = np.zeros_like(seq[i])
                result[pad_h:pad_h+zh, pad_w:pad_w+zw, :] = zoomed
                seq[i] = result

    # Random shift (up to 4 pixels)
    if np.random.random() > 0.5:
        dx = np.random.uniform(-4, 4)
        dy = np.random.uniform(-4, 4)
        for i in range(seq.shape[0]):
            seq[i] = shift(seq[i], (dy, dx, 0), mode='reflect')

    # Random brightness adjustment
    brightness = np.random.uniform(-0.3, 0.3)
    seq = np.clip(seq + brightness, 0, 1)

    # Random contrast adjustment
    contrast = np.random.uniform(0.7, 1.4)
    mean = np.mean(seq, axis=(1, 2, 3), keepdims=True)
    seq = np.clip((seq - mean) * contrast + mean, 0, 1)

    # Random color jitter per channel
    for ch in range(3):
        jitter = np.random.uniform(-0.15, 0.15)
        seq[:, :, :, ch] = np.clip(seq[:, :, :, ch] + jitter, 0, 1)

    # Random Gaussian noise
    if np.random.random() > 0.5:
        noise = np.random.normal(0, 0.03, seq.shape)
        seq = np.clip(seq + noise, 0, 1)

    return seq.astype(np.float32)

def load_raw_data(data_dir="dataset", seq_len=3, img_h=48, img_w=48, class_names=["kertas", "plastik", "organik"]):
    """Load data asli TANPA augmentasi."""
    X_list = []
    y_list = []

    for label, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir): continue

        sessions = sorted([d for d in os.listdir(cls_dir) if d.startswith('sesi_')])
        for session in sessions:
            sess_dir = os.path.join(cls_dir, session)
            frames = sorted([f for f in os.listdir(sess_dir) if f.endswith('.jpg')])
            if len(frames) < seq_len: continue

            indices = np.linspace(0, len(frames)-1, seq_len, dtype=int)
            seq_images = []
            for idx in indices:
                img_path = os.path.join(sess_dir, frames[idx])
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_h, img_w))
                img_arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                seq_images.append(img_arr)

            X_list.append(np.array(seq_images, dtype=np.float32))
            y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    for i, cls in enumerate(class_names):
        print(f"  {cls}: {np.sum(y == i)} sesi asli")
    print(f"[INFO] Total {len(X)} sesi asli dari {data_dir}")
    return X, y

def stratified_split(X, y, val_ratio=0.2):
    """Split data per kelas agar semua kelas ada di train dan val."""
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    for label in np.unique(y):
        mask = (y == label)
        X_cls = X[mask]
        y_cls = y[mask]
        n = len(X_cls)

        perm = np.random.permutation(n)
        X_cls, y_cls = X_cls[perm], y_cls[perm]

        n_val = max(1, int(n * val_ratio))  # minimal 1 untuk validasi
        X_val_list.append(X_cls[:n_val])
        y_val_list.append(y_cls[:n_val])
        X_train_list.append(X_cls[n_val:])
        y_train_list.append(y_cls[n_val:])

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_val = np.concatenate(X_val_list)
    y_val = np.concatenate(y_val_list)

    return X_train, y_train, X_val, y_val

def augment_balanced(X_train, y_train, target_per_class=120):
    """Augment sehingga setiap kelas punya jumlah sample yang sama."""
    X_aug = list(X_train)
    y_aug = list(y_train)

    for label in np.unique(y_train):
        mask = (y_train == label)
        X_cls = X_train[mask]
        n_original = len(X_cls)
        n_needed = target_per_class - n_original

        for _ in range(max(0, n_needed)):
            idx = np.random.randint(0, n_original)
            aug = augment_sequence(X_cls[idx].copy())
            X_aug.append(aug)
            y_aug.append(label)

    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug, dtype=np.int32)

    for i, cls in enumerate(CLASS_NAMES):
        cnt = np.sum(y_aug == i)
        print(f"  {cls}: {cnt} (setelah augmentasi)")
    return X_aug, y_aug

# Load raw data
X_raw, y_raw = load_raw_data(seq_len=SEQUENCE_LENGTH, img_h=IMG_HEIGHT, img_w=IMG_WIDTH, class_names=CLASS_NAMES)

# STRATIFIED split SEBELUM augmentasi (no data leakage, semua kelas terwakili!)
np.random.seed(42)
X_train_raw, y_train_raw, X_val, y_val = stratified_split(X_raw, y_raw, val_ratio=0.2)

print(f"\n[INFO] Sebelum augmentasi:")
print(f"  Train: {len(X_train_raw)} | Val: {len(X_val)}")
for i, cls in enumerate(CLASS_NAMES):
    n_tr = np.sum(y_train_raw == i)
    n_va = np.sum(y_val == i)
    print(f"    {cls}: train={n_tr}, val={n_va}")

# Augmentasi BALANCED — semua kelas jadi jumlah yang sama
print("\n[INFO] Augmentasi balanced...")
X_train, y_train = augment_balanced(X_train_raw, y_train_raw, target_per_class=200)

# Shuffle training set
perm = np.random.permutation(len(X_train))
X_train, y_train = X_train[perm], y_train[perm]

print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# Class weights tetap dihitung sebagai safety net
unique, counts = np.unique(y_train, return_counts=True)
n_samples = len(y_train)
n_classes = len(unique)
class_weight = {}
for cls, cnt in zip(unique, counts):
    class_weight[int(cls)] = n_samples / (n_classes * cnt)
print(f"Class weights: {class_weight}")

# ============================================================
# STEP 2: Bangun Model End-to-End
# ============================================================
def build_cnn_feature_extractor(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Gunakan MobileNetV2 dengan alpha=0.35 + Transfer Learning dari ImageNet
    # PENTING: weights='imagenet' agar fitur visual sudah terlatih
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=0.35,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model — hanya fine-tune Dense projection layer
    base_model.trainable = False
    
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Proyeksikan 1280 fitur menjadi 64 fitur agar LSTM tidak kelebihan beban
    x = layers.Dense(64, activation='relu')(x)
    
    return Model(inputs, x, name="mobilenetv2_extractor")

cnn_extractor = build_cnn_feature_extractor((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))

# ============================================================
# TAHAP BARU: Manual LSTM Cell (Bypass TFLite Bug)
# Secara teori 100% LSTM murni (Forget, Input, Output gates)
# Secara komputasi murni matematika dasar (didukung 100% oleh ESP-NN)
# ============================================================
def apply_manual_lstm(inputs, units=32):
    x_0 = inputs[:, 0, :]
    x_1 = inputs[:, 1, :]
    x_2 = inputs[:, 2, :]
    
    # Initialize zero states tanpa menggunakan op FILL
    zero_dense_h = layers.Dense(units, kernel_initializer='zeros', bias_initializer='zeros', trainable=False, name="init_h")
    zero_dense_c = layers.Dense(units, kernel_initializer='zeros', bias_initializer='zeros', trainable=False, name="init_c")
    
    h_state = zero_dense_h(x_0)
    c_state = zero_dense_c(x_0)
    
    def lstm_cell(x_t, h_prev, c_prev, step):
        concat = layers.Concatenate(name=f"concat_{step}")([x_t, h_prev])
        f_t = layers.Dense(units, activation='sigmoid', name=f"f_{step}")(concat)
        i_t = layers.Dense(units, activation='sigmoid', name=f"i_{step}")(concat)
        o_t = layers.Dense(units, activation='sigmoid', name=f"o_{step}")(concat)
        c_prime = layers.Dense(units, activation='tanh', name=f"cprime_{step}")(concat)
        
        f_mul_c = layers.Multiply(name=f"f_mul_c_{step}")([f_t, c_prev])
        i_mul_cprime = layers.Multiply(name=f"i_mul_cprime_{step}")([i_t, c_prime])
        c_t = layers.Add(name=f"c_t_{step}")([f_mul_c, i_mul_cprime])
        
        tanh_c_t = layers.Activation('tanh', name=f"tanh_c_t_{step}")(c_t)
        h_t = layers.Multiply(name=f"h_t_{step}")([o_t, tanh_c_t])
        return h_t, c_t

    h_state, c_state = lstm_cell(x_0, h_state, c_state, 0)
    h_state, c_state = lstm_cell(x_1, h_state, c_state, 1)
    h_state, c_state = lstm_cell(x_2, h_state, c_state, 2)
    return h_state

# Buat Temporal Head Model
temporal_input = layers.Input(shape=(SEQUENCE_LENGTH, 64), name="temporal_input")
h_final = apply_manual_lstm(temporal_input, units=32)
h_drop = layers.Dropout(0.3)(h_final)
t_out = layers.Dense(NUM_CLASSES, activation='softmax', name="dense_out")(h_drop)
temporal_head = Model(temporal_input, t_out, name="temporal_head")

# Buat Combined Model (End-to-End)
inputs = layers.Input(shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
cnn_features = layers.TimeDistributed(cnn_extractor)(inputs)
outputs = temporal_head(cnn_features)

combined_model = Model(inputs, outputs, name="combined_model")

# Print model architectures
print("\n" + "=" * 60)
print("  MODEL ARCHITECTURES")
print("=" * 60)
print("\n--- CNN Feature Extractor ---")
cnn_extractor.summary()
print("\n--- Temporal Head (Manual LSTM) ---")
temporal_head.summary()
print("\n--- Combined Model ---")
combined_model.summary()

# ============================================================
# STEP 3: Training (2 Phase — Freeze lalu Fine-tune)
# ============================================================

# --- Phase 1: Freeze backbone, latih head saja ---
print("\n[INFO] Phase 1: Training head only (backbone frozen)...")
combined_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
lr_cb1 = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
history1 = combined_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                   batch_size=BATCH_SIZE, epochs=15, verbose=1,
                   class_weight=class_weight, callbacks=[lr_cb1])

# --- Phase 2: Unfreeze backbone, fine-tune semua layer ---
print("\n[INFO] Phase 2: Fine-tuning seluruh model...")
# Dapatkan base_model dari dalam cnn_extractor
base_model_layer = cnn_extractor.layers[1]  # MobileNetV2 wrapper
base_model_layer.trainable = True

# Recompile dengan learning rate lebih rendah
combined_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
lr_cb2 = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
history2 = combined_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                   batch_size=BATCH_SIZE, epochs=30, verbose=1,
                   class_weight=class_weight, callbacks=[lr_cb2])

# ============================================================
# STEP 4: Evaluasi Lengkap
# ============================================================
print("\n" + "=" * 60)
print("  EVALUASI MODEL")
print("=" * 60)

# Training history summary
print("\n--- Phase 1 History (Last 5 Epochs) ---")
print(f"{'Epoch':>6} {'Loss':>10} {'Acc':>8} {'Val Loss':>10} {'Val Acc':>8}")
for i in range(max(0, len(history1.history['loss'])-5), len(history1.history['loss'])):
    print(f"{i+1:>6} {history1.history['loss'][i]:>10.4f} {history1.history['accuracy'][i]:>8.4f} {history1.history['val_loss'][i]:>10.4f} {history1.history['val_accuracy'][i]:>8.4f}")

print(f"\n--- Phase 2 History (Last 5 Epochs) ---")
print(f"{'Epoch':>6} {'Loss':>10} {'Acc':>8} {'Val Loss':>10} {'Val Acc':>8}")
for i in range(max(0, len(history2.history['loss'])-5), len(history2.history['loss'])):
    print(f"{i+1:>6} {history2.history['loss'][i]:>10.4f} {history2.history['accuracy'][i]:>8.4f} {history2.history['val_loss'][i]:>10.4f} {history2.history['val_accuracy'][i]:>8.4f}")

# Final evaluation
print("\n--- Final Evaluation ---")
train_loss, train_acc = combined_model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc = combined_model.evaluate(X_val, y_val, verbose=0)
print(f"  Training   — Loss: {train_loss:.4f}, Accuracy: {train_acc*100:.1f}%")
print(f"  Validation — Loss: {val_loss:.4f}, Accuracy: {val_acc*100:.1f}%")

# Per-class evaluation
print("\n--- Per-Class Accuracy (Validation Set) ---")
val_preds = combined_model.predict(X_val, verbose=0)
val_pred_classes = np.argmax(val_preds, axis=1)

for i, cls in enumerate(CLASS_NAMES):
    mask = (y_val == i)
    if np.sum(mask) == 0:
        print(f"  {cls}: no samples in validation")
        continue
    cls_acc = np.mean(val_pred_classes[mask] == i)
    print(f"  {cls}: {cls_acc*100:.1f}% ({np.sum(val_pred_classes[mask] == i)}/{np.sum(mask)})")

# Confusion matrix
print("\n--- Confusion Matrix ---")
print(f"{'':>12}", end="")
for cls in CLASS_NAMES:
    print(f"{cls:>10}", end="")
print()
for i, cls_true in enumerate(CLASS_NAMES):
    print(f"{cls_true:>12}", end="")
    for j in range(NUM_CLASSES):
        count = np.sum((y_val == i) & (val_pred_classes == j))
        print(f"{count:>10}", end="")
    print()

# Save full training history
import json
full_history = {
    "phase1": {k: [float(v) for v in vals] for k, vals in history1.history.items()},
    "phase2": {k: [float(v) for v in vals] for k, vals in history2.history.items()},
}
with open("training_history.json", "w") as f:
    json.dump(full_history, f, indent=2)
print("\n[OK] Training history saved to training_history.json")

# ============================================================
# STEP 5: Split Model 
# ============================================================
print("\n[INFO] Split Model Selesai secara otomatis karena komposisi Keras!")
# Model cnn_extractor dan temporal_head sudah terupdate bobotnya.

# ============================================================
# STEP 6: Konversi TFLite
# ============================================================
print("\n[INFO] Converting to TFLite INT8...")

def representative_dataset_cnn():
    for i in range(min(50, len(X_train))):
        yield [X_train[i, 0:1].astype(np.float32)] # 1 frame saja

def representative_dataset_temporal():
    # Buat dataset temporal head dari output CNN
    for i in range(min(50, len(X_train))):
        sample = X_train[i:i+1]
        features = np.zeros((1, SEQUENCE_LENGTH, 64), dtype=np.float32)
        for t in range(SEQUENCE_LENGTH):
            features[0, t, :] = cnn_extractor.predict(sample[:, t], verbose=0)[0]
        yield [features]

# Konversi CNN
conv_cnn = tf.lite.TFLiteConverter.from_keras_model(cnn_extractor)
conv_cnn.optimizations = [tf.lite.Optimize.DEFAULT]
conv_cnn.representative_dataset = representative_dataset_cnn
conv_cnn.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv_cnn.inference_input_type = tf.int8
conv_cnn.inference_output_type = tf.int8
tflite_cnn = conv_cnn.convert()

# Konversi Temporal Head (Dense-based, 100% kompatibel TFLM)
conv_temporal = tf.lite.TFLiteConverter.from_keras_model(temporal_head)
conv_temporal.optimizations = [tf.lite.Optimize.DEFAULT]
conv_temporal.representative_dataset = representative_dataset_temporal
conv_temporal.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv_temporal.inference_input_type = tf.int8
conv_temporal.inference_output_type = tf.int8
tflite_temporal = conv_temporal.convert()

save_dir = "saved_models_split"
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, "waste_cnn_int8.tflite"), 'wb') as f:
    f.write(tflite_cnn)
with open(os.path.join(save_dir, "waste_lstm_int8.tflite"), 'wb') as f:
    f.write(tflite_temporal)

print(f"\n  [OK] CNN model: {len(tflite_cnn)/1024:.1f} KB")
print(f"  [OK] Temporal Head model: {len(tflite_temporal)/1024:.1f} KB")

# Export to C Header
os.system(f"xxd -i {os.path.join(save_dir, 'waste_cnn_int8.tflite')} > {os.path.join(save_dir, 'model_cnn_data.h')}")
os.system(f"xxd -i {os.path.join(save_dir, 'waste_lstm_int8.tflite')} > {os.path.join(save_dir, 'model_lstm_data.h')}")

# INT8 model test
print("\n--- INT8 Quantized Model Test ---")
import subprocess
result = subprocess.run(['python', 'test_model.py'], capture_output=True, text=True)
print(result.stdout)

print("\n" + "=" * 60)
print("  TRAINING SELESAI!")
print("=" * 60)


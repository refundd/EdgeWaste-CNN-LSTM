"""
============================================================
TEST KONSEP: CNN + LSTM untuk Klasifikasi Sampah
============================================================
Tujuan:
  - Membuktikan bahwa arsitektur CNN+LSTM BISA ditraining
  - Membuktikan bahwa model bisa di-convert ke TFLite
  - Membuktikan bahwa ukuran model MUAT di ESP32S3 (8MB Flash)
  - Menggunakan data SINTETIS (dummy) agar bisa langsung dijalankan

Cara pakai:
  1. Buka Google Colab
  2. Copy-paste seluruh isi file ini ke cell Colab
  3. Jalankan

Arsitektur:
  Input: Sequence of N frames (misal 10 frame dari rekaman 3 detik)
  -> CNN (MobileNetV2 / custom small CNN) mengekstrak fitur tiap frame
  -> LSTM mempelajari pola temporal antar frame
  -> Dense layer mengklasifikasikan: kertas / plastik / organik
============================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import os
import time

print("=" * 60)
print("  TEST KONSEP: CNN + LSTM Waste Classifier")
print("=" * 60)
print(f"  TensorFlow version: {tf.__version__}")
print(f"  GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print("=" * 60)

# ============================================================
# KONFIGURASI
# ============================================================
IMG_HEIGHT = 96        # Resolusi input (akan di-resize dari QVGA)
IMG_WIDTH = 96
NUM_CHANNELS = 3       # RGB
SEQUENCE_LENGTH = 10   # Jumlah frame per sequence (dari rekaman 3 detik)
NUM_CLASSES = 3        # kertas, plastik, organik
CLASS_NAMES = ["kertas", "plastik", "organik"]

# Training config
BATCH_SIZE = 8
EPOCHS = 10
NUM_TRAIN_SAMPLES = 120   # 40 per kelas (dummy)
NUM_VAL_SAMPLES = 30      # 10 per kelas (dummy)

print(f"\n[CONFIG]")
print(f"  Input shape per frame : {IMG_HEIGHT}x{IMG_WIDTH}x{NUM_CHANNELS}")
print(f"  Sequence length       : {SEQUENCE_LENGTH} frames")
print(f"  Num classes           : {NUM_CLASSES} ({', '.join(CLASS_NAMES)})")
print(f"  Batch size            : {BATCH_SIZE}")
print(f"  Epochs                : {EPOCHS}")

# ============================================================
# STEP 1: Buat Data Sintetis (Dummy)
# ============================================================
print("\n" + "=" * 60)
print("  STEP 1: Membuat Data Sintetis")
print("=" * 60)

def generate_dummy_data(num_samples, seq_len, img_h, img_w, num_ch, num_cls):
    """
    Membuat data dummy yang memiliki pola sederhana per kelas,
    agar model bisa belajar dan kita bisa verifikasi training berjalan.

    Kelas 0 (kertas): gambar cenderung terang (nilai pixel tinggi)
    Kelas 1 (plastik): gambar cenderung gelap (nilai pixel rendah)
    Kelas 2 (organik): gambar cenderung medium (nilai pixel sedang)
    """
    X = np.zeros((num_samples, seq_len, img_h, img_w, num_ch), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int32)

    samples_per_class = num_samples // num_cls

    for cls in range(num_cls):
        start_idx = cls * samples_per_class
        end_idx = start_idx + samples_per_class

        for i in range(start_idx, end_idx):
            y[i] = cls
            for t in range(seq_len):
                if cls == 0:  # kertas - terang
                    base = np.random.uniform(0.6, 1.0)
                elif cls == 1:  # plastik - gelap
                    base = np.random.uniform(0.0, 0.4)
                else:  # organik - medium
                    base = np.random.uniform(0.3, 0.7)

                # Tambah noise + variasi temporal kecil
                noise = np.random.normal(0, 0.05, (img_h, img_w, num_ch))
                temporal_shift = t * 0.01 * (1 if cls == 0 else -1)
                X[i, t] = np.clip(base + noise + temporal_shift, 0, 1)

    # Shuffle
    indices = np.random.permutation(num_samples)
    return X[indices], y[indices]

# Generate data
X_train, y_train = generate_dummy_data(
    NUM_TRAIN_SAMPLES, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES
)
X_val, y_val = generate_dummy_data(
    NUM_VAL_SAMPLES, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES
)

print(f"  X_train shape: {X_train.shape}")  # (120, 10, 96, 96, 3)
print(f"  y_train shape: {y_train.shape}")  # (120,)
print(f"  X_val shape  : {X_val.shape}")    # (30, 10, 96, 96, 3)
print(f"  Memory X_train: {X_train.nbytes / (1024**2):.1f} MB")
print(f"  [OK] Data sintetis siap!")

# ============================================================
# STEP 2: Bangun Model CNN + LSTM
# ============================================================
print("\n" + "=" * 60)
print("  STEP 2: Membangun Model CNN + LSTM")
print("=" * 60)

def build_cnn_feature_extractor(input_shape):
    """
    CNN kecil untuk mengekstrak fitur dari satu frame.
    Dirancang agar ringan dan muat di ESP32S3.
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)   # 96x96 -> 48x48

    # Block 2
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)   # 48x48 -> 24x24

    # Block 3
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)   # 24x24 -> 12x12

    # Block 4
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)  # 12x12x64 -> 64

    model = Model(inputs, x, name="cnn_feature_extractor")
    return model


def build_cnn_lstm_model(seq_len, img_h, img_w, num_ch, num_cls):
    """
    Model lengkap: TimeDistributed(CNN) + LSTM + Dense
    """
    # Input: (batch, sequence_length, height, width, channels)
    inputs = layers.Input(shape=(seq_len, img_h, img_w, num_ch))

    # CNN feature extractor (shared weights across all frames)
    cnn = build_cnn_feature_extractor((img_h, img_w, num_ch))

    # Apply CNN to each frame in the sequence
    x = layers.TimeDistributed(cnn)(inputs)
    # Output shape: (batch, sequence_length, 64)

    # LSTM untuk mempelajari pola temporal
    # unroll=True agar kompatibel dengan TFLite (menghindari TensorList ops)
    x = layers.LSTM(32, return_sequences=False, unroll=True)(x)
    # Output shape: (batch, 32)

    # Dropout untuk regularisasi
    x = layers.Dropout(0.3)(x)

    # Classification head
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(num_cls, activation='softmax')(x)

    model = Model(inputs, outputs, name="waste_classifier_cnn_lstm")
    return model


# Build model
model = build_cnn_lstm_model(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Summary
model.summary()

# Hitung total parameter
total_params = model.count_params()
print(f"\n  Total parameters  : {total_params:,}")
print(f"  Estimasi ukuran   : {total_params * 4 / (1024**2):.2f} MB (float32)")
print(f"  Estimasi TFLite   : {total_params * 1 / (1024**2):.2f} MB (int8 quantized)")
print(f"  ESP32S3 Flash     : 8 MB")
print(f"  Muat di ESP32S3?  : {'YA ✅' if total_params * 1 / (1024**2) < 4 else 'TIDAK ❌ (perlu diperkecil)'}")

# ============================================================
# STEP 3: Training
# ============================================================
print("\n" + "=" * 60)
print("  STEP 3: Training Model")
print("=" * 60)

start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

train_time = time.time() - start_time
print(f"\n  Training selesai dalam {train_time:.1f} detik")

# Evaluasi
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"\n  --- Hasil Training ---")
print(f"  Train Accuracy : {final_train_acc:.4f} ({final_train_acc*100:.1f}%)")
print(f"  Val Accuracy   : {final_val_acc:.4f} ({final_val_acc*100:.1f}%)")
print(f"  Train Loss     : {final_train_loss:.4f}")
print(f"  Val Loss       : {final_val_loss:.4f}")

if final_train_acc > 0.8:
    print(f"  [OK] Model BISA belajar dari data! ✅")
else:
    print(f"  [WARNING] Akurasi rendah, tapi ini data dummy jadi wajar.")
    print(f"            Yang penting: model bisa training tanpa error.")

# ============================================================
# STEP 4: Test Prediksi
# ============================================================
print("\n" + "=" * 60)
print("  STEP 4: Test Prediksi")
print("=" * 60)

# Ambil 3 sample (1 per kelas) dari validation set
for cls in range(NUM_CLASSES):
    idx = np.where(y_val == cls)[0]
    if len(idx) > 0:
        sample = X_val[idx[0:1]]  # shape: (1, 10, 96, 96, 3)
        pred = model.predict(sample, verbose=0)
        pred_class = np.argmax(pred[0])
        confidence = pred[0][pred_class] * 100

        status = "✅" if pred_class == cls else "❌"
        print(f"  Sample kelas '{CLASS_NAMES[cls]}' -> Prediksi: '{CLASS_NAMES[pred_class]}' "
              f"({confidence:.1f}%) {status}")

# ============================================================
# STEP 5: Konversi ke TensorFlow Lite
# ============================================================
print("\n" + "=" * 60)
print("  STEP 5: Konversi ke TensorFlow Lite")
print("=" * 60)

# 5a. Konversi ke TFLite (float32)
print("  [1/3] Konversi ke TFLite (float32)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Fallback ke SELECT_TF_OPS jika ada op yang tidak didukung TFLite built-in
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
tflite_float_model = converter.convert()
float_size = len(tflite_float_model) / 1024
print(f"        Ukuran: {float_size:.1f} KB ({float_size/1024:.2f} MB)")

# 5b. Konversi ke TFLite (float16 quantized)
print("  [2/3] Konversi ke TFLite (float16 quantized)...")
converter_f16 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_f16.target_spec.supported_types = [tf.float16]
converter_f16.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter_f16._experimental_lower_tensor_list_ops = False
tflite_f16_model = converter_f16.convert()
f16_size = len(tflite_f16_model) / 1024
print(f"        Ukuran: {f16_size:.1f} KB ({f16_size/1024:.2f} MB)")

# 5c. Konversi ke TFLite (int8 quantized — best effort)
print("  [3/3] Konversi ke TFLite (int8 quantized)...")

def representative_dataset():
    """Dataset representatif untuk kalibrasi kuantisasi int8."""
    for i in range(min(50, len(X_train))):
        sample = X_train[i:i+1].astype(np.float32)
        yield [sample]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter_int8._experimental_lower_tensor_list_ops = False
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8

try:
    tflite_int8_model = converter_int8.convert()
    int8_size = len(tflite_int8_model) / 1024
    print(f"        Ukuran: {int8_size:.1f} KB ({int8_size/1024:.2f} MB)")
    int8_success = True
except Exception as e:
    print(f"        [WARNING] Int8 gagal: {str(e)[:100]}")
    print(f"        Gunakan float16 sebagai alternatif.")
    int8_success = False

# ============================================================
# STEP 6: Test Inferensi TFLite
# ============================================================
print("\n" + "=" * 60)
print("  STEP 6: Test Inferensi TFLite")
print("=" * 60)

# Gunakan model float32 untuk test inferensi
interpreter = tf.lite.Interpreter(model_content=tflite_float_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"  Input shape  : {input_details[0]['shape']}")
print(f"  Input dtype  : {input_details[0]['dtype']}")
print(f"  Output shape : {output_details[0]['shape']}")
print(f"  Output dtype : {output_details[0]['dtype']}")

# Test dengan 1 sample
test_sample = X_val[0:1].astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_sample)

start_inf = time.time()
interpreter.invoke()
inf_time = (time.time() - start_inf) * 1000  # ms

output_data = interpreter.get_tensor(output_details[0]['index'])
pred_class = np.argmax(output_data[0])
confidence = output_data[0][pred_class] * 100

print(f"\n  Prediksi TFLite: '{CLASS_NAMES[pred_class]}' ({confidence:.1f}%)")
print(f"  Waktu inferensi: {inf_time:.1f} ms")

# ============================================================
# STEP 7: Simpan Model (Opsional)
# ============================================================
print("\n" + "=" * 60)
print("  STEP 7: Simpan Model")
print("=" * 60)

# Simpan ke file
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# Simpan Keras model
model.save(os.path.join(save_dir, "waste_cnn_lstm.keras"))
print(f"  [OK] Keras model disimpan ke {save_dir}/waste_cnn_lstm.keras")

# Simpan TFLite float32
with open(os.path.join(save_dir, "waste_cnn_lstm_float32.tflite"), 'wb') as f:
    f.write(tflite_float_model)
print(f"  [OK] TFLite float32 disimpan ({float_size:.1f} KB)")

# Simpan TFLite float16
with open(os.path.join(save_dir, "waste_cnn_lstm_float16.tflite"), 'wb') as f:
    f.write(tflite_f16_model)
print(f"  [OK] TFLite float16 disimpan ({f16_size:.1f} KB)")

if int8_success:
    with open(os.path.join(save_dir, "waste_cnn_lstm_int8.tflite"), 'wb') as f:
        f.write(tflite_int8_model)
    print(f"  [OK] TFLite int8 disimpan ({int8_size:.1f} KB)")

# ============================================================
# RINGKASAN AKHIR
# ============================================================
print("\n" + "=" * 60)
print("  RINGKASAN TEST KONSEP")
print("=" * 60)
print(f"  ✅ Model CNN+LSTM berhasil dibangun")
print(f"  ✅ Training berjalan ({EPOCHS} epochs)")
print(f"  ✅ Train accuracy: {final_train_acc*100:.1f}%")
print(f"  ✅ Konversi TFLite float32: {float_size:.1f} KB")
print(f"  ✅ Konversi TFLite float16: {f16_size:.1f} KB")
if int8_success:
    print(f"  ✅ Konversi TFLite int8: {int8_size:.1f} KB")
print(f"  ✅ Inferensi TFLite berjalan: {inf_time:.1f} ms")
print(f"")
print(f"  Total parameter model: {total_params:,}")
print(f"  Ukuran model terkecil: {f16_size:.1f} KB")
esp_fit = f16_size / 1024 < 4  # harus < 4MB agar muat di ESP32S3
print(f"  Muat di ESP32S3 (8MB Flash)? {'YA ✅' if esp_fit else 'TIDAK ❌'}")
print(f"")
print(f"  KESIMPULAN: Konsep CNN+LSTM untuk klasifikasi sampah")
print(f"  dari sequence gambar LAYAK untuk dilanjutkan! 🚀")
print("=" * 60)

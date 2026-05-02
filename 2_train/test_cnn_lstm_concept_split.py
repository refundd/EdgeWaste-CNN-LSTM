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
EPOCHS = 10
NUM_TRAIN_SAMPLES = 120
NUM_VAL_SAMPLES = 30

# ============================================================
# STEP 1: Buat Data Sintetis
# ============================================================
def generate_dummy_data(num_samples, seq_len, img_h, img_w, num_ch, num_cls):
    X = np.zeros((num_samples, seq_len, img_h, img_w, num_ch), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int32)
    samples_per_class = num_samples // num_cls
    for cls in range(num_cls):
        start_idx = cls * samples_per_class
        end_idx = start_idx + samples_per_class
        for i in range(start_idx, end_idx):
            y[i] = cls
            for t in range(seq_len):
                if cls == 0: base = np.random.uniform(0.6, 1.0)
                elif cls == 1: base = np.random.uniform(0.0, 0.4)
                else: base = np.random.uniform(0.3, 0.7)
                noise = np.random.normal(0, 0.05, (img_h, img_w, num_ch))
                temporal_shift = t * 0.01 * (1 if cls == 0 else -1)
                X[i, t] = np.clip(base + noise + temporal_shift, 0, 1)
    indices = np.random.permutation(num_samples)
    return X[indices], y[indices]

X_train, y_train = generate_dummy_data(NUM_TRAIN_SAMPLES, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES)
X_val, y_val = generate_dummy_data(NUM_VAL_SAMPLES, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES)

# ============================================================
# STEP 2: Bangun Model End-to-End
# ============================================================
def build_cnn_feature_extractor(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Gunakan MobileNetV2 dengan alpha=0.35 ala Edge Impulse
    # weights=None karena kita men-training dari awal (dummy test), 
    # aslinya nanti pakai weights='imagenet'
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=0.35,
        include_top=False,
        weights=None
    )
    
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
combined_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ============================================================
# STEP 3: Training
# ============================================================
print("\n[INFO] Training End-to-End Model...")
combined_model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

# ============================================================
# STEP 4: Split Model 
# ============================================================
print("\n[INFO] Split Model Selesai secara otomatis karena komposisi Keras!")
# Model cnn_extractor dan temporal_head sudah terupdate bobotnya.

# ============================================================
# STEP 5: Konversi TFLite
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

print(f"  [OK] CNN model: {len(tflite_cnn)/1024:.1f} KB")
print(f"  [OK] Temporal Head model: {len(tflite_temporal)/1024:.1f} KB")

# Export to C Header
os.system(f"xxd -i {os.path.join(save_dir, 'waste_cnn_int8.tflite')} > {os.path.join(save_dir, 'model_cnn_data.h')}")
os.system(f"xxd -i {os.path.join(save_dir, 'waste_lstm_int8.tflite')} > {os.path.join(save_dir, 'model_lstm_data.h')}")

print("\n[OK] Selesai! Model siap digunakan di ESP32.")

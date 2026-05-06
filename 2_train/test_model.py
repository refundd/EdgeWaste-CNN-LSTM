"""Quick test: jalankan model pada validation set dan cek apakah prediksi benar."""
import numpy as np
import tensorflow as tf

# Load TFLite models
cnn_interp = tf.lite.Interpreter(model_path="saved_models_split/waste_cnn_int8.tflite")
cnn_interp.allocate_tensors()
lstm_interp = tf.lite.Interpreter(model_path="saved_models_split/waste_lstm_int8.tflite")
lstm_interp.allocate_tensors()

cnn_in = cnn_interp.get_input_details()[0]
cnn_out = cnn_interp.get_output_details()[0]
lstm_in = lstm_interp.get_input_details()[0]
lstm_out = lstm_interp.get_output_details()[0]

print("=== CNN Model ===")
print(f"  Input: shape={cnn_in['shape']}, dtype={cnn_in['dtype']}, scale={cnn_in['quantization'][0]:.6f}, zp={cnn_in['quantization'][1]}")
print(f"  Output: shape={cnn_out['shape']}, dtype={cnn_out['dtype']}, scale={cnn_out['quantization'][0]:.6f}, zp={cnn_out['quantization'][1]}")

print("\n=== LSTM/Temporal Model ===")
print(f"  Input: shape={lstm_in['shape']}, dtype={lstm_in['dtype']}, scale={lstm_in['quantization'][0]:.6f}, zp={lstm_in['quantization'][1]}")
print(f"  Output: shape={lstm_out['shape']}, dtype={lstm_out['dtype']}, scale={lstm_out['quantization'][0]:.6f}, zp={lstm_out['quantization'][1]}")

CLASS_NAMES = ["kertas", "plastik", "organik"]
import os

# Load some samples from each class
for cls_idx, cls in enumerate(CLASS_NAMES):
    cls_dir = os.path.join("dataset", cls)
    if not os.path.exists(cls_dir):
        continue
    sessions = sorted([d for d in os.listdir(cls_dir) if d.startswith('sesi_')])[:3]
    
    print(f"\n=== Testing class: {cls} ({len(sessions)} sessions) ===")
    
    for session in sessions:
        sess_dir = os.path.join(cls_dir, session)
        frames = sorted([f for f in os.listdir(sess_dir) if f.endswith('.jpg')])
        if len(frames) < 3:
            continue
        
        indices = np.linspace(0, len(frames)-1, 3, dtype=int)
        
        # Run CNN on each frame
        cnn_features = np.zeros((1, 3, 64), dtype=np.int8)
        
        for t, idx in enumerate(indices):
            img_path = os.path.join(sess_dir, frames[idx])
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(48, 48))
            img_arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            
            # Quantize input
            img_q = np.clip(img_arr / cnn_in['quantization'][0] + cnn_in['quantization'][1], -128, 127).astype(np.int8)
            img_q = np.expand_dims(img_q, 0)
            
            cnn_interp.set_tensor(cnn_in['index'], img_q)
            cnn_interp.invoke()
            cnn_output = cnn_interp.get_tensor(cnn_out['index'])[0]
            
            # Requantize for LSTM input
            cnn_scale = cnn_out['quantization'][0]
            cnn_zp = cnn_out['quantization'][1]
            lstm_scale = lstm_in['quantization'][0]
            lstm_zp = lstm_in['quantization'][1]
            
            for i in range(64):
                real_val = (float(cnn_output[i]) - cnn_zp) * cnn_scale
                q_val = int(round(real_val / lstm_scale + lstm_zp))
                cnn_features[0, t, i] = np.clip(q_val, -128, 127)
        
        # Run LSTM
        lstm_interp.set_tensor(lstm_in['index'], cnn_features)
        lstm_interp.invoke()
        output = lstm_interp.get_tensor(lstm_out['index'])[0]
        
        # Print raw INT8 output and prediction
        pred = np.argmax(output)
        print(f"  {session}: raw=[{output[0]:4d}, {output[1]:4d}, {output[2]:4d}] -> {CLASS_NAMES[pred]} (correct={'✓' if pred == cls_idx else '✗'})")

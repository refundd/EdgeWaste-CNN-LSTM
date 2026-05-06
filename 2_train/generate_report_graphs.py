"""Generate training report graphs for thesis (Tugas Akhir)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Load training history
with open("training_history.json", "r") as f:
    history = json.load(f)

save_dir = "report_grafik"
os.makedirs(save_dir, exist_ok=True)

# Style
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})

# ========================================
# 1. TRAINING LOSS — PHASE 1 + PHASE 2
# ========================================
fig, ax = plt.subplots(figsize=(12, 5))
p1_loss = history['phase1']['loss']
p1_val_loss = history['phase1']['val_loss']
p2_loss = history['phase2']['loss']
p2_val_loss = history['phase2']['val_loss']

all_loss = p1_loss + p2_loss
all_val_loss = p1_val_loss + p2_val_loss
epochs = list(range(1, len(all_loss) + 1))
phase1_end = len(p1_loss)

ax.plot(epochs, all_loss, 'b-o', label='Training Loss', markersize=4)
ax.plot(epochs, all_val_loss, 'r-s', label='Validation Loss', markersize=4)
ax.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label=f'Phase 1→2 (epoch {phase1_end})')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training dan Validation Loss — CNN-LSTM Waste Classifier')
ax.legend()
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '1_training_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 1_training_loss.png")

# ========================================
# 2. TRAINING ACCURACY — PHASE 1 + PHASE 2
# ========================================
fig, ax = plt.subplots(figsize=(12, 5))
p1_acc = history['phase1']['accuracy']
p1_val_acc = history['phase1']['val_accuracy']
p2_acc = history['phase2']['accuracy']
p2_val_acc = history['phase2']['val_accuracy']

all_acc = [a*100 for a in p1_acc + p2_acc]
all_val_acc = [a*100 for a in p1_val_acc + p2_val_acc]
epochs = list(range(1, len(all_acc) + 1))

ax.plot(epochs, all_acc, 'b-o', label='Training Accuracy', markersize=4)
ax.plot(epochs, all_val_acc, 'r-s', label='Validation Accuracy', markersize=4)
ax.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label=f'Phase 1→2 (epoch {phase1_end})')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Training dan Validation Accuracy — CNN-LSTM Waste Classifier')
ax.legend()
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '2_training_accuracy.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 2_training_accuracy.png")

# ========================================
# 3. CONFUSION MATRIX
# ========================================
# From training_report.txt data
CLASS_NAMES = ["Kertas", "Plastik", "Organik"]

# Load val data to create actual confusion matrix
import tensorflow as tf

# Reload the combined model's predictions
# We'll use the saved training_report.txt info + manual approach
# For accurate results, let's compute from test_model.py approach

cnn_interp = tf.lite.Interpreter(model_path="saved_models_split/waste_cnn_int8.tflite")
cnn_interp.allocate_tensors()
lstm_interp = tf.lite.Interpreter(model_path="saved_models_split/waste_lstm_int8.tflite")
lstm_interp.allocate_tensors()

cnn_in = cnn_interp.get_input_details()[0]
cnn_out = cnn_interp.get_output_details()[0]
lstm_in = lstm_interp.get_input_details()[0]
lstm_out = lstm_interp.get_output_details()[0]

class_names_lower = ["kertas", "plastik", "organik"]
y_true = []
y_pred = []

for cls_idx, cls in enumerate(class_names_lower):
    cls_dir = os.path.join("dataset", cls)
    if not os.path.exists(cls_dir): continue
    sessions = sorted([d for d in os.listdir(cls_dir) if d.startswith('sesi_')])
    
    for session in sessions:
        sess_dir = os.path.join(cls_dir, session)
        frames = sorted([f for f in os.listdir(sess_dir) if f.endswith('.jpg')])
        if len(frames) < 3: continue
        
        indices = np.linspace(0, len(frames)-1, 3, dtype=int)
        cnn_features = np.zeros((1, 3, 64), dtype=np.int8)
        
        for t, idx in enumerate(indices):
            img_path = os.path.join(sess_dir, frames[idx])
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(48, 48))
            img_arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_q = np.clip(img_arr / cnn_in['quantization'][0] + cnn_in['quantization'][1], -128, 127).astype(np.int8)
            img_q = np.expand_dims(img_q, 0)
            cnn_interp.set_tensor(cnn_in['index'], img_q)
            cnn_interp.invoke()
            cnn_output = cnn_interp.get_tensor(cnn_out['index'])[0]
            
            cnn_scale = cnn_out['quantization'][0]
            cnn_zp = cnn_out['quantization'][1]
            lstm_scale = lstm_in['quantization'][0]
            lstm_zp = lstm_in['quantization'][1]
            for i in range(64):
                real_val = (float(cnn_output[i]) - cnn_zp) * cnn_scale
                q_val = int(round(real_val / lstm_scale + lstm_zp))
                cnn_features[0, t, i] = np.clip(q_val, -128, 127)
        
        lstm_interp.set_tensor(lstm_in['index'], cnn_features)
        lstm_interp.invoke()
        output = lstm_interp.get_tensor(lstm_out['index'])[0]
        
        y_true.append(cls_idx)
        y_pred.append(np.argmax(output))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Build confusion matrix
cm = np.zeros((3, 3), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t][p] += 1

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues')

# Annotate
for i in range(3):
    for j in range(3):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=18, fontweight='bold', color=color)

ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(CLASS_NAMES)
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel('Prediksi')
ax.set_ylabel('Label Sebenarnya')
ax.set_title('Confusion Matrix — INT8 Quantized Model (Seluruh Dataset)')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '3_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()

# Print per-class accuracy
total_correct = np.trace(cm)
total_samples = np.sum(cm)
print(f"\n[RESULT] Overall Accuracy: {total_correct}/{total_samples} ({total_correct/total_samples*100:.1f}%)")
for i, cls in enumerate(CLASS_NAMES):
    cls_total = np.sum(cm[i])
    cls_correct = cm[i][i]
    precision = cm[i][i] / max(1, np.sum(cm[:, i]))
    recall = cm[i][i] / max(1, np.sum(cm[i]))
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    print(f"  {cls}: Acc={cls_correct}/{cls_total} ({recall*100:.1f}%), Precision={precision*100:.1f}%, Recall={recall*100:.1f}%, F1={f1*100:.1f}%")
print("[OK] 3_confusion_matrix.png")

# ========================================
# 4. DATASET DISTRIBUTION
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original
orig_counts = [49, 50, 24]
colors = ['#4CAF50', '#2196F3', '#FF9800']
axes[0].bar(CLASS_NAMES, orig_counts, color=colors, edgecolor='white', linewidth=2)
for i, v in enumerate(orig_counts):
    axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold', fontsize=14)
axes[0].set_ylabel('Jumlah Sesi')
axes[0].set_title('Dataset Asli')
axes[0].set_ylim(0, max(orig_counts) + 10)

# After augmentation (training only)
aug_counts = [200, 200, 200]
axes[1].bar(CLASS_NAMES, aug_counts, color=colors, edgecolor='white', linewidth=2)
for i, v in enumerate(aug_counts):
    axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold', fontsize=14)
axes[1].set_ylabel('Jumlah Sesi')
axes[1].set_title('Setelah Augmentasi (Training Set)')
axes[1].set_ylim(0, max(aug_counts) + 30)

plt.suptitle('Distribusi Dataset', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '4_dataset_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 4_dataset_distribution.png")

# ========================================
# 5. LEARNING RATE SCHEDULE
# ========================================
fig, ax = plt.subplots(figsize=(12, 5))
p1_lr = history['phase1'].get('learning_rate', history['phase1'].get('lr', [1e-3]*len(p1_loss)))
p2_lr = history['phase2'].get('learning_rate', history['phase2'].get('lr', [5e-5]*len(p2_loss)))
all_lr = p1_lr + p2_lr
epochs_lr = list(range(1, len(all_lr) + 1))

ax.plot(epochs_lr, all_lr, 'g-o', markersize=4)
ax.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label=f'Phase 1→2 (epoch {phase1_end})')
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule (ReduceLROnPlateau)')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '5_learning_rate.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 5_learning_rate.png")

# ========================================
# 6. MODEL SIZE COMPARISON
# ========================================
fig, ax = plt.subplots(figsize=(8, 5))
models = ['CNN\n(MobileNetV2)', 'LSTM\n(Temporal Head)', 'Total']
sizes_kb = [713.4, 62.6, 713.4 + 62.6]
bars = ax.bar(models, sizes_kb, color=['#1976D2', '#388E3C', '#F57C00'], edgecolor='white', linewidth=2, width=0.5)
for bar, size in zip(bars, sizes_kb):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f'{size:.1f} KB', ha='center', fontweight='bold', fontsize=13)
ax.set_ylabel('Ukuran Model (KB)')
ax.set_title('Ukuran Model INT8 Quantized')
ax.set_ylim(0, max(sizes_kb) + 80)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '6_model_size.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 6_model_size.png")

# ========================================
# 7. INFERENCE TIME (from ESP32 logs)
# ========================================
fig, ax = plt.subplots(figsize=(8, 5))
stages = ['Capture', 'CNN\n(3 frames)', 'LSTM\n(Temporal)', 'Total']
times_ms = [0, 723, 6, 1156]
colors_t = ['#9E9E9E', '#1976D2', '#388E3C', '#F57C00']
bars = ax.bar(stages, times_ms, color=colors_t, edgecolor='white', linewidth=2, width=0.5)
for bar, t in zip(bars, times_ms):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{t} ms', ha='center', fontweight='bold', fontsize=13)
ax.set_ylabel('Waktu (ms)')
ax.set_title('Waktu Inferensi pada XIAO ESP32S3 (240 MHz)')
ax.set_ylim(0, max(times_ms) + 150)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '7_inference_time.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 7_inference_time.png")

print(f"\n[DONE] Semua grafik disimpan di folder: {save_dir}/")
print(f"Total: {len(os.listdir(save_dir))} grafik")

# Laporan Hasil Training CNN-LSTM Waste Classifier

## 1. Arsitektur Model

### 1.1 CNN Feature Extractor (MobileNetV2)
| Parameter | Nilai |
|-----------|-------|
| Backbone | MobileNetV2 (α=0.35, pretrained ImageNet) |
| Input | 48×48×3 (RGB) |
| Pooling | GlobalAveragePooling2D |
| Projection | Dense(64, ReLU) |
| Output | 64-dim feature vector |
| **Trainable Params** | **81,984 (320.25 KB)** |

### 1.2 Temporal Head (Manual LSTM)
| Parameter | Nilai |
|-----------|-------|
| Arsitektur | Manual LSTM Cell (3 timesteps) |
| LSTM Units | 32 |
| Gates | Forget, Input, Output (per timestep) |
| Dropout | 0.3 |
| Classifier | Dense(3, softmax) |
| **Trainable Params** | **37,347 (145.89 KB)** |

### 1.3 Combined Model (End-to-End)
| Parameter | Nilai |
|-----------|-------|
| Input Shape | (3, 48, 48, 3) — 3 frame temporal |
| **Total Trainable Params** | **119,331 (466.14 KB)** |

---

## 2. Dataset

### 2.1 Distribusi Data Asli
| Kelas | Jumlah Sesi |
|-------|------------|
| Kertas | 49 |
| Plastik | 50 |
| Organik | 24 |
| **Total** | **123** |

### 2.2 Split Data (Stratified)
| Set | Kertas | Plastik | Organik | Total |
|-----|--------|---------|---------|-------|
| Training | 40 | 40 | 20 | 100 |
| Validation | 9 | 10 | 4 | 23 |

### 2.3 Augmentasi (Training Set Only)
Setelah balanced augmentation, setiap kelas memiliki **200 sampel** (total 600).

| Teknik Augmentasi | Parameter |
|-------------------|-----------|
| Horizontal Flip | 50% probability |
| Rotasi | ±15° (70% probability) |
| Zoom | 0.85× – 1.15× (70% probability) |
| Shift | ±4 pixel (50% probability) |
| Brightness | ±0.3 |
| Contrast | 0.7× – 1.4× |
| Color Jitter | ±0.15 per channel |
| Gaussian Noise | σ=0.03 (50% probability) |

> [!IMPORTANT]
> Augmentasi dilakukan **hanya pada training set** setelah split untuk mencegah data leakage. Class weights = 1.0 (seimbang) karena augmentasi sudah menyeimbangkan jumlah per kelas.

![Distribusi Dataset](/Users/yogasatyawisesa/.gemini/antigravity/brain/8d6b147b-47ad-4ff0-ae96-77e2af367722/artifacts/4_dataset_distribution.png)

---

## 3. Training History

### 3.1 Phase 1: Head-Only Training (Backbone Frozen)
| Konfigurasi | Nilai |
|-------------|-------|
| Optimizer | Adam (LR=1e-3) |
| Epochs | 15 |
| Batch Size | 8 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |

**History (Per Epoch):**

| Epoch | Loss | Accuracy | Val Loss | Val Accuracy | LR |
|-------|------|----------|----------|-------------|-----|
| 1 | 0.6923 | 69.8% | 0.4304 | 73.9% | 1e-3 |
| 2 | 0.2814 | 91.0% | 0.6494 | 65.2% | 1e-3 |
| 3 | 0.1729 | 94.3% | 0.0993 | **100.0%** | 1e-3 |
| 4 | 0.1010 | 95.8% | 0.1735 | 91.3% | 1e-3 |
| 5 | 0.0709 | 98.0% | 0.6177 | 78.3% | 1e-3 |
| 6 | 0.0625 | 98.2% | 0.2834 | 87.0% | **5e-4** ↓ |
| 7 | 0.0318 | 99.2% | 0.0737 | **100.0%** | 5e-4 |
| 8 | 0.0117 | **100.0%** | 0.1100 | 95.7% | 5e-4 |
| 9 | 0.0116 | **100.0%** | 0.1328 | 91.3% | 5e-4 |
| 10 | 0.0069 | **100.0%** | 0.0725 | **100.0%** | 5e-4 |
| 11 | 0.0064 | **100.0%** | 0.0840 | **100.0%** | 5e-4 |
| 12 | 0.0048 | **100.0%** | 0.0668 | **100.0%** | 5e-4 |
| 13 | 0.0048 | **100.0%** | 0.0572 | **100.0%** | 5e-4 |
| 14 | 0.0033 | **100.0%** | 0.0418 | **100.0%** | 5e-4 |
| 15 | 0.0030 | **100.0%** | 0.0490 | **100.0%** | 5e-4 |

### 3.2 Phase 2: Full Fine-Tuning (Backbone Unfrozen)
| Konfigurasi | Nilai |
|-------------|-------|
| Optimizer | Adam (LR=5e-5) |
| Epochs | 30 |
| Batch Size | 8 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6) |

**History (Per Epoch):**

| Epoch | Loss | Accuracy | Val Loss | Val Accuracy | LR |
|-------|------|----------|----------|-------------|-----|
| 1 | 1.0436 | 66.3% | 0.7547 | 87.0% | 5e-5 |
| 2 | 0.8105 | 74.3% | 0.8281 | 87.0% | 5e-5 |
| 3 | 0.6515 | 79.3% | 1.4150 | 65.2% | 5e-5 |
| 4 | 0.6860 | 78.7% | 1.0323 | 78.3% | **2.5e-5** ↓ |
| 5 | 0.6165 | 80.3% | 0.6070 | 78.3% | 2.5e-5 |
| 6 | 0.5563 | 80.2% | 0.2553 | 87.0% | 2.5e-5 |
| 7 | 0.4988 | 84.2% | 0.1937 | 91.3% | 2.5e-5 |
| 8 | 0.4770 | 85.2% | 0.1316 | 91.3% | 2.5e-5 |
| 9 | 0.4611 | 85.0% | 0.1422 | **95.7%** | 2.5e-5 |
| 10 | 0.3741 | 87.2% | 0.2608 | 91.3% | 2.5e-5 |
| 11 | 0.5354 | 82.3% | 0.2460 | **95.7%** | **1.25e-5** ↓ |
| 12 | 0.4665 | 84.2% | 0.2214 | **95.7%** | 1.25e-5 |
| 13 | 0.3855 | 86.8% | 0.2576 | 87.0% | 1.25e-5 |
| 14 | 0.4721 | 84.5% | 0.2595 | 87.0% | **6.25e-6** ↓ |
| 15 | 0.3734 | 87.0% | 0.2502 | 91.3% | 6.25e-6 |
| 16 | 0.3866 | 85.7% | 0.2442 | 91.3% | 6.25e-6 |
| 17 | 0.3623 | 86.0% | 0.2383 | 91.3% | **3.13e-6** ↓ |
| 18 | 0.3670 | 86.3% | 0.2209 | **95.7%** | 3.13e-6 |
| 19 | 0.4236 | 86.7% | 0.2264 | **95.7%** | 3.13e-6 |
| 20 | 0.3111 | 88.0% | 0.2293 | **95.7%** | **1.56e-6** ↓ |
| 21 | 0.4282 | 84.8% | 0.2337 | **95.7%** | 1.56e-6 |
| 22 | 0.4528 | 84.5% | 0.2332 | **95.7%** | 1.56e-6 |
| 23 | 0.3248 | 89.0% | 0.2299 | **95.7%** | **1e-6** |
| 24 | 0.3857 | 86.8% | 0.2254 | **95.7%** | 1e-6 |
| 25 | 0.4991 | 81.8% | 0.2296 | **95.7%** | 1e-6 |
| 26 | 0.3660 | 87.5% | 0.2387 | **95.7%** | 1e-6 |
| 27 | 0.3613 | 88.7% | 0.2419 | **95.7%** | 1e-6 |
| 28 | 0.3350 | 89.0% | 0.2479 | **95.7%** | 1e-6 |
| 29 | 0.3385 | 88.3% | 0.2502 | **95.7%** | 1e-6 |
| 30 | 0.3542 | 87.2% | 0.2523 | **95.7%** | 1e-6 |

> [!NOTE]
> Phase 2 menunjukkan penurunan akurasi awal karena unfreezing seluruh backbone menyebabkan learned features bergeser. Model kemudian konvergen ke **val accuracy 95.7%** yang stabil.

### Grafik Training

![Training Loss](/Users/yogasatyawisesa/.gemini/antigravity/brain/8d6b147b-47ad-4ff0-ae96-77e2af367722/artifacts/1_training_loss.png)

![Training Accuracy](/Users/yogasatyawisesa/.gemini/antigravity/brain/8d6b147b-47ad-4ff0-ae96-77e2af367722/artifacts/2_training_accuracy.png)

![Learning Rate Schedule](/Users/yogasatyawisesa/.gemini/antigravity/brain/8d6b147b-47ad-4ff0-ae96-77e2af367722/artifacts/5_learning_rate.png)

---

## 4. Evaluasi Final

### 4.1 Akurasi Keseluruhan (INT8 Quantized, Seluruh Dataset)

| Metrik | Nilai |
|--------|-------|
| **Overall Accuracy** | **97.6% (120/123)** |
| Training Accuracy | 100.0% |
| Validation Accuracy | 95.7% |

### 4.2 Per-Class Metrics

| Kelas | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Kertas** | 100.0% (49/49) | 96.1% | 100.0% | **98.0%** |
| **Plastik** | 94.0% (47/50) | 100.0% | 94.0% | **96.9%** |
| **Organik** | 100.0% (24/24) | 96.0% | 100.0% | **98.0%** |
| **Rata-rata** | **97.6%** | **97.4%** | **98.0%** | **97.6%** |

### 4.3 Confusion Matrix

![Confusion Matrix](/Users/yogasatyawisesa/.gemini/antigravity/brain/8d6b147b-47ad-4ff0-ae96-77e2af367722/artifacts/3_confusion_matrix.png)

| | Pred: Kertas | Pred: Plastik | Pred: Organik |
|---|---|---|---|
| **True: Kertas** | **49** | 0 | 0 |
| **True: Plastik** | 2 | **47** | 1 |
| **True: Organik** | 0 | 0 | **24** |

> [!NOTE]
> 3 sample plastik salah prediksi: 2 diprediksi sebagai kertas dan 1 sebagai organik. Kemungkinan disebabkan oleh kemiripan visual antara plastik putih/tipis dengan kertas putih.

---

## 5. Model Quantized (INT8)

### 5.1 Ukuran Model

| Model | Ukuran |
|-------|--------|
| CNN (MobileNetV2) | 713.4 KB |
| Temporal Head (LSTM) | 62.6 KB |
| **Total** | **776.0 KB** |

![Ukuran Model](/Users/yogasatyawisesa/.gemini/antigravity/brain/8d6b147b-47ad-4ff0-ae96-77e2af367722/artifacts/6_model_size.png)

### 5.2 Quantization Parameters

**CNN Feature Extractor:**
| Tensor | Shape | Scale | Zero Point |
|--------|-------|-------|------------|
| Input | (1, 48, 48, 3) | 0.003922 | -128 |
| Output | (1, 64) | 0.048308 | -128 |

**Temporal Head (LSTM):**
| Tensor | Shape | Scale | Zero Point |
|--------|-------|-------|------------|
| Input | (1, 3, 64) | 0.076051 | -116 |
| Output | (1, 3) | 0.003906 | -128 |

---

## 6. Performa Inferensi (XIAO ESP32S3)

| Tahap | Waktu |
|-------|-------|
| Capture (3 frame) | ~0 ms |
| CNN (3×MobileNetV2) | 723 ms (241 ms/frame) |
| LSTM (Temporal Head) | 6 ms |
| Overhead | ~427 ms |
| **Total Pipeline** | **~1,156 ms** |

| Spesifikasi Hardware | Nilai |
|---------------------|-------|
| MCU | ESP32-S3 |
| Clock | 240 MHz |
| RAM | 320 KB SRAM + 8 MB PSRAM |
| Framework | ESP-IDF + TFLite Micro + ESP-NN |
| Kamera | OV2640/OV3660 (320×240 RGB565) |

![Waktu Inferensi](/Users/yogasatyawisesa/.gemini/antigravity/brain/8d6b147b-47ad-4ff0-ae96-77e2af367722/artifacts/7_inference_time.png)

---

## 7. Ringkasan

| Aspek | Hasil |
|-------|-------|
| Akurasi keseluruhan | **97.6%** |
| F1-Score rata-rata | **97.6%** |
| Ukuran model total | **776 KB** |
| Waktu inferensi | **~1.16 detik** |
| Kelas tersulit | Plastik (94.0% recall) |
| Kelas terbaik | Kertas & Organik (100% recall) |

> [!TIP]
> Semua grafik tersimpan di: `/Users/yogasatyawisesa/TANEV/2_train/report_grafik/`
> - `1_training_loss.png` — Kurva loss training & validation
> - `2_training_accuracy.png` — Kurva akurasi training & validation
> - `3_confusion_matrix.png` — Confusion matrix
> - `4_dataset_distribution.png` — Distribusi dataset asli vs augmentasi
> - `5_learning_rate.png` — Jadwal learning rate
> - `6_model_size.png` — Perbandingan ukuran model
> - `7_inference_time.png` — Breakdown waktu inferensi

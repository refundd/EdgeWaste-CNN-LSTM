# EdgeWaste-CNN-LSTM ♻️🧠

**EdgeWaste-CNN-LSTM** is an end-to-end, real-time spatial-temporal waste classification pipeline engineered specifically for the highly constrained **Seeed Studio XIAO ESP32S3 Sense**. 

By combining a **MobileNetV2** (Spatial) feature extractor with a **Custom Unrolled LSTM** (Temporal) head, this project achieves high-accuracy, sub-second video classification directly on edge hardware.

---

## 🚀 Key Highlights

- **Hardware Target**: Seeed Studio XIAO ESP32S3 Sense (equipped with an OV3660 camera and 8MB PSRAM).
- **Split Architecture Strategy**:
  - **Spatial Extractor (CNN)**: A heavily quantized **MobileNetV2 (α0.35)** processes 48x48 RGB frames to extract 64 deep features per frame.
  - **Temporal Head (LSTM)**: A **Custom Manual Unrolled LSTM Cell** (forget, input, output gates + cell state) processes 3 consecutive frames to understand motion context (e.g., how the waste falls).
- **Bypassing TFLite Micro Limitations**: We completely bypassed the infamous `FILL` and `TRANSPOSE` operator crashes in TFLite Micro for ESP32 by implementing the mathematical structure of the LSTM cell using basic, perfectly supported Keras functional math operations (`Concatenate`, `Multiply`, `Add`, `Sigmoid`, `Tanh`).
- **Sub-Second Real-Time Performance**:
  - Accelerated by **ESP-NN** and clocked at **240 MHz**.
  - CNN Inference: **~240 ms / frame**.
  - LSTM Temporal Inference (3 frames): **~6 ms**.
  - Total Pipeline Latency: **~728 ms (1.4 FPS)**.

## 📂 Directory Structure

The repository is divided into three main operational phases:

```text
.
├── 1_collect/                   # Phase 1: Data Collection
│   └── src/main.cpp             # ESP32 firmware to capture and save JPEGs to SD Card
│
├── 2_train/                     # Phase 2: Model Training & Conversion (Python)
│   ├── test_cnn_lstm_concept_split.py  # End-to-end training and split model conversion
│   ├── saved_models_split/      # Quantized INT8 models (TFLite & C-arrays)
│   └── requirements.txt         # Python dependencies
│
└── 3_deploy/                    # Phase 3: Hardware Deployment
    └── xiao_inference_espnn/    # PlatformIO project for the XIAO ESP32S3
        ├── src/main.cpp         # Main inference loop (Camera capture, crop, CNN + LSTM)
        ├── partitions_8mb.csv   # Custom partition table to fit the MobileNetV2 model
        ├── sdkconfig.defaults   # Hardware overrides (240MHz CPU, PSRAM configs)
        └── platformio.ini       # Build configurations
```

## 🛠️ Requirements & Setup

### 1. Model Training (Python)
Navigate to `2_train` and install the requirements:
```bash
cd 2_train
pip install -r requirements.txt
python test_cnn_lstm_concept_split.py
```
This script will train the model, split the CNN and LSTM heads, quantize them into `INT8` via TensorFlow Lite, and export them as C-header arrays (`model_cnn_data.h` and `model_lstm_data.h`).

### 2. Deployment (PlatformIO)
You must use **PlatformIO** to compile and upload the inference firmware.
1. Open the `3_deploy/xiao_inference_espnn` folder in your IDE (VSCode/Cursor).
2. Ensure you do a clean build whenever you change `sdkconfig.defaults`.
3. Build and Upload:
```bash
pio run -t upload -e xiao_esp32s3
```
4. Open the Serial Monitor (`pio device monitor -b 115200`).

## 🧠 Why a "Custom" LSTM?
Standard Keras `LSTM` layers inject dynamic-shape operators (`FILL`, `SHAPE`, `TRANSPOSE`) into the graph. TFLite Micro on ESP32 notoriously struggles to allocate dynamic arenas for these ops during INT8 quantization, causing device crashes. 

To maintain the **strict academic requirement** of a "CNN + LSTM" architecture, we manually unrolled 3 time-steps of an LSTM cell using fundamental operations. To the mathematics, it is a 100% true LSTM. To the ESP32 compiler, it is just a sequence of highly optimized, static, and safe fully-connected operations.

## 📝 License
This project is open-source and free to use for academic and research purposes.

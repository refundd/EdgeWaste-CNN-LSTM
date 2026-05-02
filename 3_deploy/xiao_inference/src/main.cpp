#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_cnn_data.h"
#include "model_lstm_data.h"

// === KONFIGURASI ===
#define NUM_FRAMES    3   // Jumlah frame sequence (awal, tengah, akhir)
#define NUM_FEATURES  64  // Output fitur CNN per frame
#define NUM_CLASSES   3
#define IMG_W         48
#define IMG_H         48
#define IMG_CH        3
#define IMG_SIZE      (IMG_W * IMG_H * IMG_CH)  // 6912 bytes

const char* CLASS_NAMES[] = {"kertas", "plastik", "organik"};

// --- CNN GLOBALS ---
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* cnn_model = nullptr;
tflite::MicroInterpreter* cnn_interpreter = nullptr;
TfLiteTensor* cnn_input = nullptr;
TfLiteTensor* cnn_output = nullptr;

// --- LSTM GLOBALS ---
const tflite::Model* lstm_model = nullptr;
tflite::MicroInterpreter* lstm_interpreter = nullptr;
TfLiteTensor* lstm_input = nullptr;
TfLiteTensor* lstm_output = nullptr;

// Tensor Arenas di PSRAM
const int cnn_arena_size = 3 * 1024 * 1024; // 3 MB
uint8_t* cnn_arena = nullptr;

const int lstm_arena_size = 1 * 1024 * 1024; // 1 MB
uint8_t* lstm_arena = nullptr;

void setup() {
  Serial.begin(115200);
  delay(3000);

  Serial.println("\n=============================================");
  Serial.println("  TEST INFERENSI SPLIT CNN-LSTM DI ESP32S3");
  Serial.println("  MobileNetV1 α0.25 | 48x48 RGB | 3 Frames");
  Serial.println("=============================================");

  // --- PSRAM CHECK ---
  if (psramFound()) {
    Serial.printf("[OK] PSRAM terdeteksi: %.1f MB\n", ESP.getPsramSize() / (1024.0 * 1024.0));
  } else {
    Serial.println("[ERROR] PSRAM tidak terdeteksi!");
    while(1);
  }

  cnn_arena = (uint8_t*)ps_malloc(cnn_arena_size);
  lstm_arena = (uint8_t*)ps_malloc(lstm_arena_size);
  
  if (!cnn_arena || !lstm_arena) {
    Serial.println("[ERROR] Gagal alokasi PSRAM arena!");
    while(1);
  }
  Serial.printf("[OK] PSRAM arena dialokasikan: CNN=%dKB, LSTM=%dKB\n",
                cnn_arena_size/1024, lstm_arena_size/1024);

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  tflite::InitializeTarget();

  // --- SETUP CNN ---
  cnn_model = tflite::GetModel(saved_models_split_waste_cnn_int8_tflite);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_cnn_interpreter(
      cnn_model, resolver, cnn_arena, cnn_arena_size, error_reporter);
  cnn_interpreter = &static_cnn_interpreter;
  if (cnn_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[ERROR] CNN AllocateTensors() gagal!");
    while(1);
  }
  cnn_input = cnn_interpreter->input(0);
  cnn_output = cnn_interpreter->output(0);
  Serial.printf("[OK] CNN siap. Input: %d bytes, Arena: %d bytes\n",
                cnn_input->bytes, cnn_interpreter->arena_used_bytes());

  // --- SETUP LSTM ---
  lstm_model = tflite::GetModel(saved_models_split_waste_lstm_int8_tflite);
  static tflite::MicroInterpreter static_lstm_interpreter(
      lstm_model, resolver, lstm_arena, lstm_arena_size, error_reporter);
  lstm_interpreter = &static_lstm_interpreter;
  if (lstm_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[ERROR] LSTM AllocateTensors() gagal!");
    while(1);
  }
  lstm_input = lstm_interpreter->input(0);
  lstm_output = lstm_interpreter->output(0);
  Serial.printf("[OK] LSTM siap. Input: %d bytes, Arena: %d bytes\n",
                lstm_input->bytes, lstm_interpreter->arena_used_bytes());

  // --- RINGKASAN MEMORI ---
  Serial.println("\n--- RINGKASAN MEMORI ---");
  Serial.printf("  PSRAM Total  : %.1f MB\n", ESP.getPsramSize() / (1024.0*1024.0));
  Serial.printf("  PSRAM Sisa   : %.1f MB\n", ESP.getFreePsram() / (1024.0*1024.0));
  Serial.printf("  Heap Internal: %d KB\n", ESP.getFreeHeap() / 1024);
  Serial.printf("  Model CNN    : %d KB (Flash)\n", saved_models_split_waste_cnn_int8_tflite_len / 1024);
  Serial.printf("  Model LSTM   : %d KB (Flash)\n", saved_models_split_waste_lstm_int8_tflite_len / 1024);
  Serial.println("=============================================\n");
}

void loop() {
  Serial.println("========================================");
  Serial.printf("[INFO] Mulai inferensi %d frames...\n", NUM_FRAMES);
  
  unsigned long pipeline_start_us = micros();

  // Array untuk menyimpan fitur CNN dari tiap frame
  int8_t cnn_features[NUM_FRAMES][NUM_FEATURES];

  // === TAHAP 1: Ekstrak Fitur CNN per Frame ===
  unsigned long cnn_total_us = 0;
  for (int f = 0; f < NUM_FRAMES; f++) {
    // Simulasi: isi frame dummy (nanti diganti data kamera asli)
    // Direct INT8 mapping: data langsung ditulis sebagai int8
    for (int i = 0; i < cnn_input->bytes; i++) {
      cnn_input->data.int8[i] = (int8_t)(f * 10 - 128 + (i % 50));
    }

    vTaskDelay(1); // Yield ke RTOS agar WDT tidak trigger

    unsigned long t0 = micros();
    if (cnn_interpreter->Invoke() != kTfLiteOk) {
      Serial.printf("[ERROR] CNN Invoke() gagal di frame %d!\n", f);
      return;
    }
    unsigned long dt = micros() - t0;
    cnn_total_us += dt;

    Serial.printf("  Frame %d/%d — CNN: %lu ms\n", f+1, NUM_FRAMES, dt/1000);

    // Simpan output fitur
    for (int i = 0; i < NUM_FEATURES; i++) {
      cnn_features[f][i] = cnn_output->data.int8[i];
    }
  }

  // === TAHAP 2: Masukkan fitur ke LSTM ===
  // Direct INT8 mapping: langsung copy int8 array tanpa konversi float
  int idx = 0;
  for (int f = 0; f < NUM_FRAMES; f++) {
    for (int i = 0; i < NUM_FEATURES; i++) {
      lstm_input->data.int8[idx++] = cnn_features[f][i];
    }
  }

  // === TAHAP 3: Inferensi LSTM ===
  unsigned long lstm_start_us = micros();
  if (lstm_interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[ERROR] LSTM Invoke() gagal!");
    return;
  }
  unsigned long lstm_us = micros() - lstm_start_us;

  // === TAHAP 4: Ambil Prediksi ===
  int best_class = 0;
  int8_t best_score = lstm_output->data.int8[0];
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (lstm_output->data.int8[i] > best_score) {
      best_score = lstm_output->data.int8[i];
      best_class = i;
    }
  }

  unsigned long pipeline_us = micros() - pipeline_start_us;

  // === LAPORAN BENCHMARK ===
  Serial.println("--- HASIL BENCHMARK ---");
  Serial.printf("  CNN  (%dx): %lu ms  (avg: %lu ms/frame)\n",
                NUM_FRAMES, cnn_total_us/1000, cnn_total_us/1000/NUM_FRAMES);
  Serial.printf("  LSTM (1x) : %lu ms\n", lstm_us/1000);
  Serial.printf("  TOTAL     : %lu ms\n", pipeline_us/1000);
  Serial.printf("  Throughput: %.1f FPS (pipeline)\n", 1000000.0 / pipeline_us);
  Serial.printf("  Prediksi  : %s (score: %d)\n", CLASS_NAMES[best_class], best_score);
  Serial.println("========================================\n");
  
  delay(5000);
}

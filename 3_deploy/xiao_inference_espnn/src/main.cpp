#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "soc/rtc.h"
#include "esp_camera.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_cnn_data.h"
#include "model_lstm_data.h"

static const char *TAG = "ESPNN";

// === KONFIGURASI ===
#define NUM_FRAMES    3
#define NUM_FEATURES  64
#define NUM_CLASSES   3

// === PIN KAMERA OV2640 — XIAO ESP32S3 Sense ===
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    10
#define CAM_PIN_SIOD    40
#define CAM_PIN_SIOC    39
#define CAM_PIN_D7      48
#define CAM_PIN_D6      11
#define CAM_PIN_D5      12
#define CAM_PIN_D4      14
#define CAM_PIN_D3      16
#define CAM_PIN_D2      18
#define CAM_PIN_D1      17
#define CAM_PIN_D0      15
#define CAM_PIN_VSYNC   38
#define CAM_PIN_HREF    47
#define CAM_PIN_PCLK    13

const char* CLASS_NAMES[] = {"kertas", "plastik", "organik"};

// === INIT KAMERA ===
static bool init_camera(void) {
    camera_config_t config = {};
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = CAM_PIN_D0;
    config.pin_d1       = CAM_PIN_D1;
    config.pin_d2       = CAM_PIN_D2;
    config.pin_d3       = CAM_PIN_D3;
    config.pin_d4       = CAM_PIN_D4;
    config.pin_d5       = CAM_PIN_D5;
    config.pin_d6       = CAM_PIN_D6;
    config.pin_d7       = CAM_PIN_D7;
    config.pin_xclk     = CAM_PIN_XCLK;
    config.pin_pclk     = CAM_PIN_PCLK;
    config.pin_vsync    = CAM_PIN_VSYNC;
    config.pin_href     = CAM_PIN_HREF;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.pin_pwdn     = CAM_PIN_PWDN;
    config.pin_reset    = CAM_PIN_RESET;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565;  // RGB untuk inferensi nanti
    config.frame_size   = FRAMESIZE_QVGA;    // 320x240
    config.fb_count     = 1;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Kamera GAGAL init! Error: 0x%x", err);
        return false;
    }

    // Test capture 1 frame
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) {
        ESP_LOGI(TAG, "[OK] Kamera OK! Test frame: %dx%d, %d bytes",
                 fb->width, fb->height, fb->len);
        esp_camera_fb_return(fb);
        return true;
    } else {
        ESP_LOGE(TAG, "Kamera init OK tapi capture gagal!");
        return false;
    }
}

extern "C" void app_main(void)
{
    // --- SET CPU 240 MHz ---
    rtc_cpu_freq_config_t cpu_config;
    rtc_clk_cpu_freq_mhz_to_config(240, &cpu_config);
    rtc_clk_cpu_freq_set_config(&cpu_config);
    ESP_LOGI(TAG, "[OK] CPU dinaikkan ke 240 MHz");

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "=============================================");
    ESP_LOGI(TAG, "  TEST INFERENSI + KAMERA DI ESP32S3");
    ESP_LOGI(TAG, "  MobileNetV2 a0.35 | 48x48 RGB | 3 Frames");
    ESP_LOGI(TAG, "  >>> ESP-NN AKTIF | CPU 240MHz <<<");
    ESP_LOGI(TAG, "=============================================");

    // --- PSRAM CHECK ---
    size_t psram_total = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    size_t psram_free  = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "[OK] PSRAM Total: %.1f MB, Free: %.1f MB",
             psram_total / (1024.0 * 1024.0),
             psram_free / (1024.0 * 1024.0));

    if (psram_total == 0) {
        ESP_LOGE(TAG, "PSRAM tidak terdeteksi! Berhenti.");
        return;
    }

    // --- ALOKASI ARENA DI PSRAM ---
    const int cnn_arena_size  = 3 * 1024 * 1024;
    const int lstm_arena_size = 1 * 1024 * 1024;

    uint8_t* cnn_arena  = (uint8_t*)heap_caps_malloc(cnn_arena_size, MALLOC_CAP_SPIRAM);
    uint8_t* lstm_arena = (uint8_t*)heap_caps_malloc(lstm_arena_size, MALLOC_CAP_SPIRAM);

    if (!cnn_arena || !lstm_arena) {
        ESP_LOGE(TAG, "Gagal alokasi PSRAM arena!");
        return;
    }
    ESP_LOGI(TAG, "[OK] Arena: CNN=%dKB, LSTM=%dKB",
             cnn_arena_size/1024, lstm_arena_size/1024);

    // --- OP RESOLVER untuk CNN (MobileNetV1) ---
    // Hanya daftarkan operator yang dibutuhkan model
    static tflite::MicroMutableOpResolver<20> cnn_resolver;
    cnn_resolver.AddConv2D();
    cnn_resolver.AddDepthwiseConv2D();
    cnn_resolver.AddAdd();
    cnn_resolver.AddMul();
    cnn_resolver.AddMean();
    cnn_resolver.AddPad();
    cnn_resolver.AddPadV2();
    cnn_resolver.AddReshape();
    cnn_resolver.AddFullyConnected();
    cnn_resolver.AddSoftmax();
    cnn_resolver.AddQuantize();
    cnn_resolver.AddDequantize();
    cnn_resolver.AddRelu();
    cnn_resolver.AddRelu6();
    cnn_resolver.AddLogistic();

    // --- SETUP CNN MODEL ---
    const tflite::Model* cnn_model = tflite::GetModel(saved_models_split_waste_cnn_int8_tflite);
    if (cnn_model == nullptr) {
        ESP_LOGE(TAG, "Model CNN invalid!");
        return;
    }

    static tflite::MicroInterpreter cnn_interp(
        cnn_model, cnn_resolver, cnn_arena, cnn_arena_size);

    if (cnn_interp.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "CNN AllocateTensors() gagal!");
        return;
    }

    TfLiteTensor* cnn_input  = cnn_interp.input(0);
    TfLiteTensor* cnn_output = cnn_interp.output(0);
    ESP_LOGI(TAG, "[OK] CNN siap. Input: %d bytes, Arena: %zu bytes",
             (int)cnn_input->bytes, cnn_interp.arena_used_bytes());

    // --- OP RESOLVER untuk Temporal Head (True LSTM Manual Unroll) ---
    static tflite::MicroMutableOpResolver<12> temporal_resolver;
    temporal_resolver.AddFullyConnected();
    temporal_resolver.AddSoftmax();
    temporal_resolver.AddQuantize();
    temporal_resolver.AddDequantize();
    temporal_resolver.AddReshape();
    temporal_resolver.AddConcatenation();
    temporal_resolver.AddLogistic();
    temporal_resolver.AddTanh();
    temporal_resolver.AddAdd();
    temporal_resolver.AddMul();
    temporal_resolver.AddStridedSlice();
    temporal_resolver.AddPack();

    // --- SETUP LSTM MODEL ---
    const tflite::Model* lstm_model = tflite::GetModel(saved_models_split_waste_lstm_int8_tflite);
    if (lstm_model == nullptr) {
        ESP_LOGE(TAG, "Model LSTM invalid!");
        return;
    }

    static tflite::MicroInterpreter lstm_interp(
        lstm_model, temporal_resolver, lstm_arena, lstm_arena_size);

    if (lstm_interp.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "LSTM AllocateTensors() gagal!");
        return;
    }

    TfLiteTensor* lstm_input  = lstm_interp.input(0);
    TfLiteTensor* lstm_output = lstm_interp.output(0);
    ESP_LOGI(TAG, "[OK] LSTM siap. Input: %d bytes, Arena: %zu bytes",
             (int)lstm_input->bytes, lstm_interp.arena_used_bytes());

    // --- RINGKASAN ---
    ESP_LOGI(TAG, "--- RINGKASAN MEMORI ---");
    ESP_LOGI(TAG, "  PSRAM Sisa   : %.1f MB",
             heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / (1024.0*1024.0));
    ESP_LOGI(TAG, "  Heap Internal: %d KB",
             (int)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
    ESP_LOGI(TAG, "  Model CNN    : %d KB (Flash)",
             (int)(saved_models_split_waste_cnn_int8_tflite_len / 1024));
    ESP_LOGI(TAG, "  Model LSTM   : %d KB (Flash)",
             (int)(saved_models_split_waste_lstm_int8_tflite_len / 1024));
    ESP_LOGI(TAG, "=============================================");

    // --- INIT KAMERA ---
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "[INFO] Menginisialisasi kamera OV2640...");
    bool camera_ok = init_camera();
    ESP_LOGI(TAG, "  PSRAM setelah kamera: %.1f MB free",
             heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / (1024.0*1024.0));
    if (!camera_ok) {
        ESP_LOGW(TAG, "Kamera tidak tersedia — lanjut benchmark tanpa kamera");
    }

    // === MAIN LOOP ===
    int iteration = 0;
    while (1) {
        iteration++;
        ESP_LOGI(TAG, "");
        ESP_LOGI(TAG, "========================================");
        ESP_LOGI(TAG, "[INFO] Inferensi #%d — %d frames (KAMERA ASLI)...", iteration, NUM_FRAMES);

        int8_t cnn_features[NUM_FRAMES][NUM_FEATURES];
        int64_t pipeline_start = esp_timer_get_time();

        // === TAHAP 1: Capture + CNN per Frame ===
        int64_t cnn_total_us = 0;
        int64_t capture_total_us = 0;
        bool frame_ok = true;

        for (int f = 0; f < NUM_FRAMES; f++) {
            // --- Capture frame dari kamera ---
            int64_t cap_start = esp_timer_get_time();
            camera_fb_t* fb = NULL;
            if (camera_ok) {
                fb = esp_camera_fb_get();
            }
            int64_t cap_us = esp_timer_get_time() - cap_start;
            capture_total_us += cap_us;

            if (fb && camera_ok) {
                // --- Center crop 320x240 → 240x240, lalu downscale ke 48x48 ---
                // RGB565: 2 bytes per pixel
                int src_w = fb->width;    // 320
                int src_h = fb->height;   // 240
                int crop_size = (src_w < src_h) ? src_w : src_h; // 240
                int crop_x = (src_w - crop_size) / 2;  // 40
                int crop_y = (src_h - crop_size) / 2;  // 0

                int dst_w = 48, dst_h = 48;
                uint16_t* rgb565 = (uint16_t*)fb->buf;

                for (int dy = 0; dy < dst_h; dy++) {
                    for (int dx = 0; dx < dst_w; dx++) {
                        // Nearest-neighbor sampling
                        int sx = crop_x + (dx * crop_size) / dst_w;
                        int sy = crop_y + (dy * crop_size) / dst_h;
                        uint16_t pixel = rgb565[sy * src_w + sx];

                        // RGB565 → RGB888
                        uint8_t r = ((pixel >> 11) & 0x1F) << 3;
                        uint8_t g = ((pixel >> 5)  & 0x3F) << 2;
                        uint8_t b = ((pixel >> 0)  & 0x1F) << 3;

                        // Normalisasi ke INT8: (0-255) → (-128 to 127)
                        int idx = (dy * dst_w + dx) * 3;
                        cnn_input->data.int8[idx + 0] = (int8_t)(r - 128);
                        cnn_input->data.int8[idx + 1] = (int8_t)(g - 128);
                        cnn_input->data.int8[idx + 2] = (int8_t)(b - 128);
                    }
                }
                esp_camera_fb_return(fb);
            } else {
                // Fallback: dummy data jika kamera tidak tersedia
                for (int i = 0; i < (int)cnn_input->bytes; i++) {
                    cnn_input->data.int8[i] = (int8_t)(f * 10 - 128 + (i % 50));
                }
                if (f == 0 && camera_ok) {
                    ESP_LOGW(TAG, "Capture gagal, pakai dummy data");
                }
            }

            vTaskDelay(1);

            // --- CNN Invoke ---
            int64_t t0 = esp_timer_get_time();
            if (cnn_interp.Invoke() != kTfLiteOk) {
                ESP_LOGE(TAG, "CNN Invoke() gagal di frame %d!", f);
                frame_ok = false;
                break;
            }
            int64_t dt = esp_timer_get_time() - t0;
            cnn_total_us += dt;

            ESP_LOGI(TAG, "  Frame %d/%d — Capture: %lld ms, CNN: %lld ms",
                     f+1, NUM_FRAMES, cap_us/1000, dt/1000);

            for (int i = 0; i < NUM_FEATURES; i++) {
                cnn_features[f][i] = cnn_output->data.int8[i];
            }

            // Delay antar frame untuk variasi temporal
            if (f < NUM_FRAMES - 1) {
                vTaskDelay(pdMS_TO_TICKS(200));
            }
        }

        if (!frame_ok) {
            vTaskDelay(pdMS_TO_TICKS(5000));
            continue;
        }

        // === TAHAP 2: Temporal Head ===
        int idx = 0;
        for (int f = 0; f < NUM_FRAMES; f++) {
            for (int i = 0; i < NUM_FEATURES; i++) {
                lstm_input->data.int8[idx++] = cnn_features[f][i];
            }
        }

        int64_t lstm_start = esp_timer_get_time();
        if (lstm_interp.Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Temporal Head Invoke() gagal!");
            break;
        }
        int64_t lstm_us = esp_timer_get_time() - lstm_start;

        // === TAHAP 3: Prediksi ===
        int best_class = 0;
        int8_t best_score = lstm_output->data.int8[0];
        for (int i = 1; i < NUM_CLASSES; i++) {
            if (lstm_output->data.int8[i] > best_score) {
                best_score = lstm_output->data.int8[i];
                best_class = i;
            }
        }

        int64_t pipeline_us = esp_timer_get_time() - pipeline_start;

        // === LAPORAN ===
        ESP_LOGI(TAG, "--- HASIL (ESP-NN + KAMERA) ---");
        ESP_LOGI(TAG, "  Capture: %lld ms total", capture_total_us/1000);
        ESP_LOGI(TAG, "  CNN  (%dx): %lld ms  (avg: %lld ms/frame)",
                 NUM_FRAMES, cnn_total_us/1000, cnn_total_us/1000/NUM_FRAMES);
        ESP_LOGI(TAG, "  Temporal : %lld ms", lstm_us/1000);
        ESP_LOGI(TAG, "  TOTAL    : %lld ms", pipeline_us/1000);
        ESP_LOGI(TAG, "  >>> Prediksi: %s (score: %d) <<<",
                 CLASS_NAMES[best_class], best_score);
        ESP_LOGI(TAG, "  Semua skor: kertas=%d, plastik=%d, organik=%d",
                 lstm_output->data.int8[0], lstm_output->data.int8[1], lstm_output->data.int8[2]);
        ESP_LOGI(TAG, "========================================");

        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}

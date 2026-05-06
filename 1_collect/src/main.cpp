/*
 * ============================================================
 * DIAGNOSTIK KAMERA XIAO ESP32S3 Sense
 * Tujuan: Mencari tahu kenapa esp_camera_fb_get() return NULL
 * ============================================================
 */

#include <Arduino.h>
#include "esp_camera.h"

#include "SD.h"
#include "FS.h"
#include "SPIFFS.h"

// ============================================================
// KONFIGURASI PIN KAMERA OV2640 — XIAO ESP32S3 Sense
// ============================================================
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    10
#define SIOD_GPIO_NUM    40
#define SIOC_GPIO_NUM    39
#define Y9_GPIO_NUM      48
#define Y8_GPIO_NUM      11
#define Y7_GPIO_NUM      12
#define Y6_GPIO_NUM      14
#define Y5_GPIO_NUM      16
#define Y4_GPIO_NUM      18
#define Y3_GPIO_NUM      17
#define Y2_GPIO_NUM      15
#define VSYNC_GPIO_NUM   38
#define HREF_GPIO_NUM    47
#define PCLK_GPIO_NUM    13

// ============================================================
// KONFIGURASI PIN LAINNYA
// ============================================================
#define SD_CS_PIN        21
#define BUTTON_PIN       D7
#define LED_FLASH_PIN    D3

// ============================================================
// KONFIGURASI CAPTURE
// ============================================================
#define RECORD_TIME_MS      3000
#define DEBOUNCE_DELAY_MS   1000

const char* CLASS_NAMES[] = {"kertas", "plastik", "organik"};
const int NUM_CLASSES = 3;

int kelasAktif = 0;
int sessionCounters[3] = {0, 0, 0};
bool sdReady = false;
bool cameraReady = false;
unsigned long lastButtonPress = 0;
const char* COUNTER_FILE = "/session_counters.dat";

// Forward declarations
void saveSessionCounters();
void printKelasAktif();

// ============================================================
// DIAGNOSTIK: Cek PSRAM
// ============================================================
void checkPSRAM() {
  Serial.println("\n--- DIAGNOSTIK PSRAM ---");
  if (psramFound()) {
    Serial.printf("[OK] PSRAM terdeteksi! Ukuran: %d bytes (%.1f MB)\n", 
                  ESP.getPsramSize(), ESP.getPsramSize() / (1024.0 * 1024.0));
    Serial.printf("     PSRAM Terpakai: %d bytes\n", ESP.getPsramSize() - ESP.getFreePsram());
    Serial.printf("     PSRAM Tersisa : %d bytes\n", ESP.getFreePsram());
  } else {
    Serial.println("[ERROR] PSRAM TIDAK TERDETEKSI!");
    Serial.println("        Kamera membutuhkan PSRAM untuk frame buffer.");
    Serial.println("        Pastikan build_flags berisi -DBOARD_HAS_PSRAM");
  }
  Serial.printf("[INFO] Heap tersedia: %d bytes\n", ESP.getFreeHeap());
}

// ============================================================
// INIT KAMERA — Versi dengan diagnostik
// ============================================================
bool initCamera() {
  Serial.println("\n--- INIT KAMERA ---");
  
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;        // 20MHz standar
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_QVGA;   // 320x240 — didukung penuh oleh OV2640
  config.jpeg_quality = 12;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

  // Pilih lokasi frame buffer berdasarkan ketersediaan PSRAM
  if (psramFound()) {
    config.fb_count    = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    Serial.println("[INFO] Menggunakan PSRAM untuk frame buffer (2 buffer).");
  } else {
    config.fb_count    = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
    Serial.println("[WARNING] PSRAM tidak ada, menggunakan DRAM (terbatas).");
  }

  Serial.println("[INFO] Menginisialisasi kamera...");
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[ERROR] esp_camera_init() GAGAL! Kode: 0x%x\n", err);
    if (err == 0x105) {
      Serial.println("        -> Ini biasanya berarti kamera TIDAK TERHUBUNG secara fisik,");
      Serial.println("           atau konektor flex cable longgar/terbalik.");
    }
    return false;
  }

  Serial.println("[OK] esp_camera_init() BERHASIL!");

  // === TEST CAPTURE 1 FRAME ===
  Serial.println("\n--- TEST CAPTURE FRAME ---");
  Serial.println("[INFO] Mencoba ambil 1 frame...");
  
  // Buang frame pertama (sering kotor/gelap)
  camera_fb_t* fb = esp_camera_fb_get();
  if (fb) {
    Serial.printf("[OK] Frame pertama berhasil! Ukuran: %d bytes, Resolusi: %dx%d\n", 
                  fb->len, fb->width, fb->height);
    esp_camera_fb_return(fb);
  } else {
    Serial.println("[ERROR] Frame pertama NULL!");
    Serial.println("        Kemungkinan penyebab:");
    Serial.println("        1. PSRAM tidak cukup / tidak aktif");
    Serial.println("        2. Konektor kamera longgar");
    Serial.println("        3. Sensor kamera rusak");
    return false;
  }

  // Coba frame kedua untuk memastikan
  delay(100);
  fb = esp_camera_fb_get();
  if (fb) {
    Serial.printf("[OK] Frame kedua berhasil! Ukuran: %d bytes\n", fb->len);
    esp_camera_fb_return(fb);
  } else {
    Serial.println("[WARNING] Frame kedua NULL. Kamera mungkin lambat.");
  }

  return true;
}

// ============================================================
// SD Card
// ============================================================
bool initSDCard() {
  Serial.println("\n--- INIT SD CARD ---");
  if (!SD.begin(SD_CS_PIN)) {
    Serial.println("[ERROR] SD card tidak terbaca!");
    return false;
  }
  uint8_t cardType = SD.cardType();
  if (cardType == CARD_NONE) {
    Serial.println("[ERROR] Tidak ada SD card terdeteksi!");
    return false;
  }
  Serial.printf("[OK] SD card OK. Tipe: %s\n", 
    cardType == CARD_MMC ? "MMC" : cardType == CARD_SD ? "SD" : "SDHC");
  float freeMB = (SD.totalBytes() - SD.usedBytes()) / (1024.0 * 1024.0);
  Serial.printf("     Sisa: %.1f MB\n", freeMB);
  return true;
}

void createClassFolders() {
  for (int i = 0; i < NUM_CLASSES; i++) {
    String path = "/" + String(CLASS_NAMES[i]);
    if (!SD.exists(path)) SD.mkdir(path);
  }
}

void loadSessionCounters() {
  if (!SPIFFS.begin(true)) return;
  if (SPIFFS.exists(COUNTER_FILE)) {
    File f = SPIFFS.open(COUNTER_FILE, FILE_READ);
    if (f) {
      for (int i = 0; i < NUM_CLASSES; i++) {
        if (f.available() >= sizeof(int))
          f.read((uint8_t*)&sessionCounters[i], sizeof(int));
      }
      f.close();
    }
  }
}

void saveSessionCounters() {
  File f = SPIFFS.open(COUNTER_FILE, FILE_WRITE);
  if (f) {
    for (int i = 0; i < NUM_CLASSES; i++)
      f.write((uint8_t*)&sessionCounters[i], sizeof(int));
    f.close();
  }
}

void printKelasAktif() {
  Serial.println("========================================");
  Serial.printf("[KELAS AKTIF] >> %s << (sesi berikutnya: sesi_%03d)\n",
                CLASS_NAMES[kelasAktif], sessionCounters[kelasAktif] + 1);
  Serial.println("========================================");
}

// ============================================================
// CAPTURE 3 DETIK
// ============================================================
void captureSession3Seconds() {
  if (!sdReady || !cameraReady) {
    Serial.println("[ERROR] SD atau Kamera tidak siap!");
    return;
  }

  sessionCounters[kelasAktif]++;
  int sesiNum = sessionCounters[kelasAktif];

  char sesiFolderPath[64];
  snprintf(sesiFolderPath, sizeof(sesiFolderPath), "/%s/sesi_%03d",
           CLASS_NAMES[kelasAktif], sesiNum);

  if (!SD.mkdir(sesiFolderPath)) {
    Serial.printf("[ERROR] Gagal membuat folder '%s'!\n", sesiFolderPath);
    sessionCounters[kelasAktif]--;
    return;
  }

  Serial.println("----------------------------------------");
  Serial.printf("[CAPTURE] Merekam selama 3 detik...\n");
  Serial.printf("Target Folder: %s\n", sesiFolderPath);

  // Nyalakan LED Flash
  digitalWrite(LED_FLASH_PIN, HIGH);

  int frameCount = 0;
  int failCount = 0;
  unsigned long startTime = millis();

  while (millis() - startTime < RECORD_TIME_MS) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      failCount++;
      if (failCount <= 3) {
        Serial.printf("[WARNING] Frame gagal (%d)\n", failCount);
      }
      delay(100);
      continue;
    }

    frameCount++;
    char filePath[80];
    snprintf(filePath, sizeof(filePath), "%s/frame_%03d.jpg",
             sesiFolderPath, frameCount);

    File file = SD.open(filePath, FILE_WRITE);
    if (file) {
      file.write(fb->buf, fb->len);
      file.close();
    } else {
      Serial.printf("[ERROR] Gagal tulis '%s'\n", filePath);
    }

    esp_camera_fb_return(fb);
  }

  // Matikan LED Flash
  digitalWrite(LED_FLASH_PIN, LOW);

  Serial.printf("[SELESAI] %d frame OK, %d frame gagal.\n", frameCount, failCount);
  Serial.println("----------------------------------------");

  saveSessionCounters();
  float freeMB = (SD.totalBytes() - SD.usedBytes()) / (1024.0 * 1024.0);
  Serial.printf("[STORAGE] Sisa SD card: %.1f MB\n", freeMB);
  Serial.println("[INFO] Tekan tombol untuk merekam sesi berikutnya.");
  printKelasAktif();
}

void checkSerialInput() {
  if (Serial.available() > 0) {
    char input = Serial.read();
    while (Serial.available() > 0) Serial.read();
    switch (input) {
      case 'k': case 'K': kelasAktif = 0; break;
      case 'p': case 'P': kelasAktif = 1; break;
      case 'o': case 'O': kelasAktif = 2; break;
      case 'r': case 'R': 
        for(int i=0; i<NUM_CLASSES; i++) sessionCounters[i] = 0;
        saveSessionCounters();
        Serial.println("\n[RESET] Counter sesi telah direset (selanjutnya mulai dari sesi_001).");
        printKelasAktif();
        return;
      case 'c': case 'C':
        Serial.println("\n[CAPTURE] Merekam sesi dari input Serial...");
        captureSession3Seconds();
        return;
      default: return;
    }
    Serial.printf("\n[GANTI KELAS] Kelas aktif: %s\n", CLASS_NAMES[kelasAktif]);
    printKelasAktif();
  }
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(3000); // Tunggu Serial Monitor terbuka

  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_FLASH_PIN, OUTPUT);
  digitalWrite(LED_FLASH_PIN, LOW);

  Serial.println("\n\n============================================");
  Serial.println("  XIAO ESP32S3 Sense — DIAGNOSTIK KAMERA");
  Serial.println("============================================");

  // 1. Cek PSRAM
  checkPSRAM();

  // 2. Init Kamera (termasuk test capture)
  cameraReady = initCamera();

  // 3. Init SD Card
  sdReady = initSDCard();
  if (sdReady) createClassFolders();

  // 4. Load counter
  loadSessionCounters();

  // 5. Status akhir
  Serial.println("\n============================================");
  Serial.printf("  Kamera : %s\n", cameraReady ? "OK" : "GAGAL");
  Serial.printf("  SD Card: %s\n", sdReady ? "OK" : "GAGAL");
  Serial.println("============================================");

  if (cameraReady && sdReady) {
    Serial.println("  SISTEM SIAP! Tekan tombol untuk merekam.");
    Serial.println("  Serial: k=kertas p=plastik o=organik");
    printKelasAktif();
  } else {
    Serial.println("  [ERROR] Sistem TIDAK SIAP. Periksa output di atas.");
  }
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  checkSerialInput();

  if (digitalRead(BUTTON_PIN) == LOW) {
    unsigned long now = millis();
    if (now - lastButtonPress > DEBOUNCE_DELAY_MS) {
      lastButtonPress = now;
      captureSession3Seconds();
      while (digitalRead(BUTTON_PIN) == LOW) { delay(10); }
      lastButtonPress = millis();
    }
  }
  delay(10);
}

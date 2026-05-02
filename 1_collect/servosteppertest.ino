const int buttonPin = D7; // Pin untuk Button
const int ledPin = D3;    // Pin untuk LED Flash

void setup() {
  Serial.begin(115200);
  
  // Konfigurasi pin
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(ledPin, OUTPUT);
  
  Serial.println("=== TEST LED & BUTTON ===");
  Serial.println("Tekan tombol untuk menyalakan LED");
}

void loop() {
  // Membaca status button (LOW jika ditekan, HIGH jika dilepas)
  int buttonState = digitalRead(buttonPin);
  
  if (buttonState == LOW) {
    // Tombol ditekan
    digitalWrite(ledPin, HIGH); // Nyalakan LED
    Serial.println("Tombol ditekan! LED NYALA");
    
    // Tunggu sampai tombol dilepas agar tidak spam di Serial Monitor
    while(digitalRead(buttonPin) == LOW) {
      delay(10);
    }
    
    Serial.println("Tombol dilepas! LED MATI");
  } else {
    // Tombol dilepas
    digitalWrite(ledPin, LOW);  // Matikan LED
  }
  
  delay(10);
}

int pinA2 = 14;   // D5
int pinA1 = 12;   // D6
int pinA0 = 13;   // D7

int pinB2 = 16;   // D0
int pinB1 = 5;    // D1
int pinB0 = 4;    // D2

void setup() {
  Serial.begin(115200);
  delay(1000);  

  pinMode(pinA2, OUTPUT);
  pinMode(pinA1, OUTPUT);
  pinMode(pinA0, OUTPUT);

  pinMode(pinB2, OUTPUT);
  pinMode(pinB1, OUTPUT);
  pinMode(pinB0, OUTPUT);

  Serial.println("READY");  // indicates ESP is ready to receive
}

void loop() {
  if (Serial.available()) {
    String incoming = Serial.readStringUntil('\n');
    int commaIndex = incoming.indexOf(',');
    if (commaIndex == -1) {
      Serial.println("Invalid data format!");
      return;
    }

    int num1 = incoming.substring(0, commaIndex).toInt();
    int num2 = incoming.substring(commaIndex + 1).toInt();

    // DEBUG: print received numbers
    Serial.print("Received first integer: ");
    Serial.println(num1);
    Serial.print("Received second integer: ");
    Serial.println(num2);

    // Set pins based on numbers
    setPins(num1, pinA2, pinA1, pinA0);
    setPins(num2, pinB2, pinB1, pinB0);

    Serial.println("OK");  // signal Python that processing is done
  }
}

void setPins(int num, int p2, int p1, int p0) {
  int bit2 = (num >> 2) & 1;
  int bit1 = (num >> 1) & 1;
  int bit0 = (num >> 0) & 1;

  digitalWrite(p2, bit2);
  digitalWrite(p1, bit1);
  digitalWrite(p0, bit0);

  // DEBUG: print pin states
  Serial.print("Pin ");
  Serial.print(p2);
  Serial.print(" is ");
  Serial.println(bit2 ? "HIGH" : "LOW");

  Serial.print("Pin ");
  Serial.print(p1);
  Serial.print(" is ");
  Serial.println(bit1 ? "HIGH" : "LOW");

  Serial.print("Pin ");
  Serial.print(p0);
  Serial.print(" is ");
  Serial.println(bit0 ? "HIGH" : "LOW");
}

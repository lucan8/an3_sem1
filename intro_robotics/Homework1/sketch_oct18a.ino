/*TODO:
vector of "Color" class (RGB) 
"Color" class has the following fields:
name, pot_pin, led_pin
*/
// RGB constants
const char* RGB_COLOR_RED = "RED";
const char* RGB_COLOR_GREEN = "GREEN";
const char* RGB_COLOR_BLUE = "BLUE";

const int RGB_POT_PIN_RED = A0;
const int RGB_POT_PIN_GREEN = A1;
const int RGB_POT_PIN_BLUE = A2;

const int RGB_LED_PIN_RED = 11;
const int RGB_LED_PIN_GREEN = 5;
const int RGB_LED_PIN_BLUE = 10;

// Limits
const int ANALOG_READ_MIN = 0;
const int ANALOG_READ_MAX = 1023;
const int ANALOG_WRITE_MIN = 0;
const int ANALOG_WRITE_MAX = 255;

const int SERIAL_BAUD = 9600;

void setup() {
  Serial.begin(SERIAL_BAUD);
  setupInput();
  setupOutput();
}

void loop() {
  int pot_r = analogRead(RGB_POT_PIN_RED);
  int pot_g = analogRead(RGB_POT_PIN_GREEN);
  int pot_b = analogRead(RGB_POT_PIN_BLUE);

  int led_r_val = map(pot_r, ANALOG_READ_MIN, ANALOG_READ_MAX, ANALOG_WRITE_MIN, ANALOG_WRITE_MAX);
  int led_g_val = map(pot_g, ANALOG_READ_MIN, ANALOG_READ_MAX, ANALOG_WRITE_MIN, ANALOG_WRITE_MAX);
  int led_b_val = map(pot_b, ANALOG_READ_MIN, ANALOG_READ_MAX, ANALOG_WRITE_MIN, ANALOG_WRITE_MAX);

  handlePin(RGB_LED_PIN_RED, led_r_val, RGB_COLOR_RED);
  Serial.print(", ");
  handlePin(RGB_LED_PIN_GREEN, led_g_val, RGB_COLOR_GREEN);
  Serial.print(", ");
  handlePin(RGB_LED_PIN_BLUE, led_b_val, RGB_COLOR_BLUE);
  Serial.print('\n');
}

// Print and set pin value
void handlePin(int led_pin, int led_val, const char* led_color){
  Serial.print(led_color);
  Serial.print(": ");
  Serial.print(led_val);

  analogWrite(led_pin, led_val);
}

void setupInput(){
  pinMode(A0, INPUT);
  pinMode(A1, INPUT);
  pinMode(A2, INPUT);
}

void setupOutput(){
  pinMode(RGB_LED_PIN_RED, OUTPUT);
  pinMode(RGB_LED_PIN_GREEN, OUTPUT);
  pinMode(RGB_LED_PIN_BLUE, OUTPUT);
}
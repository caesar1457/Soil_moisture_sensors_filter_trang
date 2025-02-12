// Define the analog pin connected to the soil moisture sensor
const int sensorPin = A0; // SIG pin connected to A0

// EKF variables
float x = 0.0;         // State: Estimated moisture level
float P = 1.0;         // Covariance: Uncertainty in the estimate
float Q = 0.01;        // Process noise covariance
float R = 1.0;         // Measurement noise covariance

// EKF update function
void EKF_Update(float z) {
    // Prediction step
    float x_pred = x;             // Predicted state (no dynamics in this case)
    float P_pred = P + Q;         // Predicted covariance

    // Update step
    float K = P_pred / (P_pred + R);  // Kalman gain
    x = x_pred + K * (z - x_pred);   // Update state estimate
    P = (1 - K) * P_pred;            // Update covariance
}

void setup() {
    // Initialize serial communication
    Serial.begin(9600);
    while (!Serial) {
        ; // Wait for serial port to connect
    }
    Serial.println("Soil Moisture Sensor with EKF");
}

void loop() {
    // Read raw data from the sensor
    int rawValue = analogRead(sensorPin);


    // 10mh
    // Convert the raw data to a percentage (0-100%)
    float moisture = map(rawValue, 470, 840, 100, 0);

    // 50kh
    // Convert the raw data to a percentage (0-100%)
    // float moisture = map(rawValue, 200, 855, 100, 0);
    
    if (moisture < 0)
      moisture = 0;
    else if (moisture > 100)
      moisture = 100;

    // Apply EKF
    EKF_Update(moisture);

    // Print raw and filtered values
    Serial.print("Raw Value: ");
    Serial.print(rawValue);
    Serial.print(" - Moisture: ");
    Serial.print(moisture);
    Serial.print("% - EKF Filtered Moisture: ");
    Serial.print(x);
    Serial.println("%");

    // Delay before next reading
    delay(250);
}

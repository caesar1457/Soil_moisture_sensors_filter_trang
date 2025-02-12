# EKF Soil Moisture Sensor Project

## Project Overview
This project implements a soil moisture sensor system based on Extended Kalman Filter (EKF). The system consists of Arduino code and Python code, which are used for sensor data acquisition and filtering, respectively.

## File Structure
```
final_codes/
│── Arduino \ 10mh_sensor/
│   ├── EKF_sensor.ino  # Arduino code for sensor data processing
│── python/
│   ├── ekf_module.py    # EKF-related module
│   ├── main.py          # Main program entry
│   ├── sensor_reader.py # Sensor data reading module
│── __pycache__/        # Python cache files
```

## Required Libraries
Before running the Python code, make sure the following libraries are installed:
```python
import time
import queue
import matplotlib.pyplot as plt
```
Install missing libraries using `pip`:
```sh
pip install matplotlib
```

## Usage Instructions
1. **Arduino Code**
   - Open `EKF_sensor.ino` using Arduino IDE.
   - Select the correct port (`Tools` -> `Port`).
   - Upload the code to the Arduino device.

2. **Python Code**
   - Run `python/main.py`, ensuring the Arduino device is connected.
   - Select the correct serial port, typically `/dev/ttyUSB0` (Linux) or `COMx` (Windows). The port can be modified in `sensor_reader.py`.
   - Run `main.py` to read data from Arduino, perform EKF processing, and visualize the results.

## Important Notes
- Ensure the Arduino code is successfully uploaded and running before executing the Python code.
- Incorrect port selection will prevent data reading. Make sure to configure the correct serial port in `sensor_reader.py`.

## License
This project follows the MIT License. Refer to the LICENSE file for details.

---
For any issues, contact the developer or submit an issue.

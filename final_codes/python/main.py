import time
import queue
import threading
import matplotlib.pyplot as plt

from sensor_reader import SensorReader
from ekf_module import EKF
from visualizer import CombinedVisualizer

def map_raw_to_percentage(raw_value, sensor_id):
    """
    Convert the raw sensor data to a moisture value represented as a percentage.
    A lower raw value corresponds to higher moisture. Uses sensor-specific calibration parameters,
    and clamps the result between 0 and 100.
    """
    if sensor_id == 1:
        in_min, in_max = 200, 855
    elif sensor_id == 2:
        in_min, in_max = 250, 900
    else:
        in_min, in_max = 200, 855  # Default parameters
    percentage = 100 - ((raw_value - in_min) * 100 / (in_max - in_min))
    return max(min(percentage, 100), 0)

def sensor_data_loop(sensor1_queue, sensor2_queue, ekf1, ekf2, visualizer):
    """
    Continuously processes sensor data and updates the visualizer in a background thread.
    Uses plt.fignum_exists() to check if the figure window still exists; exits the loop if the window is closed.
    """
    while plt.fignum_exists(visualizer.fig.number):
        # Process Sensor 1 data
        try:
            data1 = sensor1_queue.get_nowait()
            raw_value = data1['raw']
            percentage = map_raw_to_percentage(raw_value, sensor_id=1)
            filtered = ekf1.update(percentage)
            ci = ekf1.get_confidence_interval()
            visualizer.update_sensor_data(1, percentage, filtered, ci)
            print(f"Sensor 1: Raw {raw_value}, Moisture: {percentage:.2f}%, EKF: {filtered:.2f}%")
        except queue.Empty:
            pass

        # Process Sensor 2 data
        try:
            data2 = sensor2_queue.get_nowait()
            raw_value = data2['raw']
            percentage = map_raw_to_percentage(raw_value, sensor_id=2)
            filtered = ekf2.update(percentage)
            ci = ekf2.get_confidence_interval()
            visualizer.update_sensor_data(2, percentage, filtered, ci)
            print(f"Sensor 2: Raw {raw_value}, Moisture: {percentage:.2f}%, EKF: {filtered:.2f}%")
        except queue.Empty:
            pass

        time.sleep(0.01)  # Avoid excessive CPU usage

def main():
    # Create queues for receiving sensor data
    sensor1_queue = queue.Queue()
    sensor2_queue = queue.Queue()

    # Initialize the EKF estimators, one for each sensor
    ekf1 = EKF()
    ekf2 = EKF()

    # Initialize sensor reader threads (modify the port numbers "COM6" and "COM7" according to your actual configuration)
    sensor1_reader = SensorReader(port="COM6", sensor_id=1, data_queue=sensor1_queue)
    sensor2_reader = SensorReader(port="COM7", sensor_id=2, data_queue=sensor2_queue)
    
    sensor1_reader.start()
    sensor2_reader.start()

    # Create a new visualizer instance (make sure that in visualizer.py the CombinedVisualizer
    # uses tight_layout on the figure or call tight_layout() here to avoid overlapping subplots)
    visualizer = CombinedVisualizer(sensor_ids=[1, 2], window_size=50)
    visualizer.fig.tight_layout()  # Automatically adjust subplot layout

    # Start a background thread to process sensor data updates
    sensor_thread = threading.Thread(target=sensor_data_loop, args=(sensor1_queue, sensor2_queue, ekf1, ekf2, visualizer))
    sensor_thread.daemon = True  # Set as a daemon thread so it will exit when the main window is closed
    sensor_thread.start()

    # Use a while loop with plt.pause() for real-time figure updates, allowing Ctrl+C to interrupt the program
    try:
        while True:
            plt.pause(0.1)  # Periodically update the figure window
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, exiting...")
    finally:
        # Stop the sensor reader threads
        sensor1_reader.stop()
        sensor2_reader.stop()
        sensor1_reader.join()
        sensor2_reader.join()
        print("Sensor readers stopped. Exiting program.")

if __name__ == "__main__":
    main()

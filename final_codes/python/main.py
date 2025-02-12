import time
import queue
import matplotlib.pyplot as plt

from sensor_reader import SensorReader
from ekf_module import EKF

class CombinedVisualizer:
    """Real-time visualization with 4 subplots (2 sensor plots, EKF difference, and data comparison)."""
    def __init__(self, sensor_ids, window_size=100):
        self.sensor_ids = sensor_ids
        self.window_size = window_size  # Display only the latest window_size data points per plot
        # Initialize data storage for each sensor
        self.data = {sid: {'raw': [], 'ekf': [], 'ci_lower': [], 'ci_upper': []} for sid in sensor_ids}
        
        # Create a 2x2 subplot layout
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.ax1 = self.axes[0, 0]  # Sensor 1 plot (top left)
        self.ax2 = self.axes[0, 1]  # Sensor 2 plot (top right)
        self.ax_diff = self.axes[1, 0]  # EKF difference (bottom left)
        self.ax_comp = self.axes[1, 1]  # Data comparison (bottom right)

        plt.ion()  # Enable interactive mode
        plt.show()

    def update_sensor_data(self, sensor_id, raw, ekf, ci):
        """Update data for a given sensor."""
        self.data[sensor_id]['raw'].append(raw)
        self.data[sensor_id]['ekf'].append(ekf)
        self.data[sensor_id]['ci_lower'].append(ci[0])
        self.data[sensor_id]['ci_upper'].append(ci[1])

    def _get_recent(self, data_list):
        """Return the last window_size data points and corresponding x-axis (sample indices)."""
        n = len(data_list)
        if n > self.window_size:
            return list(range(n - self.window_size + 1, n + 1)), data_list[-self.window_size:]
        return list(range(1, n + 1)), data_list

    def refresh(self):
        """Refresh all subplots to display the latest data."""
        # Clear all subplots
        for ax in [self.ax1, self.ax2, self.ax_diff, self.ax_comp]:
            ax.cla()

        # --- Sensor 1 Plot (Top Left) ---
        s1 = self.data[1]
        x1, raw1 = self._get_recent(s1['raw'])
        _, ekf1 = self._get_recent(s1['ekf'])
        _, ci_lower1 = self._get_recent(s1['ci_lower'])
        _, ci_upper1 = self._get_recent(s1['ci_upper'])
        if raw1:
            self.ax1.scatter(x1, raw1, color='orange', label='Raw Data', s=30)
            self.ax1.plot(x1, ekf1, color='blue', linewidth=2, label='EKF Estimate')
            self.ax1.fill_between(x1, ci_lower1, ci_upper1, color='blue', alpha=0.2, label='Confidence Interval')
            self.ax1.legend()
        self.ax1.set_title("Sensor 1 Real-time Data")
        self.ax1.set_xlabel("Sample Index")
        self.ax1.set_ylabel("Moisture (%)")
        self.ax1.set_ylim(0, 110)
        self.ax1.grid(True)

        # --- Sensor 2 Plot (Top Right) ---
        s2 = self.data[2]
        x2, raw2 = self._get_recent(s2['raw'])
        _, ekf2 = self._get_recent(s2['ekf'])
        _, ci_lower2 = self._get_recent(s2['ci_lower'])
        _, ci_upper2 = self._get_recent(s2['ci_upper'])
        if raw2:
            self.ax2.scatter(x2, raw2, color='red', label='Raw Data', s=30)
            self.ax2.plot(x2, ekf2, color='purple', linewidth=2, label='EKF Estimate')
            self.ax2.fill_between(x2, ci_lower2, ci_upper2, color='purple', alpha=0.2, label='Confidence Interval')
            self.ax2.legend()
        self.ax2.set_title("Sensor 2 Real-time Data")
        self.ax2.set_xlabel("Sample Index")
        self.ax2.set_ylabel("Moisture (%)")
        self.ax2.set_ylim(0, 110)
        self.ax2.grid(True)

        # --- EKF Difference Plot (Bottom Left): Sensor1 EKF - Sensor2 EKF ---
        min_len = min(len(s1['ekf']), len(s2['ekf']))
        if min_len > 0:
            if min_len > self.window_size:
                x_diff = list(range(min_len - self.window_size + 1, min_len + 1))
                diff = [s1['ekf'][i] - s2['ekf'][i] for i in range(min_len - self.window_size, min_len)]
            else:
                x_diff = list(range(1, min_len + 1))
                diff = [s1['ekf'][i] - s2['ekf'][i] for i in range(min_len)]
            self.ax_diff.plot(x_diff, diff, color='green', label="EKF Diff (S1 - S2)")
            self.ax_diff.legend()
        self.ax_diff.set_title("EKF Difference")
        self.ax_diff.set_xlabel("Sample Index")
        self.ax_diff.set_ylabel("Difference (%)")
        self.ax_diff.grid(True)

        # --- Data Comparison Plot (Bottom Right) ---
        if raw1 and raw2:
            self.ax_comp.scatter(x1, raw1, color='orange', label="Sensor 1 Raw", s=30)
            self.ax_comp.plot(x1, ekf1, color='blue', label="Sensor 1 EKF", linewidth=2)
            self.ax_comp.scatter(x2, raw2, color='red', label="Sensor 2 Raw", s=30)
            self.ax_comp.plot(x2, ekf2, color='purple', label="Sensor 2 EKF", linewidth=2)
            self.ax_comp.legend()
        self.ax_comp.set_title("Sensor Data Comparison")
        self.ax_comp.set_xlabel("Sample Index")
        self.ax_comp.set_ylabel("Moisture (%)")
        self.ax_comp.grid(True)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.00001)  # Short pause to update the GUI

def map_raw_to_percentage(raw_value, sensor_id):
    """
    Convert raw sensor value to moisture percentage.
    Lower raw values indicate higher moisture.
    Uses sensor-specific calibration parameters and clamps the result between 0 and 100.
    """
    if sensor_id == 1:
        in_min, in_max = 200, 855
    elif sensor_id == 2:
        in_min, in_max = 250, 900
    else:
        in_min, in_max = 200, 855  # Default parameters

    percentage = 100 - ((raw_value - in_min) * 100 / (in_max - in_min))
    return max(min(percentage, 100), 0)

def main():
    # Create queues for sensor data
    sensor1_queue = queue.Queue()
    sensor2_queue = queue.Queue()

    # Create EKF instances for each sensor
    ekf1 = EKF()
    ekf2 = EKF()

    # Initialize sensor reader threads (modify port names as needed)
    sensor1_reader = SensorReader(port="COM3", sensor_id=1, data_queue=sensor1_queue)
    sensor2_reader = SensorReader(port="COM4", sensor_id=2, data_queue=sensor2_queue)

    sensor1_reader.start()
    sensor2_reader.start()

    # Initialize the visualization window with 4 subplots and a window size of 50 samples
    visualizer = CombinedVisualizer(sensor_ids=[1, 2], window_size=50)

    try:
        while True:
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

            visualizer.refresh()
    except KeyboardInterrupt:
        print("Stopping sensor readers...")
        sensor1_reader.stop()
        sensor2_reader.stop()
        sensor1_reader.join()
        sensor2_reader.join()
        print("Press Ctrl+C or close the plots to exit.")
        plt.show()

if __name__ == "__main__":
    main()

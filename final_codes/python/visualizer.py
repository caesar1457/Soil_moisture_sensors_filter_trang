import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class CombinedVisualizer:
    """
    Real-time display of 4 subplots (individual displays for 2 sensors and a comparison between the two sensors),
    supporting a scrolling window.
    
    Subplot descriptions:
      - ax1: Sensor 1's raw data and EKF estimate (do not update fill_between to avoid issues).
      - ax2: Sensor 2's raw data and EKF estimate (same as above).
      - ax_diff: Comparison of the two sensors' data (raw data and EKF data for each, using the smaller number of data points).
      - ax_comp: Comparison of the two sensors' data (same as ax_diff).
    """
    def __init__(self, sensor_ids, window_size=50):
        self.sensor_ids = sensor_ids
        self.window_size = window_size  # Each subplot displays only the latest 'window_size' data points.
        # Store sensor data: raw, ekf, ci_lower, ci_upper
        self.data = {sid: {'raw': [], 'ekf': [], 'ci_lower': [], 'ci_upper': []} for sid in sensor_ids}
        
        # Create a 2x2 subplot layout and use tight_layout for automatic spacing adjustment.
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.tight_layout()
        self.ax1 = self.axes[0, 0]  # Sensor 1
        self.ax2 = self.axes[0, 1]  # Sensor 2
        self.ax_diff = self.axes[1, 0]  # Comparison between the two sensors
        self.ax_comp = self.axes[1, 1]  # Comparison between the two sensors
        
        # ----------------- Sensor 1 subplot -----------------
        self.ax1.set_title("10mh Sensor Real-time Data")
        self.ax1.set_xlabel("Sample Index")
        self.ax1.set_ylabel("Moisture (%)")
        self.ax1.set_ylim(0, 110)
        self.ax1.grid(True)
        self.s1_line, = self.ax1.plot([], [], 'b-', linewidth=2, label='EKF Estimate')
        self.s1_scatter, = self.ax1.plot([], [], 'o', color='orange', markersize=6, label='Raw Data')
        # Do not update fill_between to ensure display stability
        # self.s1_fill = self.ax1.fill_between([0, 1], [0, 0], [0, 0], color='blue', alpha=0.2, label='Confidence Interval')
        
        # ----------------- Sensor 2 subplot -----------------
        self.ax2.set_title("50kh Sensor Real-time Data")
        self.ax2.set_xlabel("Sample Index")
        self.ax2.set_ylabel("Moisture (%)")
        self.ax2.set_ylim(0, 110)
        self.ax2.grid(True)
        self.s2_line, = self.ax2.plot([], [], 'purple', linewidth=2, label='EKF Estimate')
        self.s2_scatter, = self.ax2.plot([], [], 'o', color='red', markersize=6, label='Raw Data')
        # self.s2_fill = self.ax2.fill_between([0, 1], [0, 0], [0, 0], color='purple', alpha=0.2, label='Confidence Interval')
        
        # ----------------- ax_diff subplot -----------------
        # ----------------- ax_diff subplot: Display the difference between sensor data -----------------
        self.ax_diff.set_title("Sensor Data Difference (10mh Sensor - 50kh Sensor)")
        self.ax_diff.set_xlabel("Sample Index")
        self.ax_diff.set_ylabel("Difference in Moisture (%)")
        self.ax_diff.grid(True)
        # Create only two line objects
        self.diff_line_ekf, = self.ax_diff.plot([], [], 'b-', linewidth=2, label="EKF Difference")
        self.diff_line_raw, = self.ax_diff.plot([], [], 'r--', linewidth=2, label="Raw Data Difference")
        
        # ----------------- ax_comp subplot -----------------
        self.ax_comp.set_title("Sensor Data Comparison")
        self.ax_comp.set_xlabel("Sample Index")
        self.ax_comp.set_ylabel("Moisture (%)")
        self.ax_comp.grid(True)
        self.comp_line1, = self.ax_comp.plot([], [], 'b-', linewidth=2, label="Sensor 1 EKF")
        self.comp_scatter1, = self.ax_comp.plot([], [], 'o', color='orange', markersize=6, label="Sensor 1 Raw")
        self.comp_line2, = self.ax_comp.plot([], [], 'purple', linewidth=2, label="Sensor 2 EKF")
        self.comp_scatter2, = self.ax_comp.plot([], [], 'o', color='red', markersize=6, label="Sensor 2 Raw")
        
        # Set the initial x-axis range for all subplots
        for ax in [self.ax1, self.ax2, self.ax_diff, self.ax_comp]:
            ax.set_xlim(0, self.window_size)
        
        # Add legends to the subplots
        self.ax1.legend()
        self.ax2.legend()
        self.ax_diff.legend()
        self.ax_comp.legend()
        
        # In this implementation, disable blitting to ensure that all subplots update
        self.ani = animation.FuncAnimation(self.fig, self._update, interval=50, blit=False)
    
    def update_sensor_data(self, sensor_id, raw, ekf, ci):
        """
        Update data for the specified sensor:
          raw: Raw data (as a percentage)
          ekf: EKF-filtered estimated value
          ci: Tuple (ci_lower, ci_upper) representing the 3σ confidence interval
        """
        self.data[sensor_id]['raw'].append(raw)
        self.data[sensor_id]['ekf'].append(ekf)
        self.data[sensor_id]['ci_lower'].append(ci[0])
        self.data[sensor_id]['ci_upper'].append(ci[1])
    
    def _update(self, frame):
        """Animation update function: update the display content of each subplot"""
        # ----------------- Update Sensor 1 subplot -----------------
        s1 = self.data[1]
        n1 = len(s1['raw'])
        if n1 > 0:
            if n1 > self.window_size:
                x1 = np.arange(n1 - self.window_size + 1, n1 + 1)
                raw1 = s1['raw'][-self.window_size:]
                ekf1 = s1['ekf'][-self.window_size:]
            else:
                x1 = np.arange(1, n1 + 1)
                raw1 = s1['raw']
                ekf1 = s1['ekf']
            self.s1_line.set_data(x1, ekf1)
            self.s1_scatter.set_data(x1, raw1)
            self.ax1.set_xlim(x1[0], x1[-1])
        
        # ----------------- Update Sensor 2 subplot -----------------
        s2 = self.data[2]
        n2 = len(s2['raw'])
        if n2 > 0:
            if n2 > self.window_size:
                x2 = np.arange(n2 - self.window_size + 1, n2 + 1)
                raw2 = s2['raw'][-self.window_size:]
                ekf2 = s2['ekf'][-self.window_size:]
            else:
                x2 = np.arange(1, n2 + 1)
                raw2 = s2['raw']
                ekf2 = s2['ekf']
            self.s2_line.set_data(x2, ekf2)
            self.s2_scatter.set_data(x2, raw2)
            self.ax2.set_xlim(x2[0], x2[-1])
        
        # ----------------- Update ax_diff subplot: Display the difference between sensor data -----------------
        if n1 > 0 and n2 > 0:
            common_len = min(n1, n2, self.window_size)
            x_diff = np.arange(1, common_len + 1)
            # Calculate the difference of raw data and EKF data (Sensor1 - Sensor2)
            diff_raw = np.array(self.data[1]['raw'][-common_len:]) - np.array(self.data[2]['raw'][-common_len:])
            diff_ekf = np.array(self.data[1]['ekf'][-common_len:]) - np.array(self.data[2]['ekf'][-common_len:])
            
            # Update the two line objects with new data
            self.diff_line_ekf.set_data(x_diff, diff_ekf)
            self.diff_line_raw.set_data(x_diff, diff_raw)
            self.ax_diff.set_xlim(1, common_len)
            self.ax_diff.relim()                   # Recalculate the data limits
            self.ax_diff.autoscale_view(scalex=False, scaley=True)  # Auto-adjust the y-axis range

        # ----------------- Update ax_comp subplot -----------------
        if n1 > 0 and n2 > 0:
            common_len = min(n1, n2, self.window_size)
            x_comp = np.arange(1, common_len + 1)
            raw1_comp = self.data[1]['raw'][-common_len:]
            ekf1_comp = self.data[1]['ekf'][-common_len:]
            raw2_comp = self.data[2]['raw'][-common_len:]
            ekf2_comp = self.data[2]['ekf'][-common_len:]
            self.comp_line1.set_data(x_comp, ekf1_comp)
            self.comp_scatter1.set_data(x_comp, raw1_comp)
            self.comp_line2.set_data(x_comp, ekf2_comp)
            self.comp_scatter2.set_data(x_comp, raw2_comp)
            self.ax_comp.set_xlim(1, common_len)
            # Auto-adjust the y-axis limits
            self.ax_comp.relim()
            self.ax_comp.autoscale_view(scalex=False, scaley=True)

        # Return an empty list (because with blit=False, there is no need to return artist objects)
        return []
    
    def show(self):
        plt.show()

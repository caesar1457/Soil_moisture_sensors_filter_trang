# sensor_reader.py
import threading
import serial
import time
import queue
import re

class SensorReader(threading.Thread):
    def __init__(self, port, sensor_id, data_queue, baud_rate=9600):
        super(SensorReader, self).__init__()
        self.port = port
        self.sensor_id = sensor_id
        self.data_queue = data_queue
        self.baud_rate = baud_rate
        self.serial_conn = None
        self.stop_event = threading.Event()

    def run(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            print(f"Sensor {self.sensor_id} connected on {self.port}")
        except serial.SerialException as e:
            print(f"Error opening {self.port} for sensor {self.sensor_id}: {e}")
            return
        

        line_pattern = re.compile(r"Raw Value:\s*(\d+)")
        while not self.stop_event.is_set():
            try:
                line = self.serial_conn.readline().decode('utf-8').strip()
                if line:
                    match = line_pattern.search(line)
                    if match:
                        raw_value = int(match.group(1))
                        timestamp = time.time()
                        self.data_queue.put({'sensor_id': self.sensor_id, 'timestamp': timestamp, 'raw': raw_value, 'line': line})
                    else:
                        print(f"Sensor {self.sensor_id}: Unrecognized format: {line}")
            except Exception as e:
                print(f"Sensor {self.sensor_id} error reading line: {e}")
        if self.serial_conn:
            self.serial_conn.close()

    def stop(self):
        self.stop_event.set()

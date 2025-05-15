import sys
import serial
import serial.tools.list_ports
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QComboBox, QGroupBox,
                             QGridLayout)
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg
from collections import deque

# === Model Setup ===
MODEL_PATH = "mobilenet_fmd.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint['class_names']

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_from_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# === GUI Class ===
class ArduinoControlGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arduino & Material Classifier")
        self.setGeometry(100, 100, 1100, 900)

        self.serial_connection = None
        self.baudrate = 115200

        self.max_data_points = 500
        self.time_data = deque(maxlen=self.max_data_points)
        self.angle_data = deque(maxlen=self.max_data_points)
        self.z_data = deque(maxlen=self.max_data_points)
        self.voltage_data = deque(maxlen=self.max_data_points)

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.read_serial_data)
        self.timer.start(50)

        self.cam_timer = QTimer()
        self.cam_timer.timeout.connect(self.update_camera_prediction)
        self.cam_timer.start(1000)  # Predict every 1 second

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Could not open webcam.")

        self.time_counter = 0
        self.last_prediction = "--"

    def init_ui(self):
        self.setStyleSheet("""
            QLabel { font-size: 14px; }
            QPushButton { font-size: 14px; padding: 8px 16px; border-radius: 8px; background-color: #1976d2; color: white; }
            QPushButton:checked { background-color: #ef6c00; }
            QPushButton:hover { background-color: #1565c0; }
            QComboBox { padding: 4px; font-size: 14px; }
            QGroupBox { font-weight: bold; font-size: 15px; border: 1px solid #888; border-radius: 10px; margin-top: 10px; padding: 10px; }
        """)

        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Connection Group
        connection_group = QGroupBox("Connection Settings")
        connection_layout = QHBoxLayout()
        self.port_combo = QComboBox()
        self.refresh_ports()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        connection_layout.addWidget(QLabel("Port:"))
        connection_layout.addWidget(self.port_combo)
        connection_layout.addWidget(self.connect_btn)
        connection_group.setLayout(connection_layout)

        # Motor Control
        control_group = QGroupBox("Motor Control")
        control_layout = QHBoxLayout()
        self.button1 = QPushButton("BUTTON1 (Grip)")
        self.button1.setCheckable(True)
        self.button1.clicked.connect(self.send_button1)
        self.button2 = QPushButton("BUTTON2 (Release)")
        self.button2.setCheckable(True)
        self.button2.clicked.connect(self.send_button2)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.send_reset)
        control_layout.addWidget(self.button1)
        control_layout.addWidget(self.button2)
        control_layout.addWidget(self.reset_btn)
        control_group.setLayout(control_layout)

        # Sensor Data Display
        self.data_display_group = QGroupBox("Live Sensor Data")
        data_layout = QGridLayout()
        self.angle_label = self.create_data_label("Angle (rad):", "--")
        self.z_label = self.create_data_label("Z Magnetic Field (mT):", "--")
        self.voltage_label = self.create_data_label("Voltage (V):", "--")
        data_layout.addWidget(self.angle_label[0], 0, 0)
        data_layout.addWidget(self.angle_label[1], 0, 1)
        data_layout.addWidget(self.z_label[0], 1, 0)
        data_layout.addWidget(self.z_label[1], 1, 1)
        data_layout.addWidget(self.voltage_label[0], 2, 0)
        data_layout.addWidget(self.voltage_label[1], 2, 1)
        self.data_display_group.setLayout(data_layout)

        # Classification Display
        self.prediction_label = QLabel("Predicted Material: --")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #d32f2f; padding: 10px;")

        # Status Display
        self.status_label = QLabel("Status: Disconnected")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")

        # Plotting
        pg.setConfigOption('background', '#f0f0f0')
        pg.setConfigOption('foreground', 'black')

        plot_layout = QVBoxLayout()
        self.angle_plot = pg.PlotWidget(title="Motor Angle")
        self.angle_plot.setLabel('left', 'Angle', 'rad')
        self.angle_plot.setLabel('bottom', 'Time', 's')
        self.angle_curve = self.angle_plot.plot(pen=pg.mkPen(color='orange', width=2))

        self.z_plot = pg.PlotWidget(title="Z-axis Magnetic Field")
        self.z_plot.setLabel('left', 'Field Strength', 'mT')
        self.z_plot.setLabel('bottom', 'Time', 's')
        self.z_curve = self.z_plot.plot(pen=pg.mkPen(color='green', width=2))

        self.voltage_plot = pg.PlotWidget(title="Target Voltage")
        self.voltage_plot.setLabel('left', 'Voltage', 'V')
        self.voltage_plot.setLabel('bottom', 'Time', 's')
        self.voltage_curve = self.voltage_plot.plot(pen=pg.mkPen(color='red', width=2))

        plot_layout.addWidget(self.angle_plot)
        plot_layout.addWidget(self.z_plot)
        plot_layout.addWidget(self.voltage_plot)

        # Assemble
        main_layout.setSpacing(12)
        main_layout.addWidget(connection_group)
        main_layout.addWidget(control_group)
        main_layout.addWidget(self.data_display_group)
        main_layout.addWidget(self.prediction_label)
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(plot_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def create_data_label(self, label_text, initial_value):
        label = QLabel(label_text)
        label.setStyleSheet("font-size: 15px; font-weight: bold;")
        value = QLabel(initial_value)
        value.setStyleSheet("font-size: 16px; background: white; padding: 4px 12px; border-radius: 8px;")
        return label, value

    def refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(port.device)

    def toggle_connection(self):
        if self.serial_connection and self.serial_connection.is_open:
            self.disconnect_serial()
        else:
            self.connect_serial()

    def connect_serial(self):
        port = self.port_combo.currentText()
        if not port:
            self.status_label.setText("Status: No port selected!")
            return
        try:
            self.serial_connection = serial.Serial(port, self.baudrate, timeout=1)
            self.status_label.setText(f"Status: Connected to {port}")
            self.connect_btn.setText("Disconnect")
            self.time_data.clear()
            self.angle_data.clear()
            self.z_data.clear()
            self.voltage_data.clear()
            self.time_counter = 0
        except serial.SerialException as e:
            self.status_label.setText(f"Status: Connection failed - {str(e)}")

    def disconnect_serial(self):
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        self.serial_connection = None
        self.status_label.setText("Status: Disconnected")
        self.connect_btn.setText("Connect")

    def send_button1(self):
        if self.serial_connection and self.serial_connection.is_open:
            if self.button1.isChecked():
                self.button2.setChecked(False)
                self.serial_connection.write(b"B20\n")
                self.serial_connection.write(b"B11\n")
            else:
                self.serial_connection.write(b"B10\n")

    def send_button2(self):
        if self.serial_connection and self.serial_connection.is_open:
            self.button1.setChecked(False)
            self.serial_connection.write(b"B10\n")
            self.button2.setChecked(True)
            self.serial_connection.write(b"B21\n")
            QTimer.singleShot(1000, self.release_button2)

    def release_button2(self):
        if self.serial_connection and self.serial_connection.is_open:
            self.button2.setChecked(False)
            self.serial_connection.write(b"B20\n")

    def send_reset(self):
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.write(b"RESET\n")
            self.button1.setChecked(False)
            self.button2.setChecked(False)

    def read_serial_data(self):
        if not (self.serial_connection and self.serial_connection.is_open):
            return

        while self.serial_connection.in_waiting:
            try:
                line = self.serial_connection.readline().decode('utf-8').strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 3:
                        try:
                            angle = float(parts[0])
                            z = float(parts[1])
                            voltage = float(parts[2])

                            self.time_counter += 1
                            self.time_data.append(self.time_counter)
                            self.angle_data.append(angle)
                            self.z_data.append(z)
                            self.voltage_data.append(voltage)
                            self.update_displays(angle, z, voltage)
                            self.update_plots()
                        except ValueError:
                            pass
            except UnicodeDecodeError:
                pass

    def update_displays(self, angle, z, voltage):
        self.angle_label[1].setText(f"{angle:.2f}")
        self.z_label[1].setText(f"{z:.2f}")
        self.voltage_label[1].setText(f"{voltage:.2f}")

    def update_plots(self):
        if len(self.time_data) > 0:
            time_list = list(self.time_data)
            self.angle_curve.setData(time_list, list(self.angle_data))
            self.z_curve.setData(time_list, list(self.z_data))
            self.voltage_curve.setData(time_list, list(self.voltage_data))

    def update_camera_prediction(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                prediction = predict_from_frame(frame)
                self.last_prediction = prediction
                self.prediction_label.setText(f"Predicted Material: {prediction}")

                # Send material info to Arduino
                if self.serial_connection and self.serial_connection.is_open:
                    command = f"MATERIAL:{prediction.strip().lower()}\n"
                    self.serial_connection.write(command.encode('utf-8'))


    def closeEvent(self, event):
        self.disconnect_serial()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ArduinoControlGUI()
    window.show()
    sys.exit(app.exec_())

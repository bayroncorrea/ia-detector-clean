import sys
import cv2
import torch
import numpy as np
from PyQt6.QtWidgets import QInputDialog
from scipy.spatial import distance as dist
from collections import defaultdict, deque
from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QHBoxLayout
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from PyQt6.QtCore import Qt


class VehicleDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Vehículos - YOLOv5")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
        self.cap = None

        self.tracker = CentroidTracker()
        self.total_count = 0
        self.line_position = 300  # y position for the counting line

        self.total_count = 0
        self.previous_centroids = {}
        self.line_position = 300  # y-coordinate para la línea de conteo
        self.object_id = 0

        # Widgets
        self.image_label = QLabel("Vista de detección")
        self.image_label.setFixedSize(800, 450)  # Tamaño fijo para no cubrir toda la ventana
        self.image_label.setStyleSheet("background-color: black;")  # Opcional, para mejor visual

        self.vehicle_count_label = QLabel("Vehículos detectados: 0")

        self.btn_back = QPushButton("Atrás")
        self.btn_back.setEnabled(False)
        self.btn_back.clicked.connect(self.stop_camera_and_reset)

        self.btn_camera = QPushButton("Iniciar Cámara")
        self.btn_image = QPushButton("Cargar Imagen")
        self.btn_video = QPushButton("Cargar Video")


        # Conexiones
        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_image.clicked.connect(self.load_image)
        self.btn_video.clicked.connect(self.load_video)

        # Layouts
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_camera)
        button_layout.addWidget(self.btn_image)
        button_layout.addWidget(self.btn_video)
        button_layout.addWidget(self.btn_back)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.vehicle_count_label)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def select_camera_source(self):
        # Escanear cámaras USB disponibles (índices 0 al 5)
        available_cameras = []
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(f"USB {i}")
                cap.release()

        options = available_cameras + ["Cámara IP"]

        if not options:
            self.vehicle_count_label.setText("No se detectaron cámaras disponibles.")
            return None

        option, ok = QInputDialog.getItem(self, "Seleccionar cámara", "Elige una opción:", options, 0, False)
        if ok and option:
            if option.startswith("USB"):
                index = int(option.split(" ")[1])
                return index
            elif option == "Cámara IP":
                url, ok_url = QInputDialog.getText(self, "URL de cámara IP", "Ingresa la URL (rtsp o http):")
                if ok_url and url:
                    return url
        return None

    def start_camera(self):
        self.release_capture()
        source = self.select_camera_source()
        self.btn_back.setEnabled(True)
        if source is not None:
            self.cap = cv2.VideoCapture(source)
            if self.cap.isOpened():
                self.timer.start(30)
            else:
                self.vehicle_count_label.setText("No se pudo abrir la cámara.")

    def load_video(self):
        self.release_capture()
        self.btn_back.setEnabled(True)
        video_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar video", "", "Videos (*.mp4 *.avi)")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.timer.start(30)

    def stop_camera_and_reset(self):
        self.release_capture()
        self.image_label.setText("Vista de detección")
        self.vehicle_count_label.setText("Vehículos detectados: 0")
        self.btn_back.setEnabled(False)
        self.total_count = 0
        self.tracker = CentroidTracker()  # Reinicia el seguimiento

    def load_image(self):
        self.release_capture()
        image_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", "", "Imágenes (*.jpg *.png)")
        if image_path:
            image = cv2.imread(image_path)
            self.display_detections(image)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.display_detections(frame)
            else:
                self.timer.stop()  # Fin de video

    def display_detections(self, frame):
        results = self.model(frame)
        detections = results.pandas().xyxy[0]
        vehicle_classes = ['car', 'truck', 'bus', 'motorbike']
        vehicle_detections = detections[detections['name'].isin(vehicle_classes)]

        rects = []
        for _, det in vehicle_detections.iterrows():
            x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
            rects.append((x1, y1, x2, y2))

        objects = self.tracker.update(rects)

        for object_id, centroid in objects.items():
            cx, cy = centroid
            history = self.tracker.centroid_history[object_id]

            if len(history) >= 2:
                y_positions = [pt[1] for pt in history]
                # Verifica si cruzó la línea de arriba hacia abajo
                if y_positions[-2] < self.line_position <= y_positions[-1]:
                    if object_id not in self.tracker.counted:
                        self.total_count += 1
                        self.tracker.counted.add(object_id)

            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cv2.putText(frame, f"ID {object_id}", (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        self.vehicle_count_label.setText(f"Vehículos detectados: {self.total_count}")
        cv2.line(frame, (0, self.line_position), (frame.shape[1], self.line_position), (0, 0, 255), 2)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))


    def release_capture(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def closeEvent(self, event):
        self.release_capture()


class CentroidTracker:
    def __init__(self, max_disappeared=10, history_len=5):
        self.next_object_id = 0
        self.objects = {}  # ID -> centroid actual
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.centroid_history = defaultdict(lambda: deque(maxlen=history_len))
        self.counted = set()

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.centroid_history[self.next_object_id].append(centroid)
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        self.centroid_history.pop(object_id, None)
        self.counted.discard(object_id)

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids[i] = (cx, cy)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = dist.cdist(np.array(object_centroids), input_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.centroid_history[object_id].append(input_centroids[col])
                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(D.shape[0])) - used_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in set(range(D.shape[1])) - used_cols:
                self.register(input_centroids[col])

        return self.objects


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VehicleDetector()
    window.resize(900, 700)
    window.show()
    sys.exit(app.exec())
    window.resize(850, 600)



from ultralytics import YOLO
import os

model = YOLO('yolo11n.pt')

yaml_path = os.path.join('.', 'Yolo_data', 'data.yaml')

model.train(
    data = yaml_path,
    epochs = 50,
    imgsz = 640,
    device = 'cpu',
    project = 'DeepTrack_Results',
    name = 'Inventory_Model'
)



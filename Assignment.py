from ultralytics import YOLO
import os


weights_path = os.path.join('runs', 'detect', 'DeepTrack_Results', 'Inventory_Model', 'weights', 'best.pt')
model = None

if os.path.exists(weights_path):
    print("--- Model found ---")
    model = YOLO(weights_path)
else:
    print("--- Model not found. Start training ---")
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






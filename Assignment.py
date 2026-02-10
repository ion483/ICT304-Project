from ultralytics import YOLO
import os
import time

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


# test case id 2
test_images_path = os.path.join('Yolo_data', 'test', 'images')

start_time = time.time()

results = model.predict(source=test_images_path, conf=0.55, save=True)

end_time = time.time()

total_images = len(results)
total_time = (end_time-start_time)
avg_time = (total_time/total_images) * 1000 # convert to ms

print(f"\n--- Test case id 2 result ---")
print(f"Total Images Processed: {total_images}")
print(f"Total Inference Time: {total_time} secs")
print(f"Average Inference Time: {avg_time:.2f}ms per image")





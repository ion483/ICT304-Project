from ultralytics import YOLO
import os
import time
from datetime import datetime
import random
import csv

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


# test case id 2 (time efficiency)
test_images_path = os.path.join('Yolo_data', 'test', 'images')
test_labels_path = os.path.join('Yolo_data', 'test', 'labels')

start_time = time.time()

results = model.predict(source=test_images_path, conf=0.36, save=True)

end_time = time.time()

total_images = len(results)
total_time = (end_time-start_time)
avg_time = (total_time/total_images) * 1000 # convert to ms

print(f"\n--- Test case id 2 result ---")
print(f"Total Images Processed: {total_images}")
print(f"Total Inference Time: {total_time} secs")
print(f"Average Inference Time: {avg_time:.2f}ms per image")


# test case id 1 (accuracy)
print("\n --- Prediction Accuracy Verification ---")
# we are testing the precision accuracy, not recall accuracy
total_actual_items = 0
total_detection_items = 0
total_correct_detections = 0

for r in results:
    img_name = os.path.basename(r.path)
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_file = os.path.join(test_labels_path, label_name)

    actual_labels = []

    if os.path.exists(label_file): # getting the true labels of the individual test images
        with open(label_file, 'r') as f:
            actual_labels = [line.split()[0] for line in f.readlines()]
            total_actual_items += len(actual_labels)
    
    predicted_labels = [str(int(box.cls)) for box in r.boxes]

    for pl in predicted_labels:
        total_detection_items += 1
        if pl in actual_labels:
            total_correct_detections += 1
            actual_labels.remove(pl)

# final accuracy score
if total_detection_items > 0:
    accuracy = (total_correct_detections/total_detection_items) * 100
    print(f"Total actual items: {total_actual_items}")
    print(f"Total detections: {total_detection_items}")
    print(f"Total correct detections: {total_correct_detections}")
    print(f"Final detection accuracy: {accuracy:.2f}%")


# randomly drawing 30 images from test set for prediction

random.seed(1234)
all_test_images = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
sample_size = min(len(all_test_images), 30)
random_sample_images = random.sample(all_test_images, sample_size)
random_sample_paths = [os.path.join(test_images_path, f) for f in random_sample_images]

# run the model prediction for 30 randomly selected images
results2 = model.predict(source=random_sample_paths, conf=0.36, save=False)


ground_truth_summary = {id: 0 for id in model.names.keys()}
prediction_summary = {id: 0 for id in model.names.keys()}

for r in results2:
    img_name = os.path.basename(r.path)
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_file_path = os.path.join(test_labels_path, label_name)

    with open(label_file_path, 'r') as f:
        true_labels = [int(line.split()[0]) for line in f.readlines()]
        for label in true_labels:
            if label in ground_truth_summary:
                ground_truth_summary[label] += 1
            else:
                continue
    
    for box in r.boxes:
        pred_cls = int(box.cls)
        if pred_cls in prediction_summary:
            prediction_summary[pred_cls] += 1
        else:
            continue


print(f"Randomly selected 30 images true class summary:\n {ground_truth_summary}")
print(f"Randomly selected 30 images predicted class summary:\n {prediction_summary}")


# update CurrentStocks.csv with the prediction_summary
csv_file = "CurrentStocks.csv"
current_date = datetime.now().strftime("%m/%d/%Y %H:%M")

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(['StockCode', 'Quantity', 'Date', 'UsageDemandNextThreeDays'])
    
    for cls_id, quantity in prediction_summary.items():
        writer.writerow([cls_id, quantity, current_date, '']) # use class id as stock code

print("CurrentStocks csv is now updated")



    





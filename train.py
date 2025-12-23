from ultralytics import YOLO

# Load pretrained YOLOv8 nano model (fast to train)
model = YOLO('yolov8n.pt')

# Train the model on your dataset for 30 epochs
model.train(
    data='data.yaml',  # path to your data.yaml file (adjust if needed)
    epochs=50,
    imgsz=640,
    project='runs/train',  # where to save runs (you can customize)
    name='wildlife_detection',
    exist_ok=True  # overwrite if exists
)


import cv2
from ultralytics import YOLO
from twilio.rest import Client
from playsound import playsound
import os
import requests
from geopy.geocoders import Nominatim

ALERT_SOUND = r"C:\Users\EGMS\Desktop\proj\alert.wav"
ANIMAL_MODEL_PATH = r"C:\Users\EGMS\Desktop\proj\runs\train\wildlife_detection\weights\best.pt"
HUMAN_MODEL_PATH = r"C:\Users\EGMS\Desktop\proj\yolov8n.pt"

DANGEROUS_ANIMALS = ["0", "1"]

TWILIO_SID = "AC358a7b8722c4943ada3dc658e849216a"
TWILIO_AUTH_TOKEN = "5f0fba7ee63026a03fad42614a811984"
TWILIO_PHONE_NUMBER = "+19124831435"
USER_PHONE_NUMBER = "+918056199883"

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

def play_alert_sound():
    if os.path.exists(ALERT_SOUND):
        playsound(ALERT_SOUND)

try:
    info = requests.get("https://ipinfo.io/json", timeout=5).json()
    loc = info.get("loc")
    if loc:
        lat, lng = map(float, loc.split(","))
    else:
        lat, lng = None, None
except:
    lat, lng = None, None

geolocator = Nominatim(user_agent="wildlife_demo")

try:
    if lat and lng:
        full_location = geolocator.reverse((lat, lng), language='en')
        if full_location and full_location.address:
            location_name = full_location.address
        else:
            location_name = f"{info.get('city')}, {info.get('region')}, {info.get('country')}"
    else:
        location_name = "Unknown Location"
except:
    location_name = f"{info.get('city')}, {info.get('region')}, {info.get('country')}"


def send_sms(animal_name):
    try:
        message = client.messages.create(
            body=f" ALERT: Wild Animal detected ({animal_name})!\nLocation: {location_name}",
            from_=TWILIO_PHONE_NUMBER,
            to=USER_PHONE_NUMBER
        )
        print(f"SMS sent! SID: {message.sid}")
    except Exception as e:
        print(f"SMS Failed: {e}")

animal_model = YOLO(ANIMAL_MODEL_PATH)
human_model = YOLO(HUMAN_MODEL_PATH)
ANIMAL_CLASSES = {idx: name for idx, name in enumerate(animal_model.names)}

cap = cv2.VideoCapture(0)
alert_played = False
sms_sent_for = set()
frame_count = 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_count += 1

    if frame_count < 20:
        cv2.imshow("Wildlife Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    human_detected = False
    human_results = human_model(frame_rgb, conf=0.55, iou=0.4)
    for result in human_results:
        if hasattr(result, "boxes"):
            class_ids = result.boxes.cls.cpu().numpy()
            if any(int(cls) == 0 for cls in class_ids):  # class 0 = person
                human_detected = True
                break

    if human_detected:
        cv2.putText(frame, "Human detected - Ignored", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        alert_played = False

    else:
        animal_results = animal_model(frame_rgb, conf=0.70, iou=0.50)
        any_dangerous_detected = False

        for result in animal_results:
            if not hasattr(result, "boxes"):
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                confidence = confs[i]
                class_id = str(int(class_ids[i]))
                animal_name = ANIMAL_CLASSES.get(int(class_ids[i]), "UNKNOWN")

                
                w, h = x2 - x1, y2 - y1
                if w < 120 or h < 120:
                    continue

            
                if confidence < 0.80:
                    continue

        
                color = (0, 0, 255) if class_id in DANGEROUS_ANIMALS else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{animal_name} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        
                if class_id in DANGEROUS_ANIMALS:
                    any_dangerous_detected = True
                    if class_id not in sms_sent_for:
                        send_sms(animal_name)
                        sms_sent_for.add(class_id)

        if any_dangerous_detected and not alert_played:
            play_alert_sound()
            alert_played = True
        elif not any_dangerous_detected:
            alert_played = False

    cv2.imshow("Wildlife Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

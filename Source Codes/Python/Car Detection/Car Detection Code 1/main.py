import cv2
import time
# Opencv DNN
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

model = cv2.dnn_DetectionModel(net)

#model.setInputParams(size=(800, 600), scale=1/500)
model.setInputParams(size=(320, 320), scale=1/500)

# Load class lists
classes = []
with open("labels.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# Initialize camera
#cap = cv2.VideoCapture('http://192.168.137.201:81/stream')
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    # Get frames
    ret, frame = cap.read()
    
    # Object Detection
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200,0,50), 3)
        class_name = classes[class_id]
        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, (200,0,50), 2)
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


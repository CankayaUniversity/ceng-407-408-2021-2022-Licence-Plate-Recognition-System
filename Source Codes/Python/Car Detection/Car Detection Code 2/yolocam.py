import argparse
import glob
import time
from pathlib import Path
import cv2
import numpy as np

CONFIDENCE_THRESHOLD =0.5
NMS_THRESHOLD =0.4

#vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vc = cv2.VideoCapture('http://192.168.137.201:81/stream')


weights = glob.glob("yolo/*.weights")[0]
labels = glob.glob("yolo/*.txt")[0]
cfg = glob.glob("yolo/*.cfg")[0]

class_names = list()
with open(labels, "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

COLORS = np.random.randint(0, 255, size=(len(class_names), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer = net.getLayerNames()

layer = [layer[i - 1] for i in net.getUnconnectedOutLayers()]
#layer = [layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect(frm, net, ln):
    (H, W) = frm.shape[:2]
    blob = cv2.dnn.blobFromImage(frm, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start_time = time.time()
    layerOutputs = net.forward(ln)
    end_time = time.time()

    boxes = []
    classIds = []
    confidences = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                classIds.append(classID)
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIds[i]]]
            cv2.rectangle(frm, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(class_names[classIds[i]], confidences[i])
            cv2.putText(
                frm, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
            )

            fps_label = "FPS: %.2f" % (1 / (end_time - start_time))
            cv2.putText(
                frm, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
            )


while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        break
    frame = cv2.resize(frame, (600,480))
    detect(frame, net, layer)

    cv2.imshow("detections", frame)


cv2.destroyAllWindows()

import keyboard
import cv2

capture = cv2.VideoCapture(0)
capture.set(3, 2200)
capture.set(4, 1000)


with open('resources/coco.names', 'r') as file:
    classNames = []
    classNames = file.read().rstrip('\n').split('\n')

config = 'resources/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights = 'resources/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weights, config)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:
    success, data = capture.read()

    classIds, confs, bounding_box = net.detect(data, confThreshold=0.5)
    print(classIds, bounding_box)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bounding_box):
            cv2.rectangle(data, box, color=(0, 251, 255), thickness=1)

            cv2.putText(data, classNames[classId-1], (box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 89), 1)
            
            cv2.putText(data, str(round(confidence*100, 1)), (box[0]+150, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 157, 0), 1)

    cv2.imshow('Object detection', data)

    cv2.waitKey(1)

    if keyboard.is_pressed('backspace') or keyboard.is_pressed('home'):
        exit()

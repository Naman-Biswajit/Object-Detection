import cv2

# data = cv2.imread('resources/test.jpg')

capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)


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
            cv2.rectangle(data, box, color=(245, 44, 44), thickness=1)
            cv2.putText(data,`` classNames[classId-1], (box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (94, 255, 105), 1)

    cv2.imshow('Image', data)
    print(cv2.waitKey(8))
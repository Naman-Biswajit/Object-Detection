import cv2

data = cv2.imread('test.jpg')

with open('coco.names', 'r') as file:
    classNames = []
    classNames = file.read().rstrip('\n').split('\n')

config = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weights, config)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, confs, bounding_box = net.detect(data, confThreshold=0.5)
print(classIds, bounding_box)

for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bounding_box):
    cv2.rectangle(data, box, color=(245, 44, 44), thickness=1)
    cv2.putText(data, classNames[classId-1], (box[0]+10, box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX, 1, (94, 255, 105), 1)

cv2.imshow('Image', data)
cv2.waitKey(0)

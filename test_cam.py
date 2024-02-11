import cv2, main_Yolov7, imutils

cam = cv2.VideoCapture(0)
# save images / save txt / print resutls
yolov7 = main_Yolov7.Yolo_v7(False, False, True)

yolov7.inicialize()

while (cam.isOpened()):
     _, img = cam.read()
     #img = imutils.resize(img, width=640)
     img = yolov7.detect(img, False)
     img = imutils.resize(img, width=1080)
     cv2.imshow("Video", img)
     cv2.waitKey(1)


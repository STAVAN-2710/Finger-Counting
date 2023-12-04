import cv2 as cv
import numpy as np
import time 
import hand_tracking_module as htm
import os

width_cam , height_cam = 640, 480

cap = cv.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)

folder_path = 'FINGER'
my_list = os.listdir(folder_path)
# print(my_list)

overlay_list = []
for imPath in my_list :
    image = cv.imread(f'{folder_path}/{imPath}')
    overlay_list.append(image)

p_time = 0
c_time = 0

detector = htm.handDetector(detect_con= 0.7)
tip_ID = [4, 8, 12 ,16, 20]


while 1 :
    _, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw= False)
    # print(lm_list)
    if len(lm_list) != 0 :
        fingers = []

        # for the right hand thumb we se if the point below is on the left.. then it it is up 
        if lm_list[tip_ID[0]][1] > lm_list[tip_ID[0] - 1][1] :
            fingers.append(1)
        else :
            fingers.append(0)

        # for the fingers we see the tip and the point below it n compare the pos.(cy)
        for id in range(1,5) :
            if lm_list[tip_ID[id]][2] < lm_list[tip_ID[id] - 2][2] :
                fingers.append(1)
            else :
                fingers.append(0)

        # print(fingers) 
        total_fingers = fingers.count(1)
        print(total_fingers)         
        # h, w, c = overlay_list[total_fingers - 1].shape
        # img[0:h, 0:w] = overlay_list[total_fingers - 1]
        cv.rectangle(img, (5, 150), (45, 250), (100,100,255), -1)
        cv.putText(img, str(int(total_fingers)), (10, 200), cv.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv.putText(img, str(int(fps)), (10,50), cv.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)

    # h, w, c = overlay_list[0].shape
    # img[0:h, 0:w] = overlay_list[0]


    cv.imshow('img', img)
    if cv.waitKey(2) == ord('q') :
        break



import cv2
import mediapipe as mp
import time
import numpy as np


cap = cv2.VideoCapture(0)

#myhands= mp.solutions.hands
mphand = mp.solutions.hands
hands = mphand.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
#list = []


while True:
    blank = np.zeros((480, 640, 3), dtype='uint8')
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                #list.append([id, cx, cy])
                #print(list)
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                    #cv2.line(blank, (list[0][1], list[0][2]), (list[0][1], list[0][2]), (255, 0, 255), 3)
                    #cv2.imshow('blank', blank)
                #print("INNERlen :",len(list))
                
                #print("inner list",list


            mpDraw.draw_landmarks(img,handLms,mphand.HAND_CONNECTIONS)
            

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


    cv2.imshow('video',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()

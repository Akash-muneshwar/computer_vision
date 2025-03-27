import cv2
import mediapipe as mp
import time
import numpy as np


cap = cv2.VideoCapture(0)


mphand = mp.solutions.hands
hands = mphand.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
#list = []
track =[]


while True:
    imgn = np.zeros((500, 500, 3), dtype=np.uint8)
    blank = np.zeros((480, 640, 3), dtype='uint8')
    success, img = cap.read()
    img = cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        #imgn = np.zeros((500, 500, 3), dtype=np.uint8)
        for handLms in results.multi_hand_landmarks:
            
            for id, lm in enumerate(handLms.landmark):
                print(id, lm)
                h, w, c = img.shape
                #print(img.shape)
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                if id == 8 and len(results.multi_hand_landmarks)==1:
                    track.append([cx,cy])

            
            mpDraw.draw_landmarks(img,handLms,mphand.HAND_CONNECTIONS)
            #print("outer" ,track)
            for i in range(3,len(track)):       #starting from 3 to let the index finger placed in correct position
                    if i==3:
                        continue
                    cv2.line(imgn,(track[i-1][0],track[i-1][1]),(track[i][0],track[i][1]),(255,0,255),3)
                      

    else:
        print("None")
        track = []
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("img",imgn)
    cv2.imshow('video',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

import math
import cv2
import mediapipe as mp
import time


#--------------------------------------------------Create the Class--------------------------------------------------
class HandTracking():
    #--------------------------------------------------Initialize the Class--------------------------------------------------
    def __init__(self, mode=False, maxHands=2,model_complexity=1, Confdetection=0.5, Confsegui=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.Confdetection = Confdetection
        self.Confsegui = Confsegui


    #--------------------------------------------------Create the object to detect and draw hands-----------------------------
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.Confdetection, self.Confsegui)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]


    #--------------------------------------------------Create the function to detect the hands-----------------------------
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    #--------------------------------------------------Create the function to find the position of the hands-----------------------------
    def findPosition(self, img, NumHands = 0, DrawPoints = True, DrawBox = True, color = []):
        xList = []
        yList = []
        bBox = []
        player = 0
        self.list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[NumHands]
            test = self.results.multi_hand_landmarks
            player = len(test)
            #print(player)	
            for id, lm in enumerate(myHand.landmark):
                height, width, c = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                xList.append(cx)
                yList.append(cy)
                self.list.append([id, cx, cy])
                if DrawPoints:
                    cv2.circle(img, (cx, cy), 3, (0, 0, 0), cv2.FILLED)
            
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bBox = xmin, ymin, xmax, ymax

            if DrawBox:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), color, 2)
        return self.list, bBox, player
    
    #--------------------------------------------------Create the function to find and draw the fingers up-----------------------------
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.list[self.tipIds[0]][1] > self.list[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.list[self.tipIds[id]][2] < self.list[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    
    #--------------------------------------------------Function to detect the distance between fingers-----------------------------

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.list[p1][1:]
        x2, y2 = self.list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
    
    #--------------------------------------------------Main Function--------------------------------------------------

def main():

    ptiempo = 0
    ctiempo = 0
    cap = cv2.VideoCapture(0)
    detector = HandTracking()

    while True:
        ret, img = cap.read()
        img = detector.findHands(img)
        list, bBox = detector.findPosition(img)
        #if len(list) != 0:
        #    print(list[4])

        ctiempo = time.time()
        fps = 1 / (ctiempo - ptiempo)
        ptiempo = ctiempo

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Hands", img)
        
        k = cv2.waitKey(1)

        if k == 27:
            break

        cap.release()
        cv2.destroyAllWindows()

    
import cv2 as cv
import mediapipe as mp
import time

# self,
# static_image_mode=False,
# max_num_hands=2,
# min_detection_confidence=0.5,
# min_tracking_confidence=0.5

class handDetector():

    def __init__(self, mode = False, max_hands = 2, detect_con = 0.5, track_con = 0.5 ) :
        self.mode = mode
        self.max_hands = max_hands
        self.detect_con = detect_con
        self.track_con = track_con

        self.mp_Hands = mp.solutions.hands
        self.hands = self.mp_Hands.Hands(self.mode, self.max_hands,self.detect_con, self.track_con)
        self.mp_Draw = mp.solutions.drawing_utils #this directly helps us to draw lines b/w the landmarks detected

        self.tip_ID = [4, 8, 12 ,16, 20]

    def find_hands(self, img, draw = True) :

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)      #the object hands take s onLy RGB format
        self.results = self.hands.process(imgRGB)             #this PROCESS does everything for us .. we only have to extract the info

        # print(results) this only prints <class 'mediapipe.python.solution_base.SolutionOutputs'>
        # print(results.multi_hand_landmarks)         #returns the coordinates of the landmarks whenever it detects them

        if self.results.multi_hand_landmarks :                #if hands are detected 
            for handLms in self.results.multi_hand_landmarks :    
                if draw :
                    # mp_Draw.draw_landmarks(img, handLms )  #  draw the landmarks on the main image(not RGB)
                     self.mp_Draw.draw_landmarks(img, handLms, self.mp_Hands.HAND_CONNECTIONS )    #draw the landmarks and connects them in main image

        return img

    def find_position(self, img, hand_No = 0, draw = True) :

        self.lm_list = []

        if self.results.multi_hand_landmarks :  
            my_hand = self.results.multi_hand_landmarks[hand_No]    

            for id, lm in enumerate(my_hand.landmark) :
                #print(id, lm)    #lm gives the x, y, z in scale of 0 to 1 .. mult. them with image dimension to get pixel values
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                self.lm_list.append([id, cx, cy])
                # if id == 4:
                if draw :
                    cv.circle(img, (cx, cy), 10, (0,255,255), -1)               

        return self.lm_list

    def finger_up(self) :

        self.fingers = []

        # for the right hand thumb we se if the point below is on the left.. then it it is up 
        if self.lm_list[self.tip_ID[0]][1] > self.lm_list[self.tip_ID[0] - 1][1] :
            self.fingers.append(1)
        else :
            self.fingers.append(0)

        # for the fingers we see the tip and the point below it n compare the pos.(cy)
        for id in range(1,5) :
            if self.lm_list[self.tip_ID[id]][2] < self.lm_list[self.tip_ID[id] - 2][2] :
                self.fingers.append(1)
            else :
                self.fingers.append(0)

        return self.fingers



def main() :
    # for FPS
    p_time = 0 #previous time
    c_time = 0 #current time

    cap = cv.VideoCapture(0)
    detector = handDetector()

    while 1 :
        _, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)

        if len(lm_list) != 0 :
            print(lm_list[4])

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv.putText(img, str(int(fps)), (10,50), cv.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)
            
        cv.imshow('image', img)
        if cv.waitKey(1) == ord('q') :
            break





if __name__ == "__main__" :
    main()
import cv2
import mediapipe as mp
import math
import time
import os
import sys
print(mp.__file__)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


password_sequence = ["one", "two", "three"]
max_password_length = 5
entered = []

hand_found = False
gesture_start_time = None
gesture_time = 2
draw_circles = True

capture = cv2.VideoCapture(0)
results = None
img = None
file_path = "H:\Videos\Hand\code\hand-gesture-password\Secret.txt"

index_extended = False
middle_extended = False
ring_extended = False
pinky_extended = False
thumb_extended = False

handedness = None #left / right



def draw_circle(img, landmark):
    img_h, img_w, _ = img.shape

    cx, cy = int(landmark.x * img_w), int(landmark.y * img_h)

    #draw a circle
    cv2.circle(img, (cx, cy), radius=5, color=(255, 0, 0), thickness=cv2.FILLED)


def check_fingers_extended():
    global index_extended, middle_extended, ring_extended, pinky_extended, thumb_extended
    index_extended = is_finger_extended(firstHand, 8,6)
    middle_extended = is_finger_extended(firstHand, 12,10)
    ring_extended = is_finger_extended(firstHand, 16,14)
    pinky_extended = is_finger_extended(firstHand, 20,18)
    thumb_extended = is_thumb_extended(firstHand)

def is_finger_extended(hand, tip, pip):
    return hand.landmark[tip].y < hand.landmark[pip].y


def is_thumb_extended(hand):
    if handedness == "Left":
        if hand.landmark[4].y < hand.landmark[6].y:
            return hand.landmark[4].x < hand.landmark[3].x
        else:
            return hand.landmark[4].x > hand.landmark[3].x
    elif handedness == "Right":
        if hand.landmark[4].y < hand.landmark[6].y:
            return hand.landmark[4].x > hand.landmark[2].x
        else:
            return hand.landmark[4].x < hand.landmark[3].x

def peace_gesture(hand):
    return (index_extended
            and middle_extended
            and not ring_extended
            and not pinky_extended)

def fist_gesture(hand):
    return (not     index_extended
            and not middle_extended
            and not ring_extended
            and not pinky_extended
            and not thumb_extended)

def ok_gesture(hand):
    return(distance_3d_normalized(hand.landmark[8],hand.landmark[4]) <= 0.05)

def thumbs_up_gesture(hand):
    return(not      index_extended
            and not middle_extended
            and not ring_extended
            and not pinky_extended
            and     thumb_extended)

def one_gesture(hand):
    return(index_extended
            and not middle_extended
            and not ring_extended
            and not pinky_extended)

def three_gesture(hand):
    return(index_extended
            and middle_extended
            and ring_extended
            and not pinky_extended)

def four_gesture(hand):
    return(index_extended
            and middle_extended
            and ring_extended
            and pinky_extended
            and not thumb_extended)

def five_gesture(hand):
    return(index_extended
            and middle_extended
            and ring_extended
            and pinky_extended
            and thumb_extended)


def check_gesture(hand):
    for name, func in gestures:
        if func(hand):
            return name
    return None

def open_file(path):
    os.startfile(path)

        
gestures = [

    ("fist", fist_gesture),
    ("thumbs_up", thumbs_up_gesture),
    ("ok", ok_gesture),
    ("one", one_gesture),
    ("two", peace_gesture),
    ("three", three_gesture),
    ("four", four_gesture),
    ("five", five_gesture)

]

def distance_3d_normalized(lm1, lm2):
    dx = lm2.x - lm1.x
    dy = lm2.y - lm1.y
    dz = lm2.z - lm1.z
    return math.sqrt(dx*dx + dy*dy + dz*dz) #z is guessed by mediapipe

def handle_capture():
    global results, img, firstHand
    success, img = capture.read()

    if not success:
        return False

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)

    
    if results.multi_hand_landmarks:
        firstHand = results.multi_hand_landmarks[0]
        if draw_circles:
            for landmark in firstHand.landmark: #draw circles
                draw_circle(img, landmark)

    img_resized = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)
    cv2.imshow("Image", img_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):  #quits
        return False

    return True


print("Please hold up hand")
while True:

    if not handle_capture(): #captures video and sets img / result
        break

    if not results.multi_hand_landmarks:  #skip to next frame if no hand
        gesture_start_time = None  #reset if hand is lost
        continue  

    if not hand_found:
        print("hand found! ")
        hand_found = True
        os.system('cls')
        print(f"Please perform gesture " + str(1))
        gesture_start_time = None #new prompt
        

 
    check_fingers_extended()
    handedness  = results.multi_handedness[0].classification[0].label  #left or right

    detected_gesture = check_gesture(firstHand)

    

    if detected_gesture:
        if gesture_start_time is None:
            gesture_start_time = time.time()
            last_gesture = detected_gesture
        elif detected_gesture == last_gesture:

            if time.time() - gesture_start_time >= gesture_time:
                entered.append(detected_gesture)
                os.system('cls')

                if entered == password_sequence:
                    if file_path:
                        open_file(file_path)
                    else:
                        print("No file path provided.")
                    break

                if len(entered) >= max_password_length:
                    print("WRONG")
                    break
                print(entered)
                print(f"Please perform gesture " + str(len(entered)+1))
                gesture_start_time = None  #reset for next time

        else:
            #gesture changed early
            gesture_start_time = time.time()
            last_gesture = detected_gesture
    else:
        gesture_start_time = None  #no gesture detected



    

capture.release()

cv2.destroyAllWindows()


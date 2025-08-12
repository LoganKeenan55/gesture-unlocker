import cv2
import mediapipe as mp
import math
import time
from cryptography.fernet import Fernet
import tempfile
import os
import numpy as np
print(mp.__file__)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)



# == File path to encryped file ==
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Secret.enc")
# hard code if you're into that -> file_path = "H:\Secret.enc" 


# == Secret, don't look! avert thine eyes!! ==
password_sequence = ["one", "two", "three"] #gestures
key = "8Tp5ZKq0o_lXASmSZNnu-9HL22YzdbqEi8XbICNeDFU=" #ecryption key

# == Color of background, set to "None" to see camera ==
use_background_color = None #(33, 26, 26)

max_password_length = 5
entered = []

fernet = Fernet(key)

hand_found = False
gesture_start_time = None
gesture_time = 2
draw_circles = True


capture = cv2.VideoCapture(0)
results = None
img = None


index_extended = False
middle_extended = False
ring_extended = False
pinky_extended = False
thumb_extended = False

handedness = None #left / right



def draw_circle(img, landmark, r, c):
    img_h, img_w, _ = img.shape

    cx, cy = int(landmark.x * img_w), int(landmark.y * img_h)

    #draw a circle
    cv2.circle(img, (cx, cy), radius=r, color=c, thickness=cv2.FILLED)


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
            return hand.landmark[4].x > hand.landmark[3].x
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
    f = open(path, "rb")
    encrypted_data = f.read()
    f.close()

    decrypted_data = fernet.decrypt(encrypted_data)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tmp.write(decrypted_data)
    tmp_path = tmp.name
    tmp.close()

    os.startfile(tmp_path)
        
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

    img_h, img_w, _ = img.shape
    #color to block camera
    if use_background_color:
        background_bgr = (33, 26, 26)
        img = np.full((img_h, img_w, 3), background_bgr, dtype=np.uint8)

    if results.multi_hand_landmarks:
        firstHand = results.multi_hand_landmarks[0]

        #connections
        for connection in mpHands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = firstHand.landmark[start_idx]
            end = firstHand.landmark[end_idx]
            x1, y1 = int(start.x * img_w), int(start.y * img_h)
            x2, y2 = int(end.x * img_w), int(end.y * img_h)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #draw circles
        for landmark in firstHand.landmark:
            draw_circle(img, landmark, 5, (255, 0, 0))
    
        if index_extended:
            draw_circle(img, firstHand.landmark[8], 5, (0, 0, 255))  #
        if middle_extended:
            draw_circle(img, firstHand.landmark[12], 5, (0, 0, 255)) #
        if ring_extended:
            draw_circle(img, firstHand.landmark[16], 5, (0, 0, 255)) # tips of fingers
        if pinky_extended:
            draw_circle(img, firstHand.landmark[20], 5, (0, 0, 255)) # 
        if thumb_extended:
            draw_circle(img, firstHand.landmark[4], 5, (0, 0, 255))  # 
    img_resized = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)
    cv2.imshow("Image", img_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
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


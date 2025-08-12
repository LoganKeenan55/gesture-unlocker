gesture-unlock

A Python-based hand gesture password system using MediaPipe and OpenCV. 

## Features

- Real-time hand tracking and gesture recognition using MediaPipe  
- Supports multiple gestures: fist, thumbs up, OK, one, two , three, four, five  
- Customizable gesture password sequences  

Use the "encrypt_file.py" script to encrypt a file with Fernet, this will give you a key that you can put inside "gesture_unlocker.py". Set the sequence gesture sequence, and run the script when you want to view your secret file.

```bash
pip install opencv-python mediapipe cryptography

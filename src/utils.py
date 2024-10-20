import cv2

def preprocess_frame(frame):
    return cv2.resize(frame, (84, 84)).transpose((2, 0, 1))

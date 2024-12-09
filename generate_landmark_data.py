import csv
import os
from tkinter import messagebox

import cv2
import mediapipe as mp
import numpy as np
import yaml


def is_keyboard_char(char: str):
    return isinstance(char, str) and len(char) == 1 and char.isprintable()


def label_dict_from_config_file(relative_path):
    with open(relative_path, "r") as f:
        label_tag = yaml.full_load(f)["gestures"]
    return label_tag


class HandDatasetWriter:
    def __init__(self, filepath) -> None:
        self.csv_file = open(filepath, "a")
        self.file_writer = csv.writer(self.csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    def add(self, hand, label):
        self.file_writer.writerow([label, *np.array(hand).flatten().tolist()])

    def close(self):
        self.csv_file.close()


class HandLandmarksDetector:
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(False, max_num_hands=1, min_detection_confidence=0.5)

    def detect_hand(self, frame):
        hands = []
        frame = cv2.flip(frame, 1)
        annotated_image = frame.copy()
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = []
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                for landmark in hand_landmarks.landmark:
                    x, y, z = landmark.x, landmark.y, landmark.z
                    hand.extend([x, y, z])
            hands.append(hand)
        return hands, annotated_image


def open_error_dialog():
    messagebox.showinfo("Input Error", "Please press a character again")


def handle_input(key, recording, label, saved_frame, sign_img_path, label_tag):
    status_text = None

    if key == ord('q') or key == 27:  # Exit conditions
        recording = False
        return "press a character to record", saved_frame, recording

    if key != -1:  # Valid key pressed
        key = chr(key)
        if is_keyboard_char(key):
            recording = True
            status_text = f"Recording {label_tag[label]}, press {key} again to stop"
            if saved_frame is not None and label >= 0:
                cv2.imwrite(f"./{sign_img_path}/{label_tag[label]}.jpg", saved_frame)
            saved_frame = None
        else:
            open_error_dialog()

    return status_text, saved_frame, recording


def run_camera(data_path, sign_img_path, label_tag, split="val", resolution=(1280, 720)):
    hand_detector = HandLandmarksDetector()
    cam = cv2.VideoCapture(0)
    cam.set(3, resolution[0])
    cam.set(4, resolution[1])

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(sign_img_path, exist_ok=True)
    print(sign_img_path)
    dataset_path = f"./{data_path}/landmark_{split}.csv"
    hand_dataset = HandDatasetWriter(dataset_path)

    # Init data
    label = 0
    recording = False
    status_text = "press a character to record"
    saved_frame = None

    while cam.isOpened():
        _, frame = cam.read()
        hands, annotated_image = hand_detector.detect_hand(frame)

        key = cv2.waitKey(1)

        if recording and hands:
            hand = hands[0]
            hand_dataset.add(hand=hand, label=label)
            saved_frame = frame

        new_status_text, saved_frame, recording = handle_input(key, recording, label,
                                                               saved_frame, sign_img_path, label_tag)
        status_text = new_status_text if new_status_text is not None else status_text

        cv2.putText(annotated_image, status_text, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(f"{split}", annotated_image)

    cam.release()
    cv2.destroyAllWindows()


def run():
    LABEL_TAG = label_dict_from_config_file("hand_gesture.yaml")
    data_path = './data2'
    sign_img_path = './sign_imgs2'

    run_camera(data_path, sign_img_path, LABEL_TAG, "train", (1280, 720))
    run_camera(data_path, sign_img_path, LABEL_TAG, "val", (1280, 720))
    run_camera(data_path, sign_img_path, LABEL_TAG, "test", (1280, 720))

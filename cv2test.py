#!/usr/bin/env python3

import numpy as np
import cv2
import dispvideo as dv

def convert_to_bgr(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
def convert_to_gray(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
def threshold(x, t):
    return cv2.threshold(x, t, 255, cv2.THRESH_BINARY)

x = np.load("npdata/vids/cam1_1.npy")
length, W, H, _ = x.shape
print("Number of frames:", length)

tracker = cv2.TrackerCSRT_create()
initBB = None

first = True

x_positions = np.full((length,), np.NaN)
y_positions = np.full((length,), np.NaN)

for i, frame in enumerate(x):
    # For some reason, doing this for each frame works
    # but doing something like "np.flip" fails
    frame = convert_to_gray(frame)
    _, frame = threshold(frame, 250)
    if first:
        cv2.imshow("test", frame)
        first = False

    if not initBB:
        initBB = cv2.selectROI("test", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)

    success, box = tracker.update(frame)
    
    if success:
        x, y, w, h = [int(v) for v in box]
        center_x = x + 0.5*w
        center_y = y + 0.5*h
        # NaN's otherwise
        x_positions[i] = center_x
        y_positions[i] = center_y
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    info = [("Success", "Yes" if success else "No")]

    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i*20)+20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("test", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
print("Number of failed measurements: {}".format(np.isnan(x_positions).sum()))

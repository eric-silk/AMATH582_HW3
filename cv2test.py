#!/usr/bin/env python3

import numpy as np
import cv2

x = np.load("npdata/cam1_1.npy")
length, W, H, _ = x.shape
print("Number of frames:", length)

tracker = cv2.TrackerKCF_create()
initBB = None

for frame in x:

    cv2.imshow("test", frame)

    if not initBB:
        initBB = cv2.selectROI("test", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)

    if initBB:
        success, box = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        info = [("Success", "Yes" if success else "No")]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i*20)+20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(text)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break


cv2.destroyAllWindows()

#!/usr/bin/env python3

import numpy as np
import cv2

x = np.load("npdata/cam1_1.npy")
print("x shape:", x.shape)


print("Press any key to exit...")
for frame in x:
    cv2.imshow("test", frame)
    cv2.waitKey(20)

cv2.destroyAllWindows()

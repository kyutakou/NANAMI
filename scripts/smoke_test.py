import sys

print("Python:", sys.version)
print("Interpreter:", sys.executable)

import numpy as np
import cv2
import onnxruntime as ort
import scipy

print("numpy:", np.__version__)
print("opencv:", cv2.__version__)
print("onnxruntime:", ort.__version__)
print("scipy:", scipy.__version__)
print("ONNX providers:", ort.get_available_providers())

# Tiny functional check: make a blank frame and resize it
frame = np.zeros((720, 1280, 3), dtype=np.uint8)
small = cv2.resize(frame, (416, 416))
print("resize OK:", small.shape)
print("SMOKE TEST PASSED")
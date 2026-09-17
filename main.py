import numpy as np
import cv2 as cv
import cameraFinder as cF
import modelHandler as mH

print(cF.findCamera())

camIndex = int(input("Enter camera index: "))
if type(camIndex) != int:
    raise TypeError("Camera index is not of type Integer")

cap = cv.VideoCapture(camIndex) # Opens the camera at entered index
if not cap.isOpened(): # Checks if you can actually open the camera
    print("Cannot open camera object")
    raise TypeError("Camera object cannot open")

print("--Z to exit--")
while True:
    ret, frame = cap.read()

    if not ret: # Checks for frames
        print("Cannot find new frame. Camera disconnect?")
        break

    # Match the training pipeline: resize with the same interpolation. The
    # detector converts this OpenCV BGR frame to RGB and float32 internally.
    frame = cv.resize(frame, (640, 640), interpolation=cv.INTER_AREA)

    prediction = mH.detect(frame)
    print(prediction)
    detections = np.asarray(prediction)
    if detections.ndim == 1:
        detections = detections.reshape(1, -1)

    frameHeight, frameWidth = frame.shape[:2]
    for detection in detections:
        if detection.size < 5:
            continue

        # Prediction format: x_center, y_center, width, height, confidence
        xCenter, yCenter, boxWidth, boxHeight, confidence = detection[:5]
        xCenter *= frameWidth
        yCenter *= frameHeight
        boxWidth *= frameWidth
        boxHeight *= frameHeight

        x1 = int(xCenter - boxWidth / 2)
        y1 = int(yCenter - boxHeight / 2)
        x2 = int(xCenter + boxWidth / 2)
        y2 = int(yCenter + boxHeight / 2)

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{float(confidence):.2f}"
        cv.putText(frame, label, (x1, max(y1 - 8, 0)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv.imshow("frame", frame) # Apparently doing ("frame", frame) makes it go into normal color instead of papa smurf
    if cv.waitKey(1) == ord("z"): # Quits if "z" is pressed
        break

# Releases memory and gets rid of all windows
cap.release()
cv.destroyAllWindows()
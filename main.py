import numpy as np
import cv2 as cv
import cameraFinder as cF

print(cF.findCamera())

camIndex = int(input("Enter camera index: "))
if type(camIndex) != int:
    raise TypeError("Camera index is not of type Integer")

cap = cv.VideoCapture(camIndex) # Opens the camera at entered index
if not cap.isOpened(): # Checks if you can actually open the camera
    print("Cannot open camera object")
    raise TypeError("Camera object cannot open")

while True:
    ret, frame = cap.read()

    if not ret: # Checks for frames
        print("Cannot find new frame. Camera disconnect?")
        break

    colorScale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # Converts viewed video to whatever color format

    cv.imshow("frame", frame) # Apparently doing ("frame", frame) makes it go into normal color instead of papa smurf

    if cv.waitKey(1) == ord("z"): # Quits if "z" is pressed
        break

# Releases memory and gets rid of all windows
cap.release()
cv.destroyAllWindows()
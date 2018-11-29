import numpy as np
import cv2

video = cv2.VideoCapture("SourceVideo/DrivingCropped.mp4")


def ShowGrayscale(clip):
    while (clip.isOpened()):
        ret, frame = clip.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', gray)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

#ROI = [(0,height), (width / 2, height / 2), (width, height)]

ShowGrayscale(video)

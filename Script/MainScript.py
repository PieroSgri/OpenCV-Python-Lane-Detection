import numpy as np
import cv2

video = cv2.VideoCapture("SourceVideo/DrivingCropped.mp4")
v_height = video.get(cv2.CAP_PROP_FRAME_WIDTH)
v_width = video.get(cv2.CAP_PROP_FRAME_HEIGHT)


def ShowGrayscale(clip):
    while(clip.isOpened()):
        ret, frame = clip.read()

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', grayscale)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def ApplyGrayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def ROI(clip):
    while (clip.isOpened()):
        ret, frame = clip.read()

        try:
            lower_left = [frame.shape[1] / 8, frame.shape[0]]
            lower_right = [frame.shape[1] - frame.shape[1] / 8, frame.shape[0]]
            top_left = [frame.shape[1] / 2 - frame.shape[1] / 6, frame.shape[0] / 2 + frame.shape[0] / 12]
            top_right = [frame.shape[1] / 2 + frame.shape[1] / 6, frame.shape[0] / 2 + frame.shape[0] / 12]
            vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
        except AttributeError:
            break

        # Define a blank matrix that matches the image height/width.
        mask = np.zeros_like(frame)

        # Retrieve the number of color channels of the image.
        if len(frame.shape) > 2:
            channel_count = frame.shape[2]
            # Create a match color with the same color channel counts.
            match_mask_color = (255,) * channel_count
        else:
            match_mask_color = 255

        # Fill inside the polygon
        cv2.fillPoly(mask, vertices, match_mask_color)

        # Returning the image only where mask pixels match
        frame = cv2.bitwise_and(frame, mask)

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', grayscale)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



#ShowGrayscale(video)
ROI(video)
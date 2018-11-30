import numpy as np
import cv2

# Passing the video as a stream, for a live stream from a camera the parameter will be the devices ID
video = cv2.VideoCapture("SourceVideo/DrivingCropped.mp4")

# Useless at the moment, clip.read() is not launched yet so the height and width will be zero
v_height = video.get(cv2.CAP_PROP_FRAME_WIDTH)
v_width = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Basic cv2 operations to show the video in grayscale
def ShowGrayscale(clip):
    while(clip.isOpened()):
        ret, frame = clip.read()

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', grayscale)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


# Helper function, just in case...
def ApplyGrayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def ROI(clip):
    while (clip.isOpened()):
        ret, frame = clip.read()

        # We define the region of interest, those value are obtained with a bit of trial and error to find best results
        # try/except needed as clip.read() will return a None type to mark the end of the video crashing the script
        try:
            lower_left = [frame.shape[1] / 8, frame.shape[0]]
            lower_right = [frame.shape[1] - frame.shape[1] / 8, frame.shape[0]]
            top_left = [frame.shape[1] / 2 - frame.shape[1] / 6, frame.shape[0] / 2 + frame.shape[0] / 12]
            top_right = [frame.shape[1] / 2 + frame.shape[1] / 6, frame.shape[0] / 2 + frame.shape[0] / 12]
            vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]

        except AttributeError:
            break

        # Define a matrix of zeros that matches the image height/width
        mask = np.zeros_like(frame)

        # Retrieve the number of color channels of the image
        if len(frame.shape) > 2:
            channel_count = frame.shape[2]
            # Create a match color with the same color channel counts
            match_mask_color = (255,) * channel_count
        else:
            match_mask_color = 255

        # Fill inside the polygon ???
        cv2.fillPoly(mask, vertices, match_mask_color)

        # Returning the image only where mask pixels match to cut out the unnecessary region
        frame = cv2.bitwise_and(frame, mask)

        # Transform current frame into grayscale
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', grayscale)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



#ShowGrayscale(video)
ROI(video)
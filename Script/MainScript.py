import numpy as np
import cv2


# Passing the video as a stream, for a live stream from a camera the parameter will be the devices ID
# Need to test it on Raspberry...
video = cv2.VideoCapture("SourceVideo/Driving3.mp4")


# Basic cv2 operations to open a video and show it in grayscale
def ShowGrayscale(clip):

    while clip.isOpened():
        ret, frame = clip.read()

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', grayscale)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


# Helper function, delete??
def ApplyGrayscale(image):

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def CropFrame(frame):

    # Define the region of interest, those values are obtained with a trial and error approach
    top_left = [frame.shape[1] / 2 - frame.shape[1] / 20, frame.shape[0] / 2 + frame.shape[0] / 6]
    top_right = [frame.shape[1] / 2 + frame.shape[1] / 20, frame.shape[0] / 2 + frame.shape[0] / 6]
    lower_left = [frame.shape[1] / 4, frame.shape[0]]
    lower_right = [frame.shape[1] - frame.shape[1] / 4, frame.shape[0]]

    vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]

    # Define a matrix of zeros that matches the frame height/width
    mask = np.zeros_like(frame)

    # Retrieve the number of color channels of the frame
    if len(frame.shape) > 2:
        channel_count = frame.shape[2]
        # Create a match color with the same color channel counts
        match_mask_color = (255,) * channel_count
    else:
        match_mask_color = 255

    # Fill the blank matrix with pixels only in the area that match
    cv2.fillPoly(mask, vertices, match_mask_color)

    # Returning the frame only where mask pixels match to cut out the unnecessary region
    frame = cv2.bitwise_and(frame, mask)

    return frame


def ApplyHoughLines(frame):
    lines = cv2.HoughLinesP(
        frame,
        rho=6,
        theta=np.pi / 60,
        threshold=200,
        lines=np.array([]),
        minLineLength=100,
        maxLineGap=2500
    )

    return lines


def LinesDrawer(frame, lines, color=[0, 0, 255], thickness=3):

    if lines is None:
        return

    # Make a copy of the original image
    img = np.copy(frame)

    # Create a blank image with the same size as the original
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8,)

    # Draw the lines on the new blank image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    # Merge the two image, the original and the one with lines
    img = cv2.addWeighted(img, 0.8, line_img, 1, 0)

    return img


def FrameProcessing(clip):

    while clip.isOpened():
        ret, frame = clip.read()

        if frame is None:
            break

        # Transform current frame into grayscale
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny Edge Detection to current frame the two value are detection thresholds
        canny = cv2.Canny(grayscale, 100, 200)

        # Crop the frame, it returns the frame containing only the region of interest
        canny = CropFrame(canny)

        lines = ApplyHoughLines(canny)

        # To prevent crashing the script if lines is None type
        if lines is None:
            continue

        for x1, y1, x2, y2 in lines[0]:
            cv2.line(lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

        frame = LinesDrawer(frame, lines)

        canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        merged_frame = np.concatenate((frame, canny), axis=0)

        # Need resize or it will not fit the screen
        merged_frame = cv2.resize(merged_frame, (1280, 720))

        cv2.imshow('merged_frame', merged_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


FrameProcessing(video)

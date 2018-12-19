import numpy as np
import cv2
import time
import matplotlib

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


def LinesDrawer(frame, lines, color=[255, 100, 0], thickness=8):
    if lines is None:
        return

    # Make a copy of the original image
    img = np.copy(frame)

    # Create a blank image with the same size as the original
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8, )

    # Draw the lines on the new blank image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    # Merge the two image, the original and the one with lines
    img = cv2.addWeighted(img, 0.8, line_img, 1, 0)

    return img


def FrameProcessing(FullVisual=False, Resize=[True, "720p"]):
    clip = cv2.VideoCapture("SourceVideo/Driving2.mp4")

    ConvertTime = 0
    HoughTime = 0
    DrawTime = 0
    VisualTime = 0

    ConvertValues = []
    HoughValues = []

    Bench = {'ConvertTime': 0,
             'HoughTime': 0,
             'ConvertValues': 0,
             'HoughValues': 0,
             'DrawTime': 0,
             'VisualTime': 0}

    while clip.isOpened():
        ret, frame = clip.read()

        if frame is None:
            break

        if Resize[0]:
            if Resize[1] == '720p':
                frame = cv2.resize(frame, (1280, 720))
            if Resize[1] == '480p':
                frame = cv2.resize(frame, (720, 480))

        startConvert = time.perf_counter()

        # Transform current frame into grayscale
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny Edge Detection to current frame the two value are detection thresholds
        canny = cv2.Canny(grayscale, 100, 200)

        # Crop the frame, it returns the frame containing only the region of interest
        canny = CropFrame(canny)

        endConvert = abs(startConvert - time.perf_counter())
        ConvertTime = ConvertTime + endConvert
        ConvertValues.append(endConvert)

        startHough = time.perf_counter()

        lines = cv2.HoughLinesP(
            canny,
            rho=6,
            theta=np.pi / 60,
            threshold=200,
            lines=np.array([]),
            minLineLength=100,
            maxLineGap=2500
        )

        endHough = abs(startHough - time.perf_counter())
        HoughTime = HoughTime + endHough
        HoughValues.append(endHough)

        # To prevent crashing the script if lines is None type
        if lines is None:
            continue

        # Draw lines on the frame
        startDraw = time.perf_counter()
        frame = LinesDrawer(frame, lines)
        endDraw = abs(startDraw - time.perf_counter())
        DrawTime = DrawTime + endDraw

        if FullVisual:

            startFullVisual = time.perf_counter()

            canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
            merged_frame = np.concatenate((frame, canny), axis=0)

            # Need resize or it will not fit the screen
            w = merged_frame.shape[1]
            if w > 720:
                merged_frame = cv2.resize(merged_frame, (1280, 720))

            cv2.imshow('merged_frame', merged_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            endFullVisual = abs(startFullVisual - time.perf_counter())
            VisualTime = VisualTime + endFullVisual

        else:

            startFullVisual = time.perf_counter()
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            endFullVisual = abs(startFullVisual - time.perf_counter())
            VisualTime = VisualTime + endFullVisual

    Bench["ConvertTime"] = ConvertTime
    Bench["HoughTime"] = HoughTime
    Bench["ConvertTime"] = DrawTime
    Bench["VisualTime"] = VisualTime
    Bench["ConvertValues"] = ConvertValues
    Bench["HoughValues"] = HoughValues
    return Bench


startTotalTime = time.perf_counter()
Benchmark = FrameProcessing(FullVisual=False, Resize=[False, '720p'])
TotalTime = abs(startTotalTime - time.perf_counter())
print("\nTotal elapsed time:", TotalTime, "Seconds\n")
print(Benchmark['HoughValues'])
print(np.std(Benchmark['HoughValues']))
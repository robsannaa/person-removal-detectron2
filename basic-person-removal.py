import cv2
import time

# initialize the HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam
cap = cv2.VideoCapture(0)

# writing the output to a video
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

# 3 seconds in order to launch the program and clear the webcam view
time.sleep(3)
n_frame = 0

# first frame is too dark, taking 19 th frame as anchor point
while True:
    _, frame = cap.read()
    n_frame += 1

    if n_frame == 20:
        frame = cv2.resize(frame, (640, 480))
        first_frame = frame.copy()
        break


# reinstantiate frame to iterate over the video
_, frame = cap.read()
frame = cv2.resize(frame, (640, 480))

while True:
    # Capture frame-by-frame

    _, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))

    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(gray,
                                          winStride=(4, 4),
                                          padding=(16, 16),
                                          scale=1.05)

    # iterate over each detected person and
    # substitute the area with the first frame
    original = frame.copy()
    rectangle = frame.copy()
    
    for (x, y, w, h) in boxes:

        crop_img = first_frame[y:y + h, x:x + w]
        frame[y:y + h, x:x + w] = crop_img
        
        # could also write
        # frame[y:y + h, x:x + w] = first_frame[y:y + h, x:x + w]

        # for debugging show bounding rectangle
        #cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(rectangle, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # also for debugging
    cv2.imshow('Current Frame', frame)
    cv2.imshow('First Frame', first_frame)
    cv2.imshow('Clean Frame', original)
    cv2.imshow('Rectangle Frame', rectangle)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

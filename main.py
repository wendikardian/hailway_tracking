# First step, u need to declare an library, try load video, and then show it to the windows

import cv2

# Step 14
# Try to import tracker to apply the object tracking from tracker.py file
from tracker import *

# Step 15
# Create a tracker object
# This method is already in tracker.py file
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway.mp4")

# Step 2 
# We need to detect a object from the stable camera using method from cv2 called createBackgroundSubtractorMOG2

# object_detector = cv2.createBackgroundSubtractorMOG2()

# Step 11, try to add parameter on the object detector 
# History parameter need to fit with the level of stable the camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, )

while True:
    ret, frame = cap.read()

    # To find out the dimension of the frame

    # _ means that u did'nt want to store the value channel of the frame
    height, width, _ = frame.shape
    # print(height, width)

    # Step 8, 
    # We need to extract region of interest
    # Because the frame dimension is to big, we need to specific the area of interest
    # This one means is you crop the frame height from 340 - 720 coordinates, and the width from 500 to 800 coordinates
    roi = frame[340: 720, 500: 800]

    # Step 3
    # Create a mask to detect an object
    # mask = object_detector.apply(frame)

    # Step 10
    # Try only draw it on ROI area
    mask = object_detector.apply(roi)

    # Step 13
    # Because the mask can detect the shadow to, let's make a threshold to remove the gray shadow
    # The more value to 0  is more dark (black)
    # The more value to 255 is more bright (white)
    # In this code we're going to keep the white area from 254 to 255 (which is white) using cv2 method called THRESH BINARY
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    # Step 5
    # After find the mask, we need the spesific coordinates of the white object(moving object) using contour
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Step 16, 
    # Create an empty array, if we find out the object to detect, we append it to the array
    detections = []

    # Step 6
    # After find out the contour try to print it and draw the contour
    for cnt in contours:
        # print(cnt)

        # Step 7. 
        # Because to many contours in the frame, we need to remove the unnecessary contour in the frame
        # We only collect the big contour that really sure is a moving object
        area = cv2.contourArea(cnt)

        # This means if the area is greeter than 100 pixels we draw the contour,
        # Other than that the contour will be removed 
        if area > 100:
            # Draw the contour
            # -1 means that we want to draw all of that
            # cv2.drawContours(frame, [cnt], -1,(0,255,0), 2 )

            # Draw it on ROI area
            # cv2.drawContours(roi, [cnt], -1,(0,255,0), 2 )

            # Step 12.
            # Try to extract the data x,y ,h, w from the bounding box
            x, y, w,h = cv2.boundingRect(cnt)

            # After find the coordinates, let's draw the rectangle
            # cv2.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 2)
            

            # Step 17
            # Append it to the list, if we find a new object
            detections.append([x,y,w,h])

    # Step 18
    # After find the list of the object, try to print it
    # If only 1 object it will the list.length == 1, it 2 length 2, so other numbers too
    # print(detections)


    # Step 19.
    # from step 1 -18 is object detection
    # After detect an object, next step we're going to implement tract the object
    # Every object/bounding box has a unique id that 
    boxes_ids =tracker.update(detections)
    # print(boxes_ids)

    # After find the boxes_id we're going to draw the bounding box and put the text on it
    for boxes_id in boxes_ids:
        # Extract the data
        x,y,w,h,id = boxes_id

        # Put the text and the bounding box
        cv2.putText(roi, str(id), (x,y -15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2 )
        # Just copy and paste from the if code up there
        cv2.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 2)
        


    cv2.imshow("Frame", frame)

    # Step 4
    # After create a mask, let's show the mask
    # Mask return an frame using grayscale images
    # The goal of a mask is making an object that not moving turning into black
    # While the moving object turning into white
    cv2.imshow("mask", mask)

    # Step 9
    # After you find out the roi area try to display it where is the roi area
    cv2.imshow("ROI AREA", roi)

    key = cv2.waitKey(30)

    if key==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

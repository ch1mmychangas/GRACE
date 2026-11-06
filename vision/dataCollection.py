
#Collection of data from gestures
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os
import time  

print("Initialize the video capture and hand detector")
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

#offset for cropping the image
offset = 20
imgSize = 300

# Save the images in a folder
folder = "gestureData/threeFingers"
os.makedirs(folder, exist_ok=True)
prev_center = None
move_threshold = 40  # Threshold for movement detection

while True:
    success, img = cap.read()
     # Find the hands in the image
    hands, img = detector.findHands(img) 
    if hands:
        hand = hands[0]
        # Get the bounding box information=
        x, y, w, h = hand['bbox']
        # Create an empty white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the image to the bounding box
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        #Prevent error if the crop is empty
        if imgCrop.size == 0 or imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
            continue
        
        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            # The aspect ratio is greater than 1, resize based on height
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            #imgResizeShape = imgResize.shape
            #Create gap to center the image
            wGap = math.ceil((imgSize - wCal) / 2)
            #clip the image to the width of imgWhite
            wEnd = min(wGap + imgResize.shape[1], imgSize)
            #Fills the white image with the resized image
            imgWhite[:, wGap:wEnd] = imgResize[:, :wEnd - wGap]

            
        else:
            #The aspect ratio is less than or equal to 1, resize based on width
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            #imgResizeShape = imgResize.shape
            #Create gap to center the image
            hGap = math.ceil((imgSize - hCal) / 2)
            #clip the image to the height of imgWhite
            hEnd= min(hGap + imgResize.shape[0], imgSize)
            #Fills the white image with the resized image
            imgWhite[hGap:hEnd, :] = imgResize[:hEnd - hGap, :]
            

        # Display the image
        cv2.imshow("imageCrop", imgCrop)
        cv2.imshow("imageWhite", imgWhite)

        center = (x + w // 2, y + h // 2)

        # Automatically save the image if the hand moves significantly
        if prev_center is None:
            prev_center = center
            timestamp = int(time.time() * 1000)  # milliseconds
            cv2.imwrite(f"{folder}/img_{timestamp}.jpg", imgWhite)
            print(f"Image saved: {timestamp}")
        else:
            dist = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
            if dist > move_threshold:
                prev_center = center
                timestamp = int(time.time() * 1000)
                cv2.imwrite(f"{folder}/img_{timestamp}.jpg", imgWhite)
                print(f"Image saved: {timestamp}")

    cv2.imshow("Image", img)
    
    key = cv2.waitKey(1)
    # Save the image on 's' key press
    if key & 0xFF == ord('s'):
        # Save the image to the folder with a unique timestamp
        timestamp = int(time.time() * 1000)
        cv2.imwrite(f"{folder}/img_{timestamp}.jpg", imgWhite)
        print(f"Image saved: {timestamp}")

    # Exit on 'q' key press
    if key & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

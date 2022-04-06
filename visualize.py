from detection import detectHandsLandmarks
from detection import hands_videos
from counter import countFingers

import cv2
import time
import pygame
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

def annotate(image, results, fingers_statuses, count, display=True):
    '''
    This function will draw an appealing visualization of each fingers up of the both hands in the image.
    Args:
        image:            The image of the hands on which the counted fingers are required to be visualized.
        results:          The output of the hands landmarks detection performed on the image of the hands.
        fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands. 
        count:            A dictionary containing the count of the fingers that are up, of both hands.
        display:          A boolean value that is if set to true the function displays the resultant image and 
                          returns nothing.
    Returns:
        output_image: A copy of the input image with the visualization of counted fingers.
    '''
    
    # Get the height and width of the input image.
    height, width, _ = image.shape
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Select the images of the hands prints that are required to be overlayed.
    ########################################################################################################################
    
    # Initialize a dictionaty to store the images paths of the both hands.
    # Initially it contains red hands images paths. The red image represents that the hand is not present in the image. 
    HANDS_IMGS_PATHS = {'LEFT': ['Assets/left_hand_not_detected.png'], 'RIGHT': ['Assets/right_hand_not_detected.png']}
    
    # Check if there is hand(s) in the image.
    if results.multi_hand_landmarks:
        
        # Iterate over the detected hands in the image.
        for hand_index, hand_info in enumerate(results.multi_handedness):
            
            # Retrieve the label of the hand.
            hand_label = hand_info.classification[0].label
            
            # Update the image path of the hand to a green color hand image.
            # This green image represents that the hand is present in the image. 
            HANDS_IMGS_PATHS[hand_label.upper()] = ['Assets/'+hand_label.lower()+'_hand_detected.png']
            
            # Check if all the fingers of the hand are up/open.
            if count[hand_label.upper()] == 5:
                
                # Update the image path of the hand to a hand image with green color palm and orange color fingers image.
                # The orange color of a finger represents that the finger is up.
                HANDS_IMGS_PATHS[hand_label.upper()] = ['Assets/'+hand_label.lower()+'_all_fingers.png']
            
            # Otherwise if all the fingers of the hand are not up/open.
            else:
                
                # Iterate over the fingers statuses of the hands.
                for finger, status in fingers_statuses.items():
                    
                    # Check if the finger is up and belongs to the hand that we are iterating upon.
                    if status == True and finger.split("_")[0] == hand_label.upper():
                        
                        # Append another image of the hand in the list inside the dictionary.
                        # This image only contains the finger we are iterating upon of the hand in orange color.
                        # As the orange color represents that the finger is up.
                        HANDS_IMGS_PATHS[hand_label.upper()].append('Assets/'+finger.lower()+'.png')
    
    ########################################################################################################################
    
    # Overlay the selected hands prints on the input image.
    ########################################################################################################################
    
    # Iterate over the left and right hand.
    for hand_index, hand_imgs_paths in enumerate(HANDS_IMGS_PATHS.values()):
        
        # Iterate over the images paths of the hand.
        for img_path in hand_imgs_paths:
            
            # Read the image including its alpha channel. The alpha channel (0-255) determine the level of visibility. 
            # In alpha channel, 0 represents the transparent area and 255 represents the visible area.
            hand_imageBGRA = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            # Retrieve all the alpha channel values of the hand image. 
            alpha_channel = hand_imageBGRA[:,:,-1]
            
            # Retrieve all the blue, green, and red channels values of the hand image.
            # As we also need the three-channel version of the hand image. 
            hand_imageBGR  = hand_imageBGRA[:,:,:-1]
            
            # Retrieve the height and width of the hand image.
            hand_height, hand_width, _ = hand_imageBGR.shape

            # Retrieve the region of interest of the output image where the handprint image will be placed.
            ROI = output_image[30 : 30 + hand_height,
                               (hand_index * width//2) + width//12 : ((hand_index * width//2) + width//12 + hand_width)]
            
            # Overlay the handprint image by updating the pixel values of the ROI of the output image at the 
            # indexes where the alpha channel has the value 255.
            ROI[alpha_channel==255] = hand_imageBGR[alpha_channel==255]

            # Update the ROI of the output image with resultant image pixel values after overlaying the handprint.
            output_image[30 : 30 + hand_height,
                         (hand_index * width//2) + width//12 : ((hand_index * width//2) + width//12 + hand_width)] = ROI
    
    ########################################################################################################################
    
    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:

        # Return the output image
        return output_image

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)

# Create named window for resizing purposes.
cv2.namedWindow('Counted Fingers Visualization', cv2.WINDOW_NORMAL)

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():
    
    # Read a frame.
    ok, frame = camera_video.read()
    
    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Perform Hands landmarks detection on the frame.
    frame, results = detectHandsLandmarks(frame, hands_videos, display=False)
    
    # Check if the hands landmarks in the frame are detected.
    if results.multi_hand_landmarks:
            
        # Count the number of fingers up of each hand in the frame.
        frame, fingers_statuses, count = countFingers(frame, results, display=False)
    
    # Visualize the counted fingers.
    frame = annotate(frame, results, fingers_statuses, count, display=False)
                
    # Display the frame.
    cv2.imshow('Counted Fingers Visualization', frame)
    
    # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF
    
    # Check if 'ESC' is pressed and break the loop.
    if(k == 27):
        break

# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()
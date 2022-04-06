from detection import detectHandsLandmarks
from detection import hands
from counter import countFingers

import cv2
import time
import pygame
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

def recognizeGestures(image, fingers_statuses, count, draw=True, display=True):
    '''
    This function will determine the gesture of the left and right hand in the image.
    Args:
        image:            The image of the hands on which the hand gesture recognition is required to be performed.
        fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands. 
        count:            A dictionary containing the count of the fingers that are up, of both hands.
        draw:             A boolean value that is if set to true the function writes the gestures of the hands on the
                          output image, after recognition.
        display:          A boolean value that is if set to true the function displays the resultant image and 
                          returns nothing.
    Returns:
        output_image:   A copy of the input image with the left and right hand recognized gestures written if it was 
                        specified.
        hands_gestures: A dictionary containing the recognized gestures of the right and left hand.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Store the labels of both hands in a list.
    hands_labels = ['RIGHT', 'LEFT']
    
    # Initialize a dictionary to store the gestures of both hands in the image.
    hands_gestures = {'RIGHT': "UNKNOWN", 'LEFT': "UNKNOWN"}
    
    # Iterate over the left and right hand.
    for hand_index, hand_label in enumerate(hands_labels):
        
        # Initialize a variable to store the color we will use to write the hands gestures on the image.
        # Initially it is red which represents that the gesture is not recognized.
        color = (0, 0, 255)
        
        # Check if the person is making the 'V' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        if count[hand_label] == 2  and fingers_statuses[hand_label+'_MIDDLE'] and fingers_statuses[hand_label+'_INDEX']:
            
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures[hand_label] = "V SIGN"
            
            # Update the color value to green.
            color=(0,255,0)
            
        ####################################################################################################################            
        
        # Check if the person is making the 'SPIDERMAN' gesture with the hand.
        ##########################################################################################################################################################
        
        # Check if the number of fingers up is 3 and the fingers that are up, are the thumb, index and the pinky finger.
        elif count[hand_label] == 3 and fingers_statuses[hand_label+'_THUMB'] and fingers_statuses[hand_label+'_INDEX'] and fingers_statuses[hand_label+'_PINKY']:
                
            # Update the gesture value of the hand that we are iterating upon to SPIDERMAN SIGN.
            hands_gestures[hand_label] = "SPIDERMAN SIGN"

            # Update the color value to green.
            color=(0,255,0)
                
        ##########################################################################################################################################################
        
        # Check if the person is making the 'HIGH-FIVE' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 5, which means that all the fingers are up.
        elif count[hand_label] == 5:
            
            # Update the gesture value of the hand that we are iterating upon to HIGH-FIVE SIGN.
            hands_gestures[hand_label] = "HIGH-FIVE SIGN"
            
            # Update the color value to green.
            color=(0,255,0)
       
        ####################################################################################################################  
        
        # Check if the hands gestures are specified to be written.
        if draw:
        
            # Write the hand gesture on the output image. 
            cv2.putText(output_image, hand_label +': '+ hands_gestures[hand_label] , (10, (hand_index+1) * 60),
                        cv2.FONT_HERSHEY_PLAIN, 4, color, 5)
            
    
    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:

        # Return the output image and the gestures of the both hands.
        return output_image, hands_gestures

# Read a sample image and perform the hand gesture recognition on it after flipping it horizontally.
image = cv2.imread('Assets/sample1.jpg')
flipped_image = cv2.flip(image, 1)
_, results = detectHandsLandmarks(flipped_image, hands, display=False)
if results.multi_hand_landmarks:
    output_image, fingers_statuses, count = countFingers(image, results, draw=False, display = False)
    recognizeGestures(image, fingers_statuses, count)

# Read another sample image and perform the hand gesture recognition on it after flipping it horizontally.
image = cv2.imread('Assets/sample2.jpg')
flipped_image = cv2.flip(image, 1)
_, results = detectHandsLandmarks(flipped_image, hands, display=False)
if results.multi_hand_landmarks:
    output_image, fingers_statuses, count =countFingers(image, results, draw=False, display = False)
    recognizeGestures(image, fingers_statuses, count)

# Read another sample image and perform the hand gesture recognition on it after flipping it horizontally.
image = cv2.imread('Assets/sample3.jpg')
flipped_image = cv2.flip(image, 1)
_, results = detectHandsLandmarks(flipped_image, hands, display=False)
if results.multi_hand_landmarks:
    output_image, fingers_statuses, count =countFingers(image, results, draw=False, display = False)
    recognizeGestures(image, fingers_statuses, count)
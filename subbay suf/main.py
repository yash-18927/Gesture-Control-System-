import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key
from time import sleep
import numpy as np

# Keyboard controller (simulating key presses)
keyboard = Controller()

# MediaPipe hand tracking using new API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task', delegate=BaseOptions.Delegate.CPU),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

hand_landmarker = HandLandmarker.create_from_options(options)

# Hand landmark indices
WRIST = 0
THUMB_TIP = 4
THUMB_MCP = 2
INDEX_FINGER_TIP = 8
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
MIDDLE_FINGER_TIP = 12
MIDDLE_FINGER_MCP = 9
RING_FINGER_TIP = 16
RING_FINGER_MCP = 13
PINKY_TIP = 20
PINKY_MCP = 17

# Start the webcam
cap = cv2.VideoCapture(0)

# Variable initialization
is_left_pressed = False
is_right_pressed = False
is_up_pressed = False
is_down_pressed = False

# Previous wrist position for movement tracking
prev_wrist_y = None

# Gesture detection functions
def is_fist(landmarks):
    """Check if hand is in a fist (all fingers closed)"""
    if len(landmarks) < 21:
        return False
    
    index_tip = landmarks[INDEX_FINGER_TIP]
    middle_tip = landmarks[MIDDLE_FINGER_TIP]
    ring_tip = landmarks[RING_FINGER_TIP]
    little_tip = landmarks[PINKY_TIP]
    
    index_mcp = landmarks[INDEX_FINGER_MCP]
    middle_mcp = landmarks[MIDDLE_FINGER_MCP]
    ring_mcp = landmarks[RING_FINGER_MCP]
    little_mcp = landmarks[PINKY_MCP]
    
    if (index_tip.y > index_mcp.y and 
        middle_tip.y > middle_mcp.y and 
        ring_tip.y > ring_mcp.y and 
        little_tip.y > little_mcp.y):
        return True
    return False

def is_pointing_up(landmarks):
    """Check if index finger is pointing up (for jump)"""
    if len(landmarks) < 21:
        return False
    
    index_tip = landmarks[INDEX_FINGER_TIP]
    index_pip = landmarks[INDEX_FINGER_PIP]
    index_mcp = landmarks[INDEX_FINGER_MCP]
    middle_tip = landmarks[MIDDLE_FINGER_TIP]
    middle_mcp = landmarks[MIDDLE_FINGER_MCP]
    
    # Index finger extended upward
    if (index_tip.y < index_pip.y and 
        index_pip.y < index_mcp.y and
        index_tip.y < middle_tip.y):  # Index finger higher than middle
        return True
    return False

def is_pointing_down(landmarks):
    """Check if hand is pointing down (for roll/slide)"""
    if len(landmarks) < 21:
        return False
    
    wrist = landmarks[WRIST]
    index_tip = landmarks[INDEX_FINGER_TIP]
    middle_tip = landmarks[MIDDLE_FINGER_TIP]
    
    # Fingers pointing downward
    if (index_tip.y > wrist.y and middle_tip.y > wrist.y):
        return True
    return False

def get_hand_side(landmarks, frame_width):
    """Determine if hand is on left or right side of screen"""
    if len(landmarks) < 21:
        return None
    
    wrist_x = landmarks[WRIST].x
    # Left side: x < 0.4, Right side: x > 0.6, Center: 0.4-0.6
    if wrist_x < 0.4:
        return "left"
    elif wrist_x > 0.6:
        return "right"
    else:
        return "center"

# Drawing is done manually using OpenCV

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the video for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]

    # Convert the frame to RGB and create MediaPipe Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Process the frame for hand detection
    detection_result = hand_landmarker.detect(mp_image)

    # Reset all key states at start of frame
    current_left = False
    current_right = False
    current_up = False
    current_down = False
    status_text = ""

    # If hands are detected
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Convert landmarks to list format for easier access
            landmarks_list = []
            for landmark in hand_landmarks:
                landmarks_list.append(landmark)
            
            # Draw hand landmarks
            for idx, landmark in enumerate(hand_landmarks):
                # Get coordinates
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                
                # Different colors for each finger's tip
                if idx == WRIST:
                    color = (255, 255, 0)  # Cyan for Wrist
                elif idx == THUMB_TIP:
                    color = (0, 0, 255)  # Red for Thumb
                elif idx == INDEX_FINGER_TIP:
                    color = (0, 255, 0)  # Green for Index Finger
                elif idx == MIDDLE_FINGER_TIP:
                    color = (255, 0, 0)  # Blue for Middle Finger
                elif idx == RING_FINGER_TIP:
                    color = (0, 255, 255)  # Yellow for Ring Finger
                elif idx == PINKY_TIP:
                    color = (255, 0, 255)  # Purple for Pinky Finger
                else:
                    color = (255, 255, 255)  # White for others

                # Draw a circle on each landmark
                cv2.circle(frame, (x, y), 5, color, -1)
            
            # Draw connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (5, 9), (9, 13), (13, 17)  # Palm
            ]
            for start_idx, end_idx in connections:
                if start_idx < len(landmarks_list) and end_idx < len(landmarks_list):
                    start = landmarks_list[start_idx]
                    end = landmarks_list[end_idx]
                    start_point = (int(start.x * frame_width), int(start.y * frame_height))
                    end_point = (int(end.x * frame_width), int(end.y * frame_height))
                    cv2.line(frame, start_point, end_point, (255, 255, 255), 2)

            # Detect gestures
            hand_side = get_hand_side(landmarks_list, frame_width)
            
            # Check for UP gesture (index finger pointing up) - JUMP
            if is_pointing_up(landmarks_list):
                current_up = True
                status_text = "JUMP (UP)"
                if not is_up_pressed:
                    print("Jump gesture detected!")
                    keyboard.press(Key.up)
                    is_up_pressed = True
            
            # Check for DOWN gesture (pointing down or fist) - ROLL/SLIDE
            elif is_pointing_down(landmarks_list) or is_fist(landmarks_list):
                current_down = True
                status_text = "ROLL/SLIDE (DOWN)"
                if not is_down_pressed:
                    print("Roll/Slide gesture detected!")
                    keyboard.press(Key.down)
                    is_down_pressed = True
            
            # Check for LEFT/RIGHT based on hand position
            elif hand_side == "left":
                current_left = True
                status_text = "MOVE LEFT"
                if not is_left_pressed:
                    print("Left gesture detected!")
                    keyboard.press(Key.left)
                    is_left_pressed = True
            
            elif hand_side == "right":
                current_right = True
                status_text = "MOVE RIGHT"
                if not is_right_pressed:
                    print("Right gesture detected!")
                    keyboard.press(Key.right)
                    is_right_pressed = True

    # Release keys that are no longer active
    if not current_left and is_left_pressed:
        keyboard.release(Key.left)
        is_left_pressed = False
    
    if not current_right and is_right_pressed:
        keyboard.release(Key.right)
        is_right_pressed = False
    
    if not current_up and is_up_pressed:
        keyboard.release(Key.up)
        is_up_pressed = False
    
    if not current_down and is_down_pressed:
        keyboard.release(Key.down)
        is_down_pressed = False

    # Display instructions and status
    cv2.putText(frame, "Subway Surfers Gesture Control", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Left: Move Left | Right: Move Right", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Point Up: Jump | Point Down/Fist: Roll", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'q' to quit", (10, frame_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display the status text on the frame if an action is detected
    if status_text:
        cv2.rectangle(frame, (10, frame_height - 100), (300, frame_height - 50), (0, 255, 0), -1)
        cv2.putText(frame, status_text, (20, frame_height - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame with hand gesture control
    cv2.imshow("Subway Surfers - Hand Gesture Control", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all keys before exiting
if is_left_pressed:
    keyboard.release(Key.left)
if is_right_pressed:
    keyboard.release(Key.right)
if is_up_pressed:
    keyboard.release(Key.up)
if is_down_pressed:
    keyboard.release(Key.down)

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()

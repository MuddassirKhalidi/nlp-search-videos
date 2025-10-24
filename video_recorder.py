#!/usr/bin/env python3
"""
Minimal video recorder script
Records video from webcam and saves as MP4 file
"""

import cv2
import datetime
import os

def record_video():
    # Create recordings directory if it doesn't exist
    os.makedirs("recordings", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recordings/video_{timestamp}.mp4"
    
    # Initialize video capture (0 for default webcam)
    cap = cv2.VideoCapture(0)
    
    # Set video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    print(f"Recording started. Saving to: {filename}")
    print("Press 'q' to stop recording...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Write frame to video file
        out.write(frame)
        
        # Display frame (optional)
        cv2.imshow('Recording...', frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Recording saved to: {filename}")

if __name__ == "__main__":
    record_video()



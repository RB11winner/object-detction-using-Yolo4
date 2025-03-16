import cv2  

# Load the video file
video_path = "6574285-hd_1280_720_25fps.mp4"  # Change this to your video path
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through the video frames
while True:
    ret, frame = cap.read()  # Read a frame
    
    if not ret:
        break  # Break if video ends
    
    # Convert frame to grayscale (optional, for processing)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Show original and grayscale frames
    cv2.imshow("Original Video", frame)
    cv2.imshow("Grayscale Video", gray_frame)

    # Press 'q' to exit video display
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

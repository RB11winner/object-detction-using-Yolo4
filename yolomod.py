import cv2
import numpy as np
import csv

# Load YOLO model
yolo_config = "yolov4.cfg"
yolo_weights = "yolov4.weights"
yolo_names = "coco.names"

# Load class labels
with open(yolo_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize YOLO
net = cv2.dnn.readNet(yolo_weights, yolo_config)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Open video
video_path = "6574285-hd_1280_720_25fps.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create & open CSV file for writing traffic data
csv_filename = "traffic_data.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Humans", "Vehicles", "Traffic Status"])  # Column headers

    frame_count = 0  # Track frame number

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        height, width, channels = frame.shape

        # Convert frame to blob for YOLO input
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        human_count = 0
        vehicle_count = 0

        # Process detections
        for output in detections:
            for detection in output:
                scores = detection[5:]  # Confidence scores for each class
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Detect only humans and vehicles
                if confidence > 0.5 and classes[class_id] in ["person", "car", "bus", "truck", "motorbike"]:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")

                    # Get top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Count objects
                    if classes[class_id] == "person":
                        human_count += 1
                        color = (0, 255, 0)  # Green for people
                    else:
                        vehicle_count += 1
                        color = (255, 0, 0)  # Blue for vehicles

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = f"{classes[class_id]}: {int(confidence * 100)}%"
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Determine traffic status
        traffic_status = "No Traffic"
        if human_count + vehicle_count > 5:  # Set threshold (adjust as needed)
            traffic_status = "Traffic Detected"

        # Write data to CSV file
        writer.writerow([frame_count, human_count, vehicle_count, traffic_status])

        # Display the count on the video
        cv2.putText(frame, f"Humans: {human_count}  Vehicles: {vehicle_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Status: {traffic_status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if traffic_status == "Traffic Detected" else (0, 255, 0), 2)

        # Show output frame
        cv2.imshow("Traffic Analysis", frame)

        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Traffic data saved to {csv_filename} ✅")

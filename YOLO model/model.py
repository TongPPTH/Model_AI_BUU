import glob
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO('./IVUF-100-1/runs/detect/train2/weights/best.pt')  # Load your YOLO model

class_name = ['Lumen']  # Define class name(s)
img_id =1
# Loop through the image files (ensure correct path)
for image_path in glob.glob('./IVUF-100-1/test/images/*.jpg'):
    results = model(image_path)  # Get the model results
    img = cv2.imread(image_path)  # Load the image with OpenCV

    for box in results[0].boxes:
        # Get bounding box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw the bounding box on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the class name and confidence score
        conf = box.conf[0]
        label = int(box.cls[0])
        cv2.putText(img, f"{class_name[label]} conf:{conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert BGR to RGB for Matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_path = f"./output/{img_id}.jpg"
    img_id += 1
    cv2.imwrite(output_path, img_rgb)
    
    
    # Optionally, save the resulting image instead of displaying:
    # output_path = image_path.replace("images", "output")  # Save to output directory
    # cv2.imwrite(output_path, img)

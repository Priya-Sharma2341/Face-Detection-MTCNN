#!pip install mtcnn
#!pip install opencv-python

from mtcnn import MTCNN
import cv2

# Load the MTCNN detector
detector = MTCNN()

# Load an image
image = cv2.imread('1D.jpg')

# Convert the image to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces in the image
faces = detector.detect_faces(image_rgb)

# Iterate over the detected faces
for face in faces:
    # Extract the bounding box coordinates
    x, y, width, height = face['box']

    # Draw the bounding box on the image
    cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)

    # Draw the landmarks (optional)
    landmarks = face['keypoints']
    for point in landmarks.values():
        cv2.circle(image, point, 2, (0, 0, 255), 2)

# Display the image with bounding boxes and landmarks
cv2.imshow('Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

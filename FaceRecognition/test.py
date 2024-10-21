import cv2
import os

face_directory = '/home/ian/Univali/Machine Learning/ComputerVision_Projects/FaceRecognition/faces'
imgs = []
for filename in os.listdir(face_directory):
    img_path = os.path.join(face_directory, filename)
    # Ensure you're only processing image files (e.g., .jpg, .png)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img = cv2.imread(img_path)
        imgs.append(img)

cv2.imshow('face-recognition', imgs[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
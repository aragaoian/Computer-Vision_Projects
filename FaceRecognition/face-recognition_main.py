import cv2
import face_recognition
import os
import pickle
import imutils 
import time

path_dataset = os.path.dirname(cv2.__file__) + '/data/haarcascade_frontalface_alt2.xml'
# print(os.path.exists(path_dataset))
cascade_classifier = cv2.CascadeClassifier(path_dataset)
names = ['Ian', 'Rosane']

known_encodings = []
face_directory = 'faces'
for filename in os.listdir(face_directory):
    img_path = os.path.join(face_directory, filename)
    # Ensure you're only processing image files (e.g., .jpg, .png)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(img_rgb, model='hog') # locate face on the img
        encodings = face_recognition.face_encodings(img_rgb, boxes) # facial embedding for the faces
        for encoding in encodings:
            known_encodings.append(encoding)

data = {"encodings": known_encodings}


cap = cv2.VideoCapture('/dev/video0')
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

if not cap.isOpened():
    print('Error')
else:
    while True:
        check, frame = cap.read()
        if not check: break
        start = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                    minSize=(60,60)) # ,flags=cv2.CASCADE_SCALE_IMAGE) # returns a list of rectangles corresponding to the detected faces
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame) # list of extracted face features

        matched_names = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(data['encodings'], encoding)
            name = 'Unkown'
            if True in matches:
                ids = [index for (index, bool) in enumerate(matches)] # A list of True/False values indicating which known_face_encodings match the face encoding to check
                matchedFaces_counts = {}
                for id in ids:
                    name = names[id]
                    matchedFaces_counts[name] = matchedFaces_counts.get(name, 0) + 1
                name = max(matchedFaces_counts, key=matchedFaces_counts.get)
            matched_names.append(name) # append the most matched name in the entire frame

            for ((x, y, w, h), name) in zip (faces, matched_names): # x-top_left | y-top_left
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 255), 2)

        end = time.time()
        fps = str(int(1/ (end-start)))
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2) 

        cv2.imshow('face-recognition', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()


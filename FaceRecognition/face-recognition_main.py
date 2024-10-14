import cv2
import face_recognition
import os
import pickle
import imutils 

path_dataset = os.path.dirname(cv2.__file__) + '/data/haarcascade_frontalface_alt2.xml'
# print(os.path.exists(path_dataset))
cascade_classifier = cv2.CascadeClassifier(path_dataset)
names = ['Ian', 'Rosane']

face_directory = 'faces'
for filename in os.listdir(face_directory):
    file_path = os.path.join(face_directory, filename)
    # Ensure you're only processing image files (e.g., .jpg, .png)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        data = cv2.imread(file_path)

cap = cv2.VideoCapture('/dev/video0')
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # see if the error is in this line due to the .mjpg type

if not cap.isOpened():
    print('Error')
else:
    while True:
        check, frame = cap.read()
        if not check: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                    minSize=(60,60), flags=cv2.CASCADE_SCALE_IMAGE)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame) # might be in here

        for encoding in encodings:
            matches = face_recognition.compare_faces(data, encoding)
            name = 'Unkown'
            if True in matches:
                ids = [i for (i, b) in enumerate(matches)]
                counts = {}
                for id in ids:
                    name = names[id] # might in there (name's based error)
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

            for ((x, y, w, h), name) in zip (faces, names): # error might be in here
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow('face-recognition', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()


import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

faculty_images = []
faculty_names = []
dataset_folder = 'faculty_dataset'
for file in os.listdir(dataset_folder):
    if file.endswith(".jpg") or file.endswith(".png"):
        image_path = os.path.join(dataset_folder, file)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if len(encoding) > 0:
            faculty_images.append(encoding[0])
            faculty_names.append(os.path.splitext(file)[0])
entry_log = {}
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(faculty_images, face_encoding)
        name = "Unknown"
        if len(faculty_images) > 0:
            face_distances = face_recognition.face_distance(faculty_images, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = faculty_names[best_match_index]

                    if name not in entry_log:
                        entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        entry_log[name] = entry_time
                        print(f"{name} entered at {entry_time}")

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Faculty Face Recognition', frame)


    if cv2.waitKey(1) == 13:  
        break

cap.release()
cv2.destroyAllWindows()

with open('entry_log.txt', 'w') as log_file:
    for name, entry_time in entry_log.items():
        log_file.write(f"{name}: {entry_time}\n")

print("Entry log saved to entry_log.txt")

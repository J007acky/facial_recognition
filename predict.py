import face_recognition
import pickle
import cv2
import pyrebase


# Initialize Firebase using Pyrebase
config = {
    "apiKey": "AIzaSyClnRJAnrJgAgkYjuYnlvu-CJ6Cxyklebo",
    "databaseURL": "https://console.firebase.google.com/project/socioverse-2025/database/socioverse-2025-default-rtdb/data/~2F",
    "authDomain": "socioverse-2025.firebaseapp.com",
    "projectId": "socioverse-2025",
    "storageBucket": "socioverse-2025.appspot.com",
    "messagingSenderId": "689574504641",
    "appId": "1:689574504641:web:a22f6a2fa343e4221acc40",
    "serviceAccount":"socioverse-2025-firebase-adminsdk-gcc6m-6bfb53e6d9.json"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()


storage.child().download("Faces/pkl/face_encodings.pkl","face_encodings.pkl")

# Load the stored face encodings and labels from the pickle file
with open("face_encodings.pkl", "rb") as file:
    data = pickle.load(file)
    face_encodings = data["encodings"]
    labels = data["labels"]

# Load a new image you want to recognize
new_image = cv2.imread("download.jpg")


new_face_encoding = face_recognition.face_encodings(new_image)

if len(new_face_encoding) == 0:
    print("No faces found in the new image.")
else:
    # Compare the new face encoding to the stored encodings
    results = face_recognition.compare_faces(face_encodings, new_face_encoding[0])

    for i, result in enumerate(results):
        if result:
            print(f"Recognised : {labels[i]}")
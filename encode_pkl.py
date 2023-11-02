import os
import cv2
import face_recognition
import pickle
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

# Define the folder containing face images in the Firebase Storage bucket
storage_folder = "Faces/"

# Create an array to store encodings and corresponding labels
face_encodings = []
labels = []

# List all files in the Firebase Storage folder
blobs = storage.child(storage_folder).list_files()


for blob in blobs:
    if blob.name.startswith(storage_folder) and (blob.name.endswith(".jpeg") or blob.name.endswith(".jpg") or blob.name.endswith(".png")):
        # Download the image to a local file
        url = storage.child(f"{blob.name}").get_url(None)

        storage.child(url).download(f"{blob.name}","temp.jpeg")
        # Load the image using OpenCV
        img = cv2.imread("temp.jpeg")

        # Convert the BGR image to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        encode = face_recognition.face_encodings(img)[0]
        face_encodings.append(encode)
        name_parts = blob.name.split('/')
        name = name_parts[-1]
        name_parts = name.split('$')
        name = name_parts[0]
        labels.append(name)

        # Delete the temporary downloaded image
        os.remove("temp.jpeg")

# Save the encodings and labels to a pickle file
data = {"encodings": face_encodings, "labels": labels}
with open("face_encodings.pkl", "wb") as file:
    pickle.dump(data, file)

# Upload the pickle file to Firebase Storage
pkl_blob = storage.child(f"{storage_folder}pkl/face_encodings.pkl")
pkl_blob.put("face_encodings.pkl")

print("Face encodings and labels saved to Firebase Storage in 'faces/pkl' folder as 'face_encodings.pkl'.")
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
import requests
import face_recognition
import pickle
import cv2
import pyrebase
import os


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


app = FastAPI()

class ImgSave(BaseModel):
    image_url: HttpUrl
    user_name: str

class ImgInput(BaseModel):
    image_url: HttpUrl

class ImgOutput(BaseModel):
    label: str

def recognize_face(image_url: HttpUrl) -> ImgOutput:

    storage.child().download("Faces/pkl/face_encodings.pkl","face_encodings.pkl")
    # Downloading image
    response = requests.get(image_url)
    with open("examp.jpg", 'wb') as file:
        file.write(response.content)

    # Load the stored face encodings and labels from the pickle file
    with open("face_encodings.pkl", "rb") as file:
        data = pickle.load(file)
        face_encodings = data["encodings"]
        labels = data["labels"]

    # Load a new image you want to recognize
    new_image = cv2.imread("examp.jpg")

    new_face_encoding = face_recognition.face_encodings(new_image)

    if len(new_face_encoding) == 0:
        print("No faces found in the new image.")
    else:
        # Compare the new face encoding to the stored encodings
        results = face_recognition.compare_faces(face_encodings, new_face_encoding[0])

        os.remove("examp.jpg")

        for i, result in enumerate(results):
            if result:
                return ImgOutput(label=labels[i])

    return ImgOutput(label="unable to detect")



def add_face(image_url: HttpUrl,user_name : str) -> ImgOutput:
    # Downloading image
    response = requests.get(image_url)
    with open("examp.jpg", 'wb') as file:
        file.write(response.content)

    # Load the stored face encodings and labels from the pickle file
    with open("face_encodings.pkl", "rb") as file:
        data = pickle.load(file)
        face_encodings = data["encodings"]
        labels = data["labels"]

    # Load a new image you want to recognize
    new_image = cv2.imread("examp.jpg")

    # Convert the BGR image to RGB
    rgb_img = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

    encode = face_recognition.face_encodings(new_image)[0]
    face_encodings.append(encode)
    labels.append(user_name)

    # Delete the temporary downloaded image
    os.remove("examp.jpg")

    # Save the encodings and labels to a pickle file
    data = {"encodings": face_encodings, "labels": labels}
    with open("face_encodings.pkl", "wb") as file:
        pickle.dump(data, file)

    # Upload the pickle file to Firebase Storage
    pkl_blob = storage.child(f"{storage_folder}pkl/face_encodings.pkl")
    pkl_blob.put("face_encodings.pkl")


@app.post('/')
async def scoring_endpoint(item:ImgInput):
    result = recognize_face(item.image_url)
    return result


@app.post('/user/')
async def scoring_endpoint(item:ImgSave):
    add_face(item.image_url, item.user_name)
    return ImgOutput(label="User Saved")

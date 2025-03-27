# %%
import face_recognition
import cv2
import numpy as np
import pathlib
from random import choice

MEMO:dict[str, int] = {}
ENCODING:list[np.ndarray] = []

def load_images(path: pathlib.Path):
    for i,j in enumerate(path.iterdir()):
        name = j.name
        MEMO[i] = name
        
        random_image = choice(list(j.iterdir()))
        ENCODING.append(next(iter(face_recognition.face_encodings(face_recognition.load_image_file(str(random_image))))))

def get_random(path: pathlib.Path):
    name = choice(list(path.iterdir()))
    
    return choice(list(name.iterdir()))

def preprocess(imagePath: pathlib.Path):
    input_image = cv2.imread(imagePath)
    lower_image = cv2.resize(input_image,(0,0),fx=0.25,fy=0.25)
    # rgb_image = cv2.cvtColor(lower_image, cv2.COLOR_BGR2RGB)
    return input_image, lower_image

def preprocessFrame(frame: pathlib.Path):
    lower_image = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    # rgb_image = cv2.cvtColor(lower_image, cv2.COLOR_BGR2RGB)
    return lower_image

def get_faces(image):
    locs =  face_recognition.face_locations(image)
    return locs, face_recognition.face_encodings(image)

def compare_faces(encodes):
    identities = []
    
    for i in encodes:
        matches = face_recognition.compare_faces(ENCODING, i)
        dist = face_recognition.face_distance(ENCODING, i)
        
        if(matches[np.argmin(dist)]):
            found = MEMO[np.argmin(dist)]
        else:
            found = "Unknown User"

        identities.append(found)
    
    return identities

def display_result(image, locations, names):
    for (top, right, bottom, left), name in zip(locations, names):
        top*=4
        right*=4
        bottom*=4
        left*=4
        
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    return image

# load_images(pathlib.Path.joinpath(pathlib.Path.cwd(),"Original Images"))
# org, image = preprocess(get_random(pathlib.Path.joinpath(pathlib.Path.cwd(),"Original Images")))

# location, encoding = get_faces(image)
# names = compare_faces(encoding)
# display_result(org,location,names)





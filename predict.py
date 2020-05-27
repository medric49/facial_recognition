import cv2
import keras.models as models
import numpy as np
from utils import get_people, WIDTH, HEIGHT

people = get_people()
class_number = len(people)

image = cv2.imread('test.jpg')
image = [cv2.resize(image, (WIDTH, HEIGHT))]
image = np.array(image)
model = models.load_model("model_state.pt")

predict = model.predict(image)

print("========")

print(people[predict.argmax(axis=1)[0]])

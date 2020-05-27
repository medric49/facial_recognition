import os
import cv2
import numpy as np
from network import create_model

WIDTH = 150
HEIGHT = 150

data_dir = 'images'
people = []

for f in os.listdir(data_dir):
    folder = "/".join([data_dir, f])
    if os.path.isdir(folder):
        people.append(f)

class_number = len(people)

if class_number == 0:
    exit()

x_train = []
y_train = []

image_class = -1

for person in people:
    folder = "/".join([data_dir, person])
    image_class += 1

    for image in os.listdir(folder):
        image = "/".join([folder, image])
        image = cv2.imread(image)
        image = cv2.resize(image, (WIDTH, HEIGHT))

        x_train.append(image)
        y_train.append([1 if i == image_class else 0 for i in range(class_number)])

x_train = np.array(x_train)
y_train = np.array(y_train)


model = create_model(WIDTH, HEIGHT, class_number)
model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.15, shuffle=True)

model.save('model_state.pt')


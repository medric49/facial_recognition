import os

WIDTH = 150
HEIGHT = 150
data_dir = 'images'


def get_people():
    people = []
    for f in os.listdir(data_dir):
        folder = "/".join([data_dir, f])
        if os.path.isdir(folder):
            people.append(f)
    return people



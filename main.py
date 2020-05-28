from __future__ import print_function, division, generators, unicode_literals

import argparse
import math
import os
import sys
import cv2
from utils import WIDTH, HEIGHT, get_people
import keras.models as models
import numpy as np

# CV compatibility stubs
if 'IMREAD_GRAYSCALE' not in dir(cv2):
    # <2.4
    cv2.IMREAD_GRAYSCALE = 0
if 'cv' in dir(cv2):
    # <3.0
    cv2.CASCADE_DO_CANNY_PRUNING = cv2.cv.CV_HAAR_DO_CANNY_PRUNING
    cv2.CASCADE_FIND_BIGGEST_OBJECT = cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT
    cv2.FONT_HERSHEY_SIMPLEX = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, cv2.cv.CV_AA)
    cv2.LINE_AA = cv2.cv.CV_AA


    def getTextSize(buf, font, scale, thickness):
        return cv2.cv.GetTextSize(buf, font)


    def putText(im, line, pos, font, scale, color, thickness, lineType):
        return cv2.cv.PutText(cv2.cv.fromarray(im), line, pos, font, color)


    cv2.getTextSize = getTextSize
    cv2.putText = putText

# Profiles
DATA_DIR = 'opencv/'
CASCADES = {}

PROFILES = {
    'HAAR_FRONTALFACE_ALT2': 'haarcascades/haarcascade_frontalface_alt2.xml',
}


# Support functions
def error(msg):
    sys.stderr.write("{}: error: {}\n".format(os.path.basename(sys.argv[0]), msg))


def fatal(msg):
    error(msg)
    sys.exit(1)


def load_cascades(data_dir):
    for k, v in PROFILES.items():
        v = os.path.join(data_dir, v)
        try:
            if not os.path.exists(v):
                raise cv2.error('no such file')
            CASCADES[k] = cv2.CascadeClassifier(v)
        except cv2.error:
            fatal("cannot load {} from {}".format(k, v))


def face_detect(im):
    side = math.sqrt(im.size)
    min_len = int(side / 20)
    max_len = int(side / 2)
    flags = cv2.CASCADE_DO_CANNY_PRUNING

    # frontal faces
    cc = CASCADES['HAAR_FRONTALFACE_ALT2']
    features = cc.detectMultiScale(im, 1.1, 4, flags, (min_len, min_len), (max_len, max_len))
    return features


def face_detect_file(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        fatal("cannot load input image {}".format(path))
    im = cv2.equalizeHist(im)
    features = face_detect(im)
    return im, features


def __main__():
    ap = argparse.ArgumentParser(description='A simple face detector for batch processing')
    ap.add_argument('file', help='Input image file')
    args = ap.parse_args()

    load_cascades(DATA_DIR)

    _, features = face_detect_file(args.file)

    im = cv2.imread(args.file)
    output = cv2.imread(args.file)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.80
    font_color = (255, 255, 255)
    line_type = 2

    people = get_people()
    model = models.load_model("model_state.pt")

    for i in range(len(features)):
        rect = features[i]

        xy1 = (rect[0], rect[1])
        xy2 = (rect[0] + rect[2], rect[1] + rect[3])

        image = im[rect[1]:rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
        image = [cv2.resize(image, (WIDTH, HEIGHT))]
        image = np.array(image)

        prediction = model.predict(image)
        predicted_person = people[prediction.argmax(axis=1)[0]]
        percentage = round(prediction.max(axis=1)[0] * 100, 2)
        text = "{} ({}%)".format(predicted_person, percentage)

        cv2.rectangle(output, xy1, xy2, (0, 0, 0), 4)
        cv2.rectangle(output, xy1, xy2, (255, 255, 255), 2)

        cv2.putText(output, text, (rect[0], rect[1] - 5), font, font_scale, font_color, line_type)

    cv2.imwrite("output.png", output)

    return 0


if __name__ == '__main__':
    sys.exit(__main__())

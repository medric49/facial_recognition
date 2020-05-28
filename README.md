# Facial Recognition
A program created to train a model in order to recognize some people on an image. This model will be trained from  photos of these people, and after it will be able to recognize any of these people on any image.
This program uses **[Keras](https://keras.io/)** to create its neural network and **[OpenCV](https://opencv.org/)** (cv2) for image processing.

## How do we train the model ?
First of all, we should add images of heads of these people in the folder **images/**. Each group of photos (
preferably the photos of head) of each person must be placed in a folder named by this person's name.
After that, we must train our model by executing this command in the project root
```shell
   python3 trainer.py
```

## How do we use the model for recognition ?
After that our model be trained, we must execute this command in the project root
```shell
   python3 main.py path/to/image.jpg
```
Where **path/to/image.jpg** is the path to the image whose we want to test.

## Example

<img src="before.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />
<img src="after.png"
     style="float: right; margin-right: 10px;" />

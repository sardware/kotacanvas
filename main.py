import gradio as gr
import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

def first(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30))
    center_image = image
    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        center_image = image[y:y+h, x:x+w]

    return center_image, image


if __name__ == '__main__':
    gr.Interface(
        fn=first,
        inputs=['image'],
        outputs=['image', 'image']
    ).launch()

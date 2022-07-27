import tensorflow as tf
import cv2
import numpy as np

model = tf.lite.Interpreter('color_model.tflite')
id = model.get_input_details()[0]
od = model.get_output_details()[0]

def run(image):
    h, w, _ = image.shape
    size = h * w
    model.resize_tensor_input(id['index'], (size, 3))
    model.allocate_tensors()

    model.set_tensor(id['index'], image.reshape(-1, 3))
    model.invoke()
    dst = model.get_tensor(od['index'])
    dst = dst.reshape(image.shape)
    return dst

if __name__ == '__main__':
    image = cv2.imread('meercat.jpg')
    dst = run(image)
    cv2.imshow('src', image)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
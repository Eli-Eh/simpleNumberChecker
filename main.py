import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.load_model("handwritten.keras")
image_number = 1

while os.path.isfile(f"digits/dig_{image_number}.png"):
    try:
        image = cv2.imread(f"digits/dig_{image_number}.png") [:, :, 0]
        image = np.invert(np.array([image]))
        prediction = model.predict(image)
        print(f"eh its like a uhhh {np.argmax(prediction)}")
        print()
        plt.imshow(image[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print("there was an error D:")
    finally:
        image_number += 1
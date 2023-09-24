import cv2
import numpy
import tensorflow as tf

cam=cv2.VideoCapture(0)
model=tf.keras.models.load_model("keras_model.h5")

while True:
    ret,frame=cam.read()
    img=cv2.resize(frame,(224,224))
    img=numpy.array(img,dtype=numpy.float32)
    img=numpy.expand_dims(img,axis=0)
    img=img/255.0
    result=model.predict(img)
    print(result)

    cv2.imshow("webcam",frame)
    if cv2.waitKey(23)==32:
        break
cv2.destroyAllWindows()
cam.release()
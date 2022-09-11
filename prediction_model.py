from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models import CTCLayer, CRNN
from data_captcha import Dataset_cap


class Captcha_reader(CRNN):
    def __init__(self, data_class, num_to_char, epoch=50):
        CRNN.__init__(self)
        self.model = CRNN.__train_model__(self)
        self.data_class = data_class
        self.num_to_char = num_to_char
        self.epoch = epoch
        self.X_train, self.X_val, self.y_train, self.y_val = self.data_class.data()

    def plot_loss(self):
        self.history = self.model.fit([self.X_train, self.y_train], epochs=self.epoch, )
        self.model_pred = self.predictive_model(self.model)
        self.model.save('/content/drive/MyDrive/model_train')
        self.model_pred.save('/content/drive/MyDrive/model_pred')
        plt.plot(self.history.history['loss'])
        plt.title('model loss')
        plt.ylabel('CTC loss')
        plt.xlabel('epoch')
        plt.legend(['train'])
        plt.show()

    def predict_test(self):
        self.prediction_model = keras.models.load_model('/content/drive/MyDrive/model_pred')
        self.prediction_model.summary()
        y_pred = self.prediction_model.predict(self.X_val)
        y_pred = keras.backend.ctc_decode(y_pred, input_length=np.ones(107)*50, greedy=True)
        y_pred = y_pred[0][0][0:107,0:5].numpy()
        acc=0
        for n, i in enumerate(y_pred):
            check = 0
            for l, p in enumerate(i):
                if p == self.y_val[n,l]:
                  check+=1
            if check == 5:
                acc+=1
        print("acccuracy: ", acc/1.07)
        print('/n')
        nrow = 1
        fig=plt.figure(figsize=(20, 5))
        for i in range(0,10):
            if i>4: nrow = 2
            fig.add_subplot(nrow, 5, i+1)
            plt.imshow(tf.squeeze(self.X_val[i].transpose((1,0,2))),cmap='gray')
            plt.title('Prediction : ' + str(list(map(lambda x:self.num_to_char[x], y_pred[i]))))
            plt.axis('off')
        plt.show()
    def predict_image(self, path, crop=False):
        img = tf.io.read_file(path)
        img = tf.io.decode_png(img, channels=1)
        # make tensors float32
        img = tf.image.convert_image_dtype(img, tf.float32)
        # crop image or not?
        if(crop==True):
            img = tf.image.crop_to_bounding_box(img, offset_height=0, offset_width=25, target_height=50, target_width=125)
            img = tf.image.resize(img,size=[50,200],method='bilinear', preserve_aspect_ratio=False,antialias=False, name=None)
        img = tf.transpose(img, perm=[1, 0, 2])
        y_pred_img = self.prediction_model.predict(img)
        y_pred_img = keras.backend.ctc_decode(y_pred_img, input_length=np.ones(1)*50, greedy=True).numpy()
        plt.imshow(img)
        plt.title('Prediction : ' + str(list(map(lambda x:self.num_to_char[x],y_pred_img))))
        plt.axis('off')
        plt.show()


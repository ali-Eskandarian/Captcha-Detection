from models import CTCLayer, CRNN
from data_captcha import Dataset_cap
from prediction_model import Captcha_reader
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

char_to_nums = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6,
               '8': 7, '9': 8,'0': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13,
               'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19, 'k': 20,
               'l': 21, 'm': 22, 'n':23, 'o': 24, 'p':25, 'q': 26, 'r': 27,
               's': 28,'t': 29, 'u': 30, 'v': 31, 'w': 32, 'x': 33, 'y': 34, 'z': 35}
img_folder_path = 'data/'

def main(data=img_folder_path, char_to_num=char_to_nums):
    val = input("Enter your value: ")
    num_to_char = {y: x for x, y in char_to_num.items()}
    data = Dataset_cap(img_folder_path, char_to_num)
    predictor = Captcha_reader(data, num_to_char, epoch= 10)
    if val==1:
      predictor.plot_loss()
    elif val==2:
      predictor.predict_test()
    elif val==3:
      predictor.model.summary()
  
if __name__ == '__main__':
    main()

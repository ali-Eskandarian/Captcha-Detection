from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


class CRNN():
    def __init__(self):
        self.model = self.__train_model__()

    def __train_model__(self):
        # Inputs to the model
        input_img = layers.Input(shape=(200, 50, 1), name="image", dtype="float32")
        labels = layers.Input(name="label", shape=(None,), dtype="float32")

        # First conv block
        x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(
            input_img)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)

        # Second conv block
        x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(
            x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        x = layers.Reshape(target_shape=(50, 768), name="reshape")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)

        # RNNs
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        # Output layer
        x = layers.Dense(37, activation="softmax", name="dense2")(x)  # 37 = 36 characters + UKN

        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        self.model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="ocr_cnn_lstm_model")

        # Compile the model and return
        self.model.compile(optimizer=keras.optimizers.Adam())
        return self.model

    def predictive_model(self, model_main):
        prediction_model = keras.models.Model(
            model_main.get_layer(name="image").input, model_main.get_layer(name="dense2").output)
        return prediction_model
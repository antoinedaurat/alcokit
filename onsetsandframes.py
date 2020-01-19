import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, \
                                    LSTM, Bidirectional, Dropout, BatchNormalization, \
                                    Reshape, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import VarianceScaling
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adadelta


def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_default_hparams():
  """Returns the default hyperparameters.

  Returns:
    A tf.contrib.training.HParams object representing the default
    hyperparameters for the model.
  """
  return dict(
      batch_size=8,
      learning_rate=1.,
      decay_rate=0.95,
      clip_norm=3.0,
      num_pitches=88,
      lstm_units=64,
      stop_onset_gradient=True,
      stop_offset_gradient=True,
      kernels=((3, 3), (3, 3), (3, 3)),
      num_filters=[48, 48, 48],
      pool_sizes=[1, 2, 2],
      dropout_cnn=[0., .25, .25],
      fc_size=128,
      fc_dropout_keep_amt=0.5,
      max_length=None,
      predict_frame_threshold=0.5,
      predict_onset_threshold=0.5,
      predict_offset_threshold=0,
  )

def get_inputs(x):
    if len(x.shape) == 4:
        return Input(shape=x.shape[1:])
    else:
        return Input(shape=(*x.shape[1:], 1))

def conv_net(inputs, hparams):
    initializer = VarianceScaling(scale=2., mode="fan_avg", distribution="uniform")
    x = inputs
    for filters, kernel, pool, dropout in zip(
            hparams["num_filters"], hparams["kernels"],
            hparams["pool_sizes"], hparams["dropout_cnn"]):
        
        x = Conv2D(activation="relu",
                    filters=filters,
                    kernel_size=kernel,
                    kernel_initializer=initializer,
                    bias_initializer=initializer,
                    padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(1, pool), strides=(1, pool))(x)
        x = Dropout(rate=dropout)(x)
    dims = tf.shape(x)
    print(dims, dims[0], x.shape[2].value, x.shape[3].value)
    x = Reshape((-1, x.shape[2].value * x.shape[3].value))(x)
    x = Dense(units=hparams["fc_size"], activation="relu",
            kernel_initializer=initializer,
            bias_initializer=initializer)(x)
    x = Dropout(rate=0.5)(x)
    return x

def bilstm(inputs, hparams):
    x = inputs
    x = Bidirectional(
        LSTM(units=hparams["lstm_units"],
            input_shape=(hparams["max_length"], x.shape[-1]),
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True))(x)
    return x


def acoustic_model(inputs, hparams):
    return bilstm(conv_net(inputs, hparams), hparams)
    # return conv_net(inputs, hparams)

def model(inputs, hparams):
    onset_outputs = acoustic_model(inputs, hparams)

    ## last activation == "tanh" -> on, + / off, -
    y_hat = Dense(
          units=hparams["num_pitches"],
          activation="sigmoid")(onset_outputs)

    def cross_entropy(target, y_hat):
        return -K.sum( \
            (target*K.log(y_hat + 1e-9)) + ((1 - target) * K.log(1 - y_hat + 1e-9)))

    onset_predictions = y_hat > hparams["predict_onset_threshold"]

    train_op = Adadelta(
            learning_rate=hparams["learning_rate"], 
            rho=hparams["decay_rate"],
            clipnorm=hparams["clip_norm"])
    model = Model(inputs=inputs, outputs=y_hat)
    model.compile(optimizer="adam",
                loss="binary_crossentropy",
                metrics=['accuracy', recall, precision, f1])

    return model



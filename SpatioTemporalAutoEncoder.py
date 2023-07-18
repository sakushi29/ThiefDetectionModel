from tensorflow import keras
from keras.layers import Input,Conv3D,ConvLSTM2D,Conv3DTranspose
from keras.models import Model

class SpatioTemporalAutoEncoder(Model):
  def __init__(self):
    super(SpatioTemporalAutoEncoder, self).__init__()
    self.enc = keras.Sequential([
      Input(shape=(227, 227, 10, 1)),
      Conv3D(filters= 128, kernel_size= (11, 11, 1), strides= (4, 4, 1), padding='valid', activation='relu'),
      Conv3D(filters=64, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='valid', activation='relu')])
    
    self.btnk = keras.Sequential([
      ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', dropout=0.4, recurrent_dropout=0.3, return_sequences=True),
      ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', dropout=0.3, return_sequences=True),
      ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, return_sequences=True, padding='same', dropout=0.5)])
    
    self.dec = keras.Sequential([
      Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='relu'),
      Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='relu')])

  def call(self, inp):
    encoded_inp = self.enc(inp)
    reduced_inp = self.btnk(encoded_inp) 
    decoded_inp = self.dec(reduced_inp)
    return decoded_inp

def loadModel():
    model = SpatioTemporalAutoEncoder() 
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

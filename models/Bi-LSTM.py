import tensorflow
import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Embedding,Activation,Dropout,SpatialDropout1D,Bidirectional,LSTM,SimpleRNN
from tensorflow.keras.layers import Conv1D,MaxPooling1D,GlobalAveragePooling1D,GlobalMaxPooling1D

import pandas as pd
from sklearn.model_selection import train_test_split import nltk
from nltk.corpus import stopwords import re
from nltk.stem import WordNetLemmatizer

#Activation Functions

#sigmoid
def sigmoid(X):
    return 1/(1+np.exp(-X))

#tanh 
def tanh_activation(X):
    return (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))

#softmax activation
def softmax(X):
    exp_X = np.exp(X)
    exp_X_sum = np.sum(exp_X,axis=1).reshape(-1,1)
    soft_exp_X = exp_X/exp_X_sum
    return soft_exp_X
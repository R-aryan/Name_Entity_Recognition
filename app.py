import numpy as np
from flask import Flask, request, jsonify, render_template
from flask import request
import pickle as pk
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import unicodedata
import spacy
import h5py
import tensorflow as tf
# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True
 
from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from keras.models import Model, Input
# from keras.layers import LSTM, Embedding, Dense
# from keras.layers import TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.models import load_model
# from keras.preprocessing.text import Tokenizer



##############################################################
app = Flask(__name__)
my_model = spacy.load("en_ner_bc5cdr_md")
model = pk.load(open('my_NER_disease_model.pkl', 'rb'))
# with tf.device('/cpu:0'):
Model = load_model('my_NER_model.h5')

##############################################################
    
word2idx ={}

# words=[]
# print(type(words))

t=['O' ,'B-indications' ,'I-indications']
tags=list(t)


# Creating tags to indices dictionary.
tag2idx = {t: i for i, t in enumerate(tags)}

 
# Maximum length of text sentences
MAXLEN = 180
# batch size
BS=48

#############Preprocessing functions##############
def predict_output(X_test,words):

  # Predicting on trained model
  pred = Model.predict(X_test)
  print("Predicted Probabilities on Test Set:\n",pred.shape)
  # taking tag class with maximum probability
  pred_index = np.argmax(pred, axis=-1)
  print("Predicted tag indices: \n",pred_index.shape)


  # Flatten both the features and predicted tags
  ids,tagids = X_test.flatten().tolist(), pred_index.flatten().tolist()
 
  # converting each word indices back to words
  words_test = [words[ind].decode('utf-8') for ind in ids]
  # converting each predicted tag indices back to tags
  tags_test = [tags[ind] for ind in tagids]
  #print(tagids)
  print("Length of words in Padded test set:",len(words_test))
  print("Length of tags in Padded test set:",len(tags_test))
  print("\nCheck few of words and predicted tags:\n",words_test[:10],tags_test[:10])
  return words_test

####################################################

def convert_input(str_test):


  test=str_test

  str_test = list(str_test.split())
  words= list(set(str_test))
  words.append("ENDPAD")
  n_words = len(words)

  words= [unicodedata.normalize('NFKD', str(w)).encode('ascii','ignore') for w in words]

  # Creating words to indices dictionary.
  word2idx = {w: i for i, w in enumerate(words)}

  X_test = [[word2idx[unicodedata.normalize('NFKD', str(w)).encode('ascii','ignore')] for w in str_test]]

  X_test = pad_sequences(maxlen=MAXLEN, sequences=X_test, padding="post", value=n_words - 1)

  #res = predict_output(X_test,words)

  return model(test)

  

  
###############################################################################

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():



    str_test= str(request.form.get('rate'))
    

    prediction = convert_input(str_test)

    if len(list(prediction.ents))==0:
        return render_template('home.html', prediction_text='No disease name present')
         
        


    return render_template('home.html', prediction_text='Disease Names are  {}'.format(list(prediction.ents)))


if __name__ == "__main__":
    app.run(threaded=False,debug=True)



from django.http import HttpResponse
from rest_framework.decorators import api_view
import json
import pickle
import numpy as np
import nltk
import demoji
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from django.apps import apps
from gensim.models import KeyedVectors


from rest_framework import status
from rest_framework.response import Response
from transformers import AutoTokenizer, RobertaTokenizer
from transformers import TFRobertaModel
from transformers import BertTokenizer,TFBertModel
from transformers import XLNetTokenizer, TFXLNetModel
import tensorflow as tf


lables = {
    7:"Neutral",
    3:"Fear",
    2:"Disgust",
    5:"Sad",
    6:"Surprise",
    1:"Anger",
    4:"Joy"
}

def get_vector_from_embedding(text):
  # Get Word Vectors
    vec = 0 
    wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')
    for word in text:
      if(word in wv):
        vec = vec+np.array(wv[word])
    if type(vec)  == type(np.array([])):
      return vec
    else:
      return np.zeros(300)

def tokenize_remove_stop_words(text):
    # Get Stop Words 
    stopWords = set(stopwords.words('english'))
    stopWords.add('[NAME]')
    # Tokenize Corpus into Words
    corpus_without_emojis = demoji.replace_with_desc(text,sep="")
    words = word_tokenize(corpus_without_emojis)
    # Filter Stopping
    wordsFiltered = []
    for w in words:
      if w not in stopWords:
        wordsFiltered.append(w.lower())
    return wordsFiltered


def stem(text):
    porter_stemmer  = PorterStemmer()
    # Stem Words
    for col,token in enumerate(text):
      text[col]=porter_stemmer.stem(token)


def tokonize(sentences, tokenizer, max_len):
    input_ids, attention_mask = [],[]
    inputs = tokenizer.encode_plus(sentences, add_special_tokens=True,   max_length=max_len, pad_to_max_length=True,
                                       return_attention_mask=True)

    input_ids.append(inputs['input_ids'])
    attention_mask.append(inputs['attention_mask'])
          
    return np.array(input_ids, dtype='int32'), np.array(attention_mask, dtype='int32')

def tokenize_data(tweet, tokenizer):
    '''
    Toknize Data sets 
    and return input_ids,attention masks  and labels
    '''
    max_len = 40
    input_ids , attention_masks = tokonize(tweet,tokenizer, max_len)
    return input_ids,attention_masks

def pool_hidden_state(last_hidden_state):
    """
    Pool the output vectors into a single mean vector 
    """
    last_hidden_state = last_hidden_state[0]
    mean_last_hidden_state = tf.reduce_mean(last_hidden_state, 1)

    return mean_last_hidden_state

def create_model(model, max_len,pool=False):
    input_ids = tf.keras.Input(shape=(max_len,),dtype='int32')
    attention_masks = tf.keras.Input(shape=(max_len,),dtype='int32')

    output = model([input_ids,attention_masks])
    
    if(pool == True ):
      output = pool_hidden_state(output)
    
    else:
      output = output[1]
    
    output = tf.keras.layers.Dense(7, activation='softmax')(output)
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

@api_view(['POST'])
def index(request):
    
        #text = json.loads(request.body)["text"]
        ##tokenized = tokenize_remove_stop_words(text)
        #stem(tokenized)
        #vector = get_vector_from_embedding(tokenized)
        #print(vector)
        #with open('saved_models/logisticRegressionModel.pkl', 'rb') as f:
        #    clf2 = pickle.load(f)
        #prediction = clf2.predict([vector])
        #return HttpResponse(lables[prediction[0]])

  try:

        map = {0:'anger' , 1 : 'disgust' , 2 : 'fear' , 3 : 'joy' , 4: 'sadness' , 5: 'surprise', 6: 'neutral'}
        max_len = 40

        # run on Roberta
        roberta_tokenizer = RobertaTokenizer.from_pretrained(
            'roberta-large',
             add_special_tokens=True,
            max_length=max_len, 
            pad_to_max_length=True)  # Tokenizer

        
        roberta_model = TFRobertaModel.from_pretrained('roberta-large')
        model = create_model(roberta_model, max_len)
        #model.summary()
        model.load_weights("saved_models/Roberta_weights.h5")
        
        # run on Bert
        bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-large-cased',
             add_special_tokens=True,
            max_length=max_len, 
            pad_to_max_length=True) #Tokenizer

        bert_model = TFBertModel.from_pretrained('bert-large-cased')
        bert_model = create_model(bert_model, max_len)
        #bert_model.summary()
        bert_model.load_weights("saved_models/Bert.h5")

        # run on Xlnet
        XLnet_tokenizer = AutoTokenizer.from_pretrained(
            'xlnet-large-cased', 
            add_special_tokens=True,
            max_length=max_len, 
            pad_to_max_length=True)


        Xlnet_model = TFXLNetModel.from_pretrained('xlnet-large-cased')
        Xlnet_model = create_model(Xlnet_model, max_len,pool=True)
        #Xlnet_model.summary()
        Xlnet_model.load_weights("saved_models/XLnet.h5")



        tweet = json.loads(request.body)["text"]


        tweet_ids_bert, tweet_mask_bert = tokenize_data(tweet, bert_tokenizer)
        predict_bert = bert_model.predict([tweet_ids_bert,tweet_mask_bert])
        bert_prediction = map[list(predict_bert[0]).index(max(predict_bert[0]))]

        tweet_ids_roberta, tweet_mask_roberta = tokenize_data(tweet, roberta_tokenizer)
        predict_roberta = model.predict([tweet_ids_roberta,tweet_mask_roberta])
        roberta_prediction = map[list(predict_roberta[0]).index(max(predict_roberta[0]))]
        
        tweet_ids_XLnet, tweet_mask_XLnet = tokenize_data(tweet, XLnet_tokenizer)
        predict_XLnet = Xlnet_model.predict([tweet_ids_XLnet,tweet_mask_XLnet])
        XLnet_prediction = map[list(predict_XLnet[0]).index(max(predict_XLnet[0]))]

       # print(bert_prediction, roberta_prediction, XLnet_prediction)
        return HttpResponse("Bert Prediction {} \n Roberta Prediction {} \n XLnet Prediction {}".format(bert_prediction, roberta_prediction, XLnet_prediction))
  except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)




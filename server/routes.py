# Create your views here.
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

@api_view(['POST','GET'])
def index(request):
    if request.method == "POST":
        text = json.loads(request.body)["text"]
        tokenized = tokenize_remove_stop_words(text)
        stem(tokenized)
        vector = get_vector_from_embedding(tokenized)
        print(vector)
        with open('saved_models/logisticRegressionModel.pkl', 'rb') as f:
            clf2 = pickle.load(f)
        prediction = clf2.predict([vector])
        return HttpResponse(lables[prediction[0]])
    return HttpResponse("Get request received")


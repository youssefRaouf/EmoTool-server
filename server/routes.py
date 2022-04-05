from rest_framework.response import Response
from rest_framework import status
from gensim.models import KeyedVectors
from django.apps import apps
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from django.http import HttpResponse
from rest_framework.decorators import api_view
import json
import numpy as np
import nltk
import demoji
import tweepy
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lablesForClassical = {
    7: "neutral",
    3: "fear",
    2: "disgust",
    5: "sadness",
    6: "surprise",
    1: "anger",
    4: "joy"
}


def get_vector_from_embedding(text):
  # Get Word Vectors
    vec = 0
    wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')
    for word in text:
        if(word in wv):
            vec = vec+np.array(wv[word])
    if type(vec) == type(np.array([])):
        return vec
    else:
        return np.zeros(300)


def tokenize_remove_stop_words(text):
    # Get Stop Words
    text = text.lower()
    stopWords = set(stopwords.words('english'))
    stopWords.add('[NAME]')
    # Tokenize Corpus into Words
    corpus_without_emojis = demoji.replace_with_desc(text, sep="")
    words = word_tokenize(corpus_without_emojis)
    # Filter Stopping
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w.lower())
    return wordsFiltered


def stem(text):
    porter_stemmer = PorterStemmer()
    # Stem Words
    for col, token in enumerate(text):
        text[col] = porter_stemmer.stem(token)


def tokonize(sentences, tokenizer, max_len):
    input_ids, attention_mask = [], []
    for sentence in sentences:
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True,   max_length=max_len, pad_to_max_length=True,
                                       return_attention_mask=True)
        input_ids.append(inputs['input_ids'])
        attention_mask.append(inputs['attention_mask'])
    # input_ids.append(inputs['input_ids'])
    # attention_mask.append(inputs['attention_mask'])

    return np.array(input_ids, dtype='int32'), np.array(attention_mask, dtype='int32')


def tokenize_data(tweet, tokenizer):
    '''
    Toknize Data sets 
    and return input_ids,attention masks  and labels
    '''
    max_len = 40
    input_ids, attention_masks = tokonize(tweet, tokenizer, max_len)
    return input_ids, attention_masks


def classify_tweet(tweet):
    server_config = apps.get_app_config('server')

    roberta_tokenizer = server_config.roberta_tokenizer

    roberta_model = server_config.roberta_model

    map = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy',
           4: 'sadness', 5: 'surprise', 6: 'neutral'}
  

    tweet_ids_roberta, tweet_mask_roberta = tokenize_data(
        [tweet], roberta_tokenizer)
    predict_roberta = roberta_model.predict(
        [tweet_ids_roberta, tweet_mask_roberta])
    roberta_prediction = map[list(
        predict_roberta[0]).index(max(predict_roberta[0]))]

    return roberta_prediction


@api_view(['POST'])
def index(request):
    try:
        tweet = json.loads(request.body)["text"]
        prediction = classify_tweet(tweet)
        return HttpResponse("Classification {}".format(prediction))
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def get_tweets(request):
    body = json.loads(request.body)
    userId = body['userId']
    # Access Token of user
    accessToken = body['accessToken']
    # Access Token secret of user
    accessTokenSecret = body['accessTokenSecret']
    # Max Result to retrieve
    max_results = 2000
    if userId:
        # Auth
        auth = tweepy.OAuth1UserHandler(
            consumer_key="pqCKc0wsOiBHZ5Sfxj5Qf0OUA",
            consumer_secret="AFeIj1XGAGyHezOVm1cGAG563rUKDqAAGshRXUuBflDTM7ZwH1",
            access_token=f"{accessToken}",
            access_token_secret=f"{accessTokenSecret}"
        )
        # Call twitter api
        api = tweepy.API(auth)
        # Get user tweets
        response = api.user_timeline(id=userId, count=max_results)
        # Getting tweets from responsse
        tweets = []
        # Labeling tweets
        text = []
        for tweet in response:
            text.append(tweet.text)
        classifications = classify_tweets(text)
        for i in range(len(response)):
            labeled_tweet = {}
            labeled_tweet['tweet'] = response[i].text
            labeled_tweet['date'] = response[i].created_at.isoformat()
            labeled_tweet['label'] = classifications[i]
            tweets.append(labeled_tweet)
        tweets = json.dumps(tweets)
        return HttpResponse(tweets)
    else:
        return Response("bad request", status.HTTP_400_BAD_REQUEST)


def classify_tweets(tweets):
    server_config = apps.get_app_config('server')
    roberta_tokenizer = server_config.roberta_tokenizer
    roberta_model = server_config.roberta_model

    map = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy',
           4: 'sadness', 5: 'surprise', 6: 'neutral'}
    tweet_ids_roberta, tweet_mask_roberta = tokenize_data(
        tweets, roberta_tokenizer)
    predict_roberta = roberta_model.predict(
        [tweet_ids_roberta, tweet_mask_roberta])
    predictions = []
    for i in range(len(predict_roberta)):
        predictions.append(map[list(
            predict_roberta[i]).index(max(predict_roberta[i]))])
    return predictions


@api_view(['POST'])
def classify_multiple_tweets(request):
    body = json.loads(request.body)
    tweets = body['tweets']
    result = []
    text = []
    # Labeling tweets
    for tweet in tweets:
        text.append(tweet['text'])
    classifications = classify_tweets(text)
    for i in range(len(tweets)):
        labeled_tweet = {}
        labeled_tweet['id'] = tweets[i]['id']
        labeled_tweet['text'] = tweets[i]['text']
        labeled_tweet['label'] = classifications[i]
        result.append(labeled_tweet)
    result = json.dumps(result)
    return HttpResponse(result)

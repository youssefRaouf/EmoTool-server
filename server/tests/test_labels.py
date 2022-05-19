from django.test import SimpleTestCase
from server.routes import classify_tweet, classify_tweets
class TestLabels(SimpleTestCase):
    def test_ClassifySingleTweet(self):

        tweet_dict = [{'text':'I love you',
                       'label':'joy'},

                      {'text':'I am very angry',
                       'label':'anger'
                      }]

        # Assert
        for tweet in tweet_dict:
            self.assertEquals(classify_tweet(tweet['text']),tweet['label'])




    def test_ClassiftMultipleTweets(self):
        tweet_dict = {
        'texts':[
            'I love you',
            "I am very angry"
        ],
        'labels':[
            'joy',
            'anger'
                      ]}

        self.assertEquals(classify_tweets(tweet_dict['texts']),tweet_dict['labels'])










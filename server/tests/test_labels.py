from django.test import SimpleTestCase
from server.routes import classify_tweet, classify_tweets

ClassifySingle_tweet_dict = [{'text': 'I love you',
                              'label': 'joy'},

                             {'text': 'I am very angry',
                              'label': 'anger'
                              }]

ClassifyMultiple_tweet_dict = {
    'texts': [
        'I love you',
        "I am very angry"
    ],
    'labels': [
        'joy',
        'anger'
    ]}
class TestLabels(SimpleTestCase):


    def test_ClassifySingleTweet(self):



        # Assert
        for tweet in ClassifySingle_tweet_dict:
            self.assertEquals(classify_tweet(tweet['text']),tweet['label'])




    def test_ClassiftMultipleTweets(self):

        self.assertEquals(classify_tweets(ClassifyMultiple_tweet_dict['texts']),ClassifyMultiple_tweet_dict['labels'])










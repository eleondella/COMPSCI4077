import tweepy
import pymongo
from pymongo import MongoClient
import json
from datetime import datetime
import numpy as np
import pandas as pd
import re
import sys
import operator
import itertools
import networkx as nx
import credentials


auth = tweepy.OAuthHandler(credentials.CONSUMER_KEY, credentials.CONSUMER_SECRET)
auth.set_access_token(credentials.ACCESS_TOKEN, credentials.ACCESS_TOKEN_SECRET)

# Implement custom StreamListener
class StreamListener(tweepy.StreamListener):
    #This is a class provided by tweepy to access the Twitter Streaming API.
    collection = ''
    def __init__(self):
        self.collection = ""

    def on_connect(self):
        # Called initially to connect to the Streaming API
        print("You are now connected to the streaming API.")

    def on_error(self, status_code):
        # On error - if an error occurs, display the error / status code
        print('An Error has occured: ' + repr(status_code))
        return False

    def on_data(self, data):
        try:
            # Connects to the MongoDB
            client = MongoClient()

            # Use twitterdb database. If it doesn't exist, it will be created.
            db = client.twitterdb

            # Decode the JSON from Twitter
            datajson = json.loads(data)

            format_created_at = datetime.strptime(datajson['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
            datajson['created_at'] = format_created_at

            #insert the data into the mongoDB into a collection called "tweets"
            #if twitter_search doesn't exist, it will be created.
            db[self.collection].insert_one(datajson)
        except Exception as e:
            print(e)

# Set up the streaming API
api=tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True,parser=tweepy.parsers.JSONParser())
streamListener = StreamListener()
streamListener.collection = "tweets"

# Stream a sample of tweets with English language
myStream = tweepy.Stream(auth = api.auth, listener=streamListener)
myStream.filter(languages = ["en"], locations=[-6.38,49.87,1.77,55.81], is_async=True)

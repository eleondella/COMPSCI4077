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

# Set up the API
api=tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

# Get all the locations where Twitter provides trends service
world_trends = api.trends_available()

# Get trending topics in United Kingdom (its WOEID is 23424975 = parent id from Birmingham)
sfo_trends = api.trends_place(id =23424975)[0]['trends']
names = [trend['name'] for trend in sfo_trends]

print("The top 5 trends in the UK are:")
print(names[0])
print(names[1])
print(names[2])
print(names[3])
print(names[4])

# Use the cursor to get 5000 tweets with English language from the top 5 trends
for tweet in tweepy.Cursor(api.search,
                   q=" OR ".join(names[0:5]),lang="en").items(5000):
    while True:
        try:
            client = MongoClient()
            # Use twitterdb database. If it doesn't exist, it will be created.
            db = client.twitterdb
            #print(tweet["_json"])
            # Decode the JSON from Twitter
            datajson = tweet._json

            format_created_at = datetime.strptime(datajson['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
            datajson['created_at'] = format_created_at

            #insert the data into the mongoDB into a collection called twitter_search
            #if twitter_search doesn't exist, it will be created.
            db["tweets"].insert_one(datajson)
            break

        except tweepy.TweepError:
            time.sleep(60 * 15)
            continue

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
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from collections import Counter
import string
import credentials

#All functions for part 2, 3 and 4 will be stored here

# Part 2

# Function to put the tweets into a database and extract important properties
def get_tweet_information(tweets):
    df = pd.DataFrame(data=[tweet['text'] for tweet in tweets],columns=['text'])
    # Getting basic tweet and user information
    df['date'] = np.array([tweet["created_at"] for tweet in tweets])
    df['tweet_id'] = np.array([tweet['_id'] for tweet in tweets])
    df['username'] = np.array([tweet["user"]['screen_name'] for tweet in tweets])
    df['followers_no'] = np.array([tweet["user"]['followers_count'] for tweet in tweets])
    df['date'] = np.array([tweet["created_at"] for tweet in tweets])
    df['source'] = np.array([tweet["source"] for tweet in tweets])
    # Basic tweet analytics
    df['len'] = np.array([len(tweet['text']) for tweet in tweets])
    df['likes'] = np.array([tweet["favorite_count"] for tweet in tweets])
    df['retweets'] = np.array([tweet["retweet_count"] for tweet in tweets])
    # Getting information if tweet is a reply
    df['in_reply_to_username'] = np.array(list(tweet['in_reply_to_screen_name'] for tweet in tweets))
    # Getting information if users have been mentioned by that tweet
    df['user_mentions_username'] = np.array([[mention['screen_name'] for mention in tweet['entities']['user_mentions']] if len(tweet['entities']['user_mentions']) != 0 else None for tweet in tweets])
    # Getting information if tweet is a retweet
    df['retweeted_username'] = np.array([tweet['retweeted_status']['user']['screen_name'] if "retweeted_status" in tweet else None for tweet in tweets])
    # Getting hashtags from each tweet
    df['hashtags'] = np.array([[hashtag['text'] for hashtag in tweet['entities']['hashtags']] if len(tweet['entities']['hashtags']) != 0 else None for tweet in tweets])

    return df

# Function to clean text of tweet
def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

# Function to analyse sentiment of tweet
def analyze_sentiment(tweet):
    analysis = TextBlob(clean_tweet(tweet))

    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1
# Removes punctuation from tweets
def remove_punctuation(s):
    punctuations = string.punctuation

    no_punct = ""
    for char in s:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct

# Tokenises tweets
def simple_text_tokenisation(s):
    # Remove punctuation
    s = remove_punctuation(s)
    # Convert to lowercase
    s = s.lower()
    stop_words = set(stopwords.words('english'))
    return [w for w in s.split() if not w in stop_words]

# Part 3

# Get the interactions between the reply tweet users and the original tweet users
def get_reply_interactions(tweet):
    # From every row of the original dataframe
    # First we obtain the username
    user = tweet["username"]

    # Be careful if there is no user id
    if user is None:
        return None, []

    # The interactions are going to be a set of tuples
    reply_interactions = []

    # Add all interactions
    # First, we add the interactions corresponding to replies adding the id and screen_name
    reply_interactions.append(tweet["in_reply_to_username"])

    # Discard if user id is in interactions
    if tweet['username'] in reply_interactions:
        reply_interactions.remove(tweet["username"])
    # Discard all not existing values
    if None in reply_interactions:
        reply_interactions.remove(None)
    # Return user and interactions
    return user, reply_interactions

def get_reply_network(dataframe):
    #Creating graph for reply
    reply_graph = nx.DiGraph()
    user_reply_interactions = {}

    #Build the graph by getting interaction for each reply tweet
    for index, tweet in dataframe.iterrows():
        #Get tweets original user and the user interaction with the original tweet
        user, interactions = get_reply_interactions(tweet)
        auth_username = user
        user_reply_interactions[auth_username] = {}
        for interaction in interactions:
            int_username = interaction
            reply_graph.add_edges_from([(auth_username, int_username)])

            if int_username in user_reply_interactions[auth_username]:
                    user_reply_interactions[auth_username][int_username] += 1
            else:
                user_reply_interactions[auth_username][int_username] = 1

    return reply_graph, user_reply_interactions

# Get the interactions between the different users
def get_mention_interactions(tweet):
    # From every row of the original dataframe
    # First we obtain the username of the tweet author
    user = tweet["username"]
    # Be careful if there is no user id
    if user[0] is None:
        return None, []

    # The interactions are going to be a set of tuples
    mention_interactions = []

    # Add all interactions
    for each in range(0,len(tweet['user_mentions_username'])):
        mention_interactions.append(tweet["user_mentions_username"][each])

    # Discard if user id is in interactions
    if tweet['username'] in mention_interactions:
        mention_interactions.remove(tweet["username"])
    # Discard all not existing values
    if None in mention_interactions:
        mention_interactions.remove(None)
    # Return user and interactions
    return user, mention_interactions

def get_mention_network(dataframe):
    #Creating graph for mentions
    mention_graph = nx.DiGraph()
    user_mention_interactions = {}

    #Build the graph by getting interaction for each reply tweet
    for index, tweet in dataframe.iterrows():
        #Get tweets original user and the user interaction with the original tweet
        user, interactions = get_mention_interactions(tweet)
        auth_username = user
        user_mention_interactions[auth_username] = {}
        for interaction in interactions:
            int_username = interaction
            mention_graph.add_edges_from([(auth_username, int_username)])

            if int_username in user_mention_interactions[auth_username]:
                    user_mention_interactions[auth_username][int_username] += 1
            else:
                user_mention_interactions[auth_username][int_username] = 1
    return mention_graph, user_mention_interactions

# Get the interactions between the different users

def get_retweet_interactions(tweet):
    # From every row of the original dataframe
    # First we obtain the 'user_id' and 'screen_name'
    user = tweet["username"]
    # Be careful if there is no user id
    if user[0] is None:
        return None, []

    # The interactions are going to be a set of tuples
    retweet_interactions = []

    # Add all interactions
    if tweet['retweeted_username'] != None:
        retweet_interactions.append(tweet["retweeted_username"])

    # Discard if user id is in interactions
    if tweet['username'] in retweet_interactions:
        retweet_interactions.remove(tweet["username"])
    # Discard all not existing values
    if None in retweet_interactions:
        retweet_interactions.remove(None)
    # Return user and interactions
    return user, retweet_interactions


def get_retweet_network(dataframe):
    #Creating graph for mentions
    retweets_graph = nx.DiGraph()
    user_retweet_interactions = {}

    #Build the graph by getting interaction for each retweet 
    for index, tweet in dataframe.iterrows():
        #Get tweets original user and the user interaction with the original tweet
        user, interactions = get_retweet_interactions(tweet)
        auth_username = user
        user_retweet_interactions[auth_username] = {}
        for interaction in interactions:
            int_username = interaction
            retweets_graph.add_edges_from([(auth_username, int_username)])

            if int_username in user_retweet_interactions[auth_username]:
                    user_retweet_interactions[auth_username][int_username] += 1
            else:
                user_retweet_interactions[auth_username][int_username] = 1

    return retweets_graph, user_retweet_interactions

def get_hashtag_network(dataframe):
    #Creating graph for mentions
    hashtag_graph = nx.DiGraph()
    hashtag_interactions = {}

    for idx, tweet in dataframe.iterrows():
        if len(tweet['hashtags']) > 1:
            for hashtag in tweet['hashtags']:
                if not hashtag_interactions.get(hashtag):
                    hashtag_interactions[hashtag] = set(tweet['hashtags'])
                else:
                    hashtag_interactions[hashtag] = hashtag_interactions[hashtag].union(tweet['hashtags'])
                hashtag_interactions[hashtag].remove(hashtag)

    for k, hashtag in hashtag_interactions.items():
        for interaction in hashtag:
            int_hashtag = interaction
            hashtag_graph.add_edges_from([(k, int_hashtag)])

    return hashtag_graph, hashtag_interactions

#Get analytics of the network eg. number of edges and nodes etc.
def analyse_networks(network_dictionary, networktype):
    analysis = pd.DataFrame()

    data_type = []
    nodes_no = []
    edges_no = []
    max_degree_username = []
    max_degree = []

    for key, network in network_dictionary.items():
        if isinstance(key, int):
            data_type.append("Cluster " + str(key))
        else:
            data_type.append(key)
        nodes_no.append(network_dictionary[key]["GRAPH"].number_of_nodes())
        edges_no.append(network_dictionary[key]["GRAPH"].number_of_edges())
        degrees = np.array(list(dict(network_dictionary[key]["GRAPH"].degree).values()))
        usernames = np.array(list(dict(network_dictionary[key]["GRAPH"].degree).keys()))
        if len(degrees)==0:
            max_degree.append(0)
            max_degree_username.append(None)
        else:
            max_degree.append(max(degrees))
            max_degree_username.append(usernames[np.argmax(degrees)])

    analysis["Data Type"] = data_type
    analysis["No. of Nodes"] = nodes_no
    analysis["No. of Edges"] = edges_no
    analysis["Node with Max Degree"] = max_degree_username
    analysis["Max Degree No."] = max_degree

    return analysis

#Plot a directed graph of the network without labels
def plot_network(graph, name):
    pos = nx.spring_layout(graph, k=0.05)

    colors_central_nodes = ['blue','red']
    try:
        plt.figure(figsize = (8,8))
        nx.draw(graph, pos=pos, node_color=range(graph.number_of_nodes()), cmap=plt.cm.PiYG, edge_color="black", linewidths=0.3, node_size=60, alpha=0.6, with_labels=False)
        nx.draw_networkx_nodes(graph, pos=pos, node_size=300, node_color=colors_central_nodes)
        plt.show()
    except:
        return plt.savefig('plots/{}.png'.format(name))

# Part 4
def extract_links_triads(network_dictionary, networktype):
    analysis = pd.DataFrame()

    #Using only some triad types which we believe are important, I do a triad_census analysis
    data_type = []
    triads_no = []
    links_no = []
    no_012 = []
    no_021D = []
    no_021C = []
    no_021U = []
    no_030T = []
    no_030C = []


    for key, network in network_dictionary.items():
        if isinstance(key, int):
            data_type.append("Cluster " + str(key))
        else:
            data_type.append(key)

        # Append number of each type found into the list
        triads_no.append(nx.algorithms.triads.triadic_census(network_dictionary[key]["GRAPH"])["300"])
        links_no.append(nx.algorithms.triads.triadic_census(network_dictionary[key]["GRAPH"])["102"])
        no_012.append(nx.algorithms.triads.triadic_census(network_dictionary[key]["GRAPH"])["012"])
        no_021D.append(nx.algorithms.triads.triadic_census(network_dictionary[key]["GRAPH"])["021D"])
        no_021C.append(nx.algorithms.triads.triadic_census(network_dictionary[key]["GRAPH"])["021C"])
        no_021U.append(nx.algorithms.triads.triadic_census(network_dictionary[key]["GRAPH"])["021U"])
        no_030C.append(nx.algorithms.triads.triadic_census(network_dictionary[key]["GRAPH"])["030C"])
        no_030T.append(nx.algorithms.triads.triadic_census(network_dictionary[key]["GRAPH"])["030T"])

    analysis["Data Type"] = data_type
    analysis["No. of Fully Connected Links (102)"] = links_no
    analysis["No. of Fully Connected Triads (300)"] = triads_no
    analysis["No. of 012 Type"] = no_012
    analysis["No. of 021D Type"] = no_021D
    analysis["No. of 021C Type"] = no_021C
    analysis["No. of 021U Type"] = no_021U
    analysis["No. of 030C Type"] = no_030C
    analysis["No. of 030T Type"] = no_030T

    return analysis

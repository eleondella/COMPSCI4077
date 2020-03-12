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
from collections import Counter
import string
import credentials
import tweet_analysis
nltk.download('stopwords')

client = MongoClient()

# Get tweets from MongoDB or sample_data.json when run
data_source = sys.argv[1]
if data_source == False:
    db = client.twitterdb
    tweets = db.tweets.find()
    cleaned_tweets = []
    for tweet in tweets:
        tweet["text"] = tweet_analysis.clean_tweet(tweet["text"])
        cleaned_tweets.append(tweet)
else:
    tweets = []
    for line in open('sample_data.json', 'r'):
        tweets.append(json.loads(line))
    cleaned_tweets = []
    for tweet in tweets:
        tweet["text"] = tweet_analysis.clean_tweet(tweet["text"])
        cleaned_tweets.append(tweet)

#Put everything in a dataframe
df = tweet_analysis.get_tweet_information(cleaned_tweets)

#Removing any duplicate tweets collected by both APIs
#df.drop_duplicates(subset ="tweet_id",keep = False, inplace = True)
df['text'].replace(inplace=True,to_replace=r'RT',value=r'')
df['sentiment'] = np.array([tweet_analysis.analyze_sentiment(tweet) for tweet in df['text']])

#Part 2

# Grouped by vectorising each tweet and applying K-Means clustering to find similar tweets and put them in 10 clusters
CLUSTER_NO = 5

#Define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(stop_words="english") # stop_words="english" provided a 1% increase to classification

# Compute Tfidf matrix based on corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(df["text"]) #fit the vectorizer to tweet text

# Compute kmeans
km = KMeans(n_clusters=CLUSTER_NO)
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

df['cluster'] = clusters
group_counts = df['cluster'].value_counts().sort_index()

plt.figure()
plt.bar(np.arange(CLUSTER_NO), group_counts, label="Total Tweets")
plt.xticks(range(CLUSTER_NO))
plt.title("Tweet Clustering")
plt.xlabel("Cluster Number")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.savefig("plots/clusters.png")

hashtags_file = open('output/tophashtags.txt', "w")
top_hashtags = {}
top_mentions = {}
tag_dict = {}

#Get most frequent hashtags for the whole dataset
print ('Whole Dataset ', file=hashtags_file)
for index, tweet in df.iterrows():
        #Update the count of the hashtags
        if tweet['hashtags'] is None:
            continue
        for hashtag in tweet['hashtags']:
            if hashtag not in tag_dict:
                tag_dict[hashtag] = 1
            else:
                tag_dict[hashtag] += 1
sorted_hashtags = sorted(tag_dict.items(), key=operator.itemgetter(1), reverse=True)
top_hashtags["EVERYTHING"] = sorted_hashtags[0:11]
print ('Top 10 hashtags:', file=hashtags_file)
print ('----------------' , file=hashtags_file)
for tag in top_hashtags["EVERYTHING"]:
    print (tag[0], '-', str(tag[1]), file=hashtags_file)
if top_hashtags["EVERYTHING"] == []:
    print('No hashtags', file=hashtags_file)

print ('\n' , file=hashtags_file)


#Get most frequent hashtags per cluster
for i in range(0,CLUSTER_NO):
    # Hashtags & mentions
    tag_dict = {}
    #For each tweet in that cluster
    print ('Cluster ' + str(i), file=hashtags_file)
    for index, tweet in df.iterrows():
        if tweet['cluster'] == i:
            #Update the count of the hashtags
            if tweet['hashtags'] is None:
                continue
            for hashtag in tweet['hashtags']:
                if hashtag not in tag_dict:
                    tag_dict[hashtag] = 1
                else:
                    tag_dict[hashtag] += 1
    sorted_hashtags = sorted(tag_dict.items(), key=operator.itemgetter(1), reverse=True)
    top_hashtags[i] = sorted_hashtags[0:11]

    print ('Top 10 hashtags:', file=hashtags_file)
    print ('----------------' , file=hashtags_file)
    for tag in top_hashtags[i]:
        print (tag[0], '-', str(tag[1]), file=hashtags_file)
    if top_hashtags[i] == []:
        print('No hashtags', file=hashtags_file)

    print ('\n' , file=hashtags_file)


mentions_file = open("output/topmentions.txt","w")

#Get most frequent mentions for the whole dataframe
mention_dict = {}
#For each tweet in that cluster
for index, tweet in df.iterrows():
    if tweet['user_mentions_username'] is None:
        continue
    for mention in tweet['user_mentions_username']:
        if mention not in mention_dict:
            mention_dict[mention] = 1
        else:
            mention_dict[mention] += 1
sorted_mentions = sorted(mention_dict.items(), key=operator.itemgetter(1),reverse=True)
top_mentions[i] = sorted_mentions[0:11]

print ('Whole Dataset', file=mentions_file)
print ('Top 10 mentions:', file=mentions_file)
print ('----------------', file=mentions_file)
for tag in top_mentions[i]:
    print (tag[0], '-', str(tag[1]), file=mentions_file)

print ('\n', file=mentions_file)

#Get most frequent mentions per cluster
for i in range(0,CLUSTER_NO):
    mention_dict = {}
    #For each tweet in that cluster
    for index, tweet in df.iterrows():
        if tweet['cluster'] == i:
            #Update the count of the hashtags
            if tweet['user_mentions_username'] is None:
                continue
            for mention in tweet['user_mentions_username']:
                if mention not in mention_dict:
                    mention_dict[mention] = 1
                else:
                    mention_dict[mention] += 1
    sorted_mentions = sorted(mention_dict.items(), key=operator.itemgetter(1),reverse=True)
    top_mentions[i] = sorted_mentions[0:11]

    print ('Cluster ' + str(i), file=mentions_file)
    print ('Top 10 mentions:', file=mentions_file)
    print ('----------------', file=mentions_file)
    for tag in top_mentions[i]:
        print (tag[0], '-', str(tag[1]), file=mentions_file)

    print ('\n', file=mentions_file)


words_file = open("output/topwords.txt","w")
tokenised=[]

# Find most common words for the whole dataset
for key, value in df['text'].items():
    tokenised.append(tweet_analysis.simple_text_tokenisation(value))

print ('Whole Dataset', file=words_file)
print ('Top 10 words:',file=words_file)
print ('----------------',file=words_file)
flattentokens = [item for sublist in tokenised for item in sublist]
value_counter = Counter(flattentokens)
top_words = value_counter.most_common(10)
for word in top_words:
    print(word[0], "-", word[1], file=words_file)
print ('\n', file=words_file)


# Find most common words for each cluster
extract_clusters = {str(i):[] for i in range(CLUSTER_NO)}

for index, tweet in df.iterrows():
    extract_clusters[str(tweet['cluster'])].append(tweet['text'])

for key, value in extract_clusters.items():
    print ('Cluster ' + str(key), file=words_file)
    print ('Top 10 words:', file=words_file)
    print ('----------------', file=words_file)
    value_counter = Counter(tweet_analysis.simple_text_tokenisation(" ".join(value)))
    top_words = value_counter.most_common(10)
    for word in top_words:
        print(word[0], "-", word[1], file=words_file)
    print ('\n', file=words_file)

#Getting the means for important properties of all data
data_means = df.mean()
print("Number of tweets in total:" + str(len(df)) + "\n")
print(data_means)

for cluster in range(CLUSTER_NO):
    print("Number of tweets in Cluster " + str(cluster) + ": " + str(len(df[df["cluster"]==cluster])) + "\n")
#Getting the means for important properties of each cluster
cluster_means = df.groupby(['cluster']).mean()
cluster_means.sort_index()
print(cluster_means)

plt.figure()
plt.bar(x=cluster_means.index, height=cluster_means.likes)
plt.xticks(range(CLUSTER_NO))
plt.title("Average Likes of Tweet per Cluster")
plt.xlabel("Cluster Number")
plt.ylabel("Likes")
plt.tight_layout()
plt.savefig("plots/clusters_likes.png")

plt.figure()
plt.bar(x=cluster_means.index, height=cluster_means.len)
plt.xticks(range(CLUSTER_NO))
plt.title("Average Length of Tweet per Cluster")
plt.xlabel("Cluster Number")
plt.ylabel("Length of Tweet")
plt.tight_layout()
plt.savefig("plots/clusters_avglen.png")

plt.figure()
plt.bar(x=cluster_means.index, height=cluster_means.retweets)
plt.xticks(range(CLUSTER_NO))
plt.title("Average Retweets of Tweet per Cluster")
plt.xlabel("Cluster Number")
plt.ylabel("Retweets")
plt.tight_layout()
plt.savefig("plots/clusters_retweets.png")

plt.figure()
plt.bar(x=cluster_means.index, height=cluster_means.sentiment)
plt.xticks(range(CLUSTER_NO))
plt.title("Sentiment Analysis per Cluster")
plt.xlabel("Cluster Number")
plt.ylabel("Sentiment")
plt.tight_layout()
plt.savefig("plots/clusters_sentiment.png")

#Getting a df with just reply tweets
reply_columns = ['text', 'date', 'tweet_id', 'username', 'followers_no','in_reply_to_username','cluster']
reply_tweets = pd.DataFrame()
reply_tweets[reply_columns] = df[reply_columns]

reply_tweets = reply_tweets.where((pd.notnull(reply_tweets)), None)

reply_tweets = reply_tweets.dropna()

reply_networks = {}

# Compute user reply dictionary with interactions for the whole dataset
graph, dictionary = tweet_analysis.get_reply_network(reply_tweets)
reply_networks['EVERYTHING'] = {}
reply_networks['EVERYTHING']['DICTIONARY'] = dictionary
reply_networks['EVERYTHING']['GRAPH'] = graph
##etwork(graph)

# Compute user reply dictionary with interactions for each cluster
for cluster in range(CLUSTER_NO):
    graph, dictionary = tweet_analysis.get_reply_network(reply_tweets[reply_tweets['cluster']==cluster])
    reply_networks[cluster] = {}
    reply_networks[cluster]['DICTIONARY'] = dictionary
    reply_networks[cluster]['GRAPH'] = graph
    ##etwork(graph)

#Getting a df with just the mention data
mention_columns = ['text', 'date', 'tweet_id', 'username', 'followers_no','user_mentions_username','cluster']
mention_tweets = pd.DataFrame()
mention_tweets[mention_columns] = df[mention_columns]

mention_tweets = mention_tweets.where((pd.notnull(mention_tweets)), None)
mention_tweets = mention_tweets.dropna()

mention_networks = {}

# Compute user reply dictionary with interactions for the whole dataset
graph, dictionary = tweet_analysis.get_mention_network(mention_tweets)
mention_networks['EVERYTHING'] = {}
mention_networks['EVERYTHING']['DICTIONARY'] = dictionary
mention_networks['EVERYTHING']['GRAPH'] = graph

# Compute user reply dictionary with interactions for each cluster
for cluster in range(CLUSTER_NO):
    graph, dictionary = tweet_analysis.get_mention_network(mention_tweets[mention_tweets['cluster']==cluster])
    mention_networks[cluster] = {}
    mention_networks[cluster]['DICTIONARY'] = dictionary
    mention_networks[cluster]['GRAPH'] = graph
    ##etwork(graph)

#Getting a df with just the retweet data
retweet_columns = ['text', 'date', 'tweet_id', 'username', 'followers_no','retweeted_username','cluster']
retweet_tweets = pd.DataFrame()
retweet_tweets[retweet_columns] = df[retweet_columns]

retweet_tweets = retweet_tweets.where((pd.notnull(retweet_tweets)), None)
retweet_tweets = retweet_tweets.dropna()

retweet_networks = {}

# Compute user reply dictionary with interactions for the whole dataset
graph, dictionary = tweet_analysis.get_retweet_network(retweet_tweets)
retweet_networks['EVERYTHING'] = {}
retweet_networks['EVERYTHING']['DICTIONARY'] = dictionary
retweet_networks['EVERYTHING']['GRAPH'] = graph

# Compute user reply dictionary with interactions for each cluster
for cluster in range(CLUSTER_NO):
    graph, dictionary = tweet_analysis.get_retweet_network(retweet_tweets[retweet_tweets['cluster']==cluster])
    retweet_networks[cluster] = {}
    retweet_networks[cluster]['DICTIONARY'] = dictionary
    retweet_networks[cluster]['GRAPH'] = graph

#Getting a df with just the hashtag data
hashtag_columns = ['text', 'date', 'tweet_id', 'username', 'followers_no','hashtags','cluster']
hashtag_tweets = pd.DataFrame()
hashtag_tweets[hashtag_columns] = df[hashtag_columns]

hashtag_tweets = hashtag_tweets.dropna()
hashtag_tweets['len'] = [len(x) for x in hashtag_tweets['hashtags']]


hashtag_networks = {}

# Compute hashtag dictionary with interactions for the whole dataset
graph, dictionary = tweet_analysis.get_hashtag_network(hashtag_tweets)
hashtag_networks['EVERYTHING'] = {}
hashtag_networks['EVERYTHING']['DICTIONARY'] = dictionary
hashtag_networks['EVERYTHING']['GRAPH'] = graph
tweet_analysis.plot_network(graph,"hashtags_everything")

# Compute hashtag dictionary with interactions for each cluster
for cluster in range(CLUSTER_NO):
    graph, dictionary = tweet_analysis.get_hashtag_network(hashtag_tweets[hashtag_tweets['cluster']==cluster])
    hashtag_networks[cluster] = {}
    hashtag_networks[cluster]['DICTIONARY'] = dictionary
    hashtag_networks[cluster]['GRAPH'] = graph
    tweet_analysis.plot_network(graph, ("cluster_" + str(cluster) +"hashtags"))


#Write Network Analysis Dataframes to html file
html_file = open("output/networkstats.html", "w")
html_file.write("<h1>Retweets Network</h1>")
html_file.write(tweet_analysis.analyse_networks(retweet_networks, 'Retweets').to_html())
html_file.write("<br/><h1>Mention Network</h1>")
html_file.write(tweet_analysis.analyse_networks(mention_networks, 'Mentions').to_html())
html_file.write("<br/><h1>Reply Network</h1>")
html_file.write(tweet_analysis.analyse_networks(reply_networks, 'Replies').to_html())
html_file.write("<br/><h1>Hashtags Network</h1>")
html_file.write(tweet_analysis.analyse_networks(hashtag_networks, 'Hashtags').to_html())
html_file.close()

#Write Triad and Links Dataframes to html file
html_file = open("output/triads_links.html", "w")
html_file.write("<h1>Retweets Network</h1>")
html_file.write(tweet_analysis.extract_links_triads(retweet_networks, 'Retweets').to_html())
html_file.write("<br/><h1>Mention Network</h1>")
html_file.write(tweet_analysis.extract_links_triads(mention_networks, 'Mentions').to_html())
html_file.write("<br/><h1>Reply Network</h1>")
html_file.write(tweet_analysis.extract_links_triads(reply_networks, 'Replies').to_html())
html_file.write("<br/><h1>Hashtags Network</h1>")
html_file.write(tweet_analysis.extract_links_triads(hashtag_networks, 'Hashtags').to_html())
html_file.close()

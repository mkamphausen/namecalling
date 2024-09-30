# namecalling
a repository for a paper about the use of namecalling in tweets regarding the US elections 2020

## Data

the data used for this is project was retrieved using the twitter api in 2020. It is accessible from:
https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets/data

## Scripts

An explanation to the scripts

### en_filter.py

A script to filters out all non-english tweets in a given datasets.

### hashtagcounter

This script was used to get all hashtags as csv, ordered by their appearence.
It counts the overall appearence of the hashtag and also, how many instances of name-calling vs non-name-calling there ar for each hashtag.

### predictBiden and predictTrump

these scripts are two copies of the same script.
It takes in a csv file and checks the tweet column of given file for instances of name-calling.
For efficiency reasons it uses a batch size of 32, to fasten the process.

### duplicate_finder

A script to check how many duplicates of the same tweet appear in the data.
Duplicates may occure, when tweets are retweeted, or when they use hashtags that were relevant for both the Trump and Biden datasets.

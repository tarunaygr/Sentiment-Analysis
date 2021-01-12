from nltk.tokenize import word_tokenize
from sentiment_analysis_training import remove_noise
import pickle

f = open('sentiment.pickle', 'rb')

classifier = pickle.load(f)

f.close()
print("Enter the Tweet:\n")
custom_tweet = str(input())

custom_tokens = remove_noise(word_tokenize(custom_tweet))

print(classifier.classify(dict([token, True] for token in custom_tokens)))
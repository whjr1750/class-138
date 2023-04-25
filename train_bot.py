#Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json
import pickle
import numpy as np

words=[]
classes = []
word_tags_list = []
ignore_words = ['?', '!',',','.', "'s", "'m"]
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)

# function for appending stem words
def get_stem_words(words,ignore_words):
    stem_words=[]
    for word in words:
        if word not in ignore_words:
            w=stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        pattern_word=nltk.word_tokenize(pattern)
        words.extend(pattern_word)
        word_tags_list.append((pattern_word,intent["tag"]))

    if intent["tag"] not in classes:
        classes.append(intent["tag"])
        stem_words=get_stem_words(words,ignore_words)

print(stem_words)
print(classes)
print(word_tags_list[0])

def create_bot_corpus(stem_words,classes):
    stem_words=sorted(list(set(stem_words)))
    classes=sorted(list(set(classes)))
    pickle.dump(stem_words,open("words.pkl","wb"))
    pickle.dump(classes,open("classes.pkl","wb"))
    return stem_words,classes

stem_words,classes=create_bot_corpus(stem_words,classes)
print(stem_words)
print(classes)
    
training_data=[]
number_of_tags=len(classes)
labels=[0]*number_of_tags

for word_tags in word_tags_list:
    bag_of_words=[]
    pattern_words=word_tags[0]
    for word in pattern_words:
        index=pattern_words.index(word)
        word=stemmer.stem(word.lower())
        pattern_words[index]=word

    for word in stem_words:
        if word in pattern_words:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    
    print(bag_of_words)

    labels_encoding=list(labels)
    tag=word_tags[1]
    tag_index=classes.index(tag)
    labels_encoding[tag_index]=1
    training_data.append([bag_of_words,labels_encoding])

print(training_data[0])

def preproccessed_trained_data(training_data):
    training_data=np.array(training_data,dtype=object)
    trainx=list(training_data[:,0])
    trainy=list(training_data[:,1])
    print(trainx[0])
    print(trainy[0])

    return trainx,trainy

trainx,trainy=preproccessed_trained_data(training_data)
    




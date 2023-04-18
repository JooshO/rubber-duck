data_dict = {}
pos_map = {"JJ" : "a", "NN" : "n"}
NAME_KEY = "name"
LANG_KEY = "lang"

import random
import nltk
from os.path import exists
# nltk.chat.eliza.eliza_chat()

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle 

from collections import Counter
import csv

import re

trainingWords = set()
tagCounts = {}
training_dict = {}
prob_dict = {}
tag_prob = {}
stop_words = set(stopwords.words("english"))

'''
This Rubber Duck Bot is a Naive Bayesian Classifier using a little bit of natural language handling to increase accuracy by 
reducing variety of words.
'''

def train():
    lemmatizer = WordNetLemmatizer()
    tag_counts = Counter()
    total = 0
    count_dict = {}
    
    # Issue: the more tags we have, the less likely a given tag is to be "true" overall
    with open('initial_data.csv', mode ='r')as file: 
        lines = file.readlines()
        for line in lines:
            total += 1
            tag_str, data = line.split(",", 1) # split only once 
            tags = [tag_str]
            # tags = tag_str.replace('<', ' ').split(">")
            # tags = [tag.replace(' ', '') for tag in tags if tag != " " and tag != ""]

            for tag in tags:
                tag_counts[tag] += 1              # count number of instances of this tag
            
            if "." in data:
                data, _ = data.split(".", 1)
                
            words = word_tokenize(data.lower())
            words = [word for word in words if word not in stop_words]
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words] 
            for word in set(lemmatized_words):
                trainingWords.add(word)
                # init counters
                for tag in tags:
                    if tag not in count_dict:
                        count_dict[tag] = Counter()
                
                for tag in tags:
                    count_dict[tag][word] += 1

    for tag, counter in count_dict.items():
        training_dict[tag] = {}
        tag_prob[tag] = tag_counts[tag] / total
        for item, count in counter.items():
            training_dict[tag][item] = (count + 1) / (tag_counts[tag] + 2)

    with open('tagprob.pkl', 'wb') as f:
        pickle.dump(tag_prob, f)

    with open('training_dict.pkl', 'wb') as f:
        pickle.dump(training_dict, f)
    

def process_message(input):
    prod = {}

    for tag, prob in tag_prob.items():
        for word in input:
            if tag not in prod.keys():
                prod[tag] = 1
            
            if word in training_dict[tag].keys():
                prod[tag] *= training_dict[tag][word]
            else:
                prod[tag] *= 0.001

        if tag in prod.keys():
            prod[tag] *= prob

    total = sum(prod.values())
    probabilities = [(prod[tag] / total, tag) for tag in prod.keys()]
    if len(probabilities) == 0:
        prinf("None found")
        return
    probability, selected = max(probabilities, key=lambda item:item[0])
    # print(f'{selected} is the tag for the message, probaility is {probability}')
    return selected

    



'''
Currently attempts to classify user input
Consider: https://github.com/openai/gpt-2, https://medium.com/analytics-vidhya/guide-to-openais-gpt-2-and-how-to-use-it-in-python-72d37d7dd64c 
Consider: https://www.nltk.org/
'''
def main():
    lemmatizer = WordNetLemmatizer()
    # data_dict[NAME_KEY] = input("Hello! What is your name?\n")
    # data_dict[LANG_KEY] = input(f"Hello {data_dict[NAME_KEY]}, what language or framework are you working in today?\n")
    global tag_prob
    global training_dict

    reviewing = False
    
    if exists('tagprob.pkl') and exists('training_dict.pkl'):
        with open('tagprob.pkl', 'rb') as f:
            tag_prob = pickle.load(f)

        with open('training_dict.pkl', 'rb') as f:
            training_dict = pickle.load(f)
    else:
        train()

    print("> Alright, I am ready to listen!")
    finished = False
    while not finished:
        user_in = input().lower()

        if (random.random() < 0.05):
            print("Quack!!")
            continue

        words = word_tokenize(user_in)
        words = [word for word in words if word not in stop_words]
        tagged_words = nltk.pos_tag(words)
        # lemmatized_words = [lemmatizer.lemmatize(word, pos=pos_map[part]) for (word, part) in tagged_words]
        lemmatized_words = [lemmatizer.lemmatize(word) for (word, part) in tagged_words] 
        
        output = process_message(lemmatized_words)

        if reviewing:
            if output == "end_review":
                reviewing = False
                # todo: different tags for solved and not solved.
                print("> I hope I was able to assist in solving your problem!")
                print("> If not, you can try reaching out to your TA or professor, or try visiting the CCLC in Rekhi hall!")
            else:
                print("> Quack quack")
            continue

        if "quit" in words or "exit" in lemmatized_words:
            finished = True
            break

        if output == "code_review":
            print("> Feel free to start typing your code in! \n> Remember that copying and pasting will be less effective for you than retyping the code.")
            print("> I'll stay out of the way as you type, until you let me know that you are done!")
            reviewing = True

        if output == "loop":
            print("> It sounds like you are having an issue with a loop.")
            print("> It's always a good idea to take a closer loop at your loop conditions and what can cause you to break out of a loop.")
            print("> So try looking at breaks and continues, and check where you change your loop condition.")
            print("> Let me know if you would like to go over code!")



    print("> Good bye!")

if __name__ == "__main__":
    main()
import random
import nltk
from os.path import exists

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import pickle

from collections import Counter
import csv

'''
This Rubber Duck Bot is a Naive Bayesian Classifier using a little bit of
natural language processing from NLTK to increase accuracy by
reducing variety of words.
Worked on by Team HAL-9000 for CS4811 S23
'''
training_words = set()                          # set of words we read
training_dict = {}                              # dictionary we use to store training data
tag_prob = {}                                   # dictionary for individual tag probabilities
stop_words = set(stopwords.words("english"))    # stop words for filtering
lemmatizer = WordNetLemmatizer()


# This tag conversion function borrowed verbatim from https://www.holisticseo.digital/python-seo/nltk/lemmatize
def nltk_pos_tagger(nltk_tag):
    '''
    Converts string tags from nltk part of speech tagging to wordnet tags for lemmatizing
    '''
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def process_input(input):
    '''
    Converts a string input to a list of lemmatized words.
    It tokenizes, removes stopwords, tags parts of speech, then lemmatizes
    '''
    words = word_tokenize(input)
    words = [word for word in words if word not in stop_words]
    tagged_words = nltk.pos_tag(words)
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), tagged_words)
    lemmatized_words = [lemmatizer.lemmatize(word, pos=part) if part is not None else lemmatizer.lemmatize(word) for (word, part) in wordnet_tagged]
    return lemmatized_words


def train():
    '''
    Reads through the training data and counts up / calculates probabilities
    for all the words in that data
    '''
    tag_counts = Counter()  # counter for individual tags
    total = 0               # total inputs read in
    count_dict = {}         # counts of words by tag

    # Issue: the more tags we have, the less likely a given tag is to be "true" overall
    with open('initial_data.csv', mode='r') as file:
        lines = file.readlines()
        for line in lines:
            total += 1
            tag, data = line.split(",", 1)  # split only once - we don't want to lose things after commas

            tag_counts[tag] += 1            # count number of instances of this tag

            # Only pay attention to the first sentance. The second one in training data does little for us usually
            if "." in data:
                data, _ = data.split(".", 1)

            # make our input usable
            lemmatized_words = process_input(data.lower())

            # for each word
            for word in set(lemmatized_words):
                # add it to our training set
                training_words.add(word)

                # init counter if needed
                if tag not in count_dict:
                    count_dict[tag] = Counter()

                count_dict[tag][word] += 1

    # pre-calculate proabilities for each word-tag
    for tag, counter in count_dict.items():
        training_dict[tag] = {}
        tag_prob[tag] = tag_counts[tag] / total
        for item, count in counter.items():
            training_dict[tag][item] = (count + 1) / (tag_counts[tag] + 2)

    # output our data into pickle files
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
    probability, selected = max(probabilities, key=lambda item: item[0])
    # print(f'{selected} is the tag for the message, probaility is {probability}')
    return selected


def main():
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
        user_in = input()

        if (random.random() < 0.05):
            print("Quack!!")
            continue

        lemmatized_words = process_input(user_in.lower())
        output = process_message(lemmatized_words)

        if reviewing:
            if output == "end_review":
                reviewing = False
                # todo: different tags for solved and not solved.
                print("> I hope I was able to assist in solving your problem!")
                print("> If not, you can try reaching out to your TA or professor, or try visiting the CCLC in Rekhi hall!")
            else:
                print("> Quack quack")
        elif ("quit" in lemmatized_words or "exit" in lemmatized_words) and len(lemmatized_words) == 1:
            finished = True
            break
        elif output == "code_review":
            print("> Feel free to start typing your code in! \n> Remember that copying and pasting will be less effective for you than retyping the code.")
            print("> I'll stay out of the way as you type, until you let me know that you are done!")
            reviewing = True
        elif output == "loop":
            print("> It sounds like you are having an issue with a loop.")
            print("> It's always a good idea to take a closer loop at your loop conditions and what can cause you to break out of a loop.")
            print("> So try looking at breaks and continues, and check where you change your loop condition.")
            print("> Let me know if you would like to go over code!")
        elif output == "hanging":
            print("> It sounds like your code is getting stuck somewhere or hanging.")
            print("> That is often caused by an infinite loop or waiting for input in some way.")
            print("> If you are working on a concurrent application it could be deadlock, try looking at where you wait/signal or unlock/lock.")
            print("> Let me know if you would like to go over code!")
        elif output == "crash":
            print("> It sounds like your project is crashing.")
            print("> If you have an error message, it is always a good idea to look it up and check what line number it references to see if the issue is a quick fix.")
            print("> If you are working in C/C++, consider running your code through Valgrind or another debugging tool to get more information.")
            print("> Let me know if you would like to go over code!")
        elif output == "stuck":
            print("> It sounds like you are stuck with your project and might be frustrated.")
            print("> If you are allowed to talk to other people, an open-hands discussion with your friends might help kickstart some ideas.")
            print("> It might also be a good idea to take a break. Try going for a walk or talking to friends to let your brain relax and come back with fresh eyes.")
            print("> Let me know if you would like to go over code!")
        elif output == "databases":
            print("> It sounds like your problem involves database.")
            print("> If your issue is with a query, I always recommend breaking it up into smaller pieces and making sure everything works as expected.")
            print("> There are lot of great resources online to help with syntax as well, I personally recommend W3Schools.")
            print("> Let me know if you would like to go over code!")
        else:
            print("> Unfortuanately, I don't have any good advice for that kind of problem (or I misunderstood).")
            print("> I can still take a look at your code happily, or you can try to rephrase the question.")
            print("> Let me know if you would like to go over code!")

    print("> Good bye!")


if __name__ == "__main__":
    main()

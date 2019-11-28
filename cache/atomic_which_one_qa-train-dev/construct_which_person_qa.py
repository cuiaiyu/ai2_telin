import os
import csv
import json
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import pandas as pd
import numpy as np
import random
import re
import nltk
import jsonlines

ATOMIC_QUESTIONS = {
    'oEffect': "What effects does the event have on others?",
    'oReact':  "How do others feel after the event?",
    'oWant':   "What would others want to do next after the event?",
    'xAttr':   "How would you describe PersonX?",
    'xEffect': "What effects does the event have on X?",
    'xIntent': "Why does X cause this event?",
    'xNeed':   "What does personX need to do before the event?",
    'xReact':  "How does PersonX feel after the event?",
    'xWant':   "What does PersonX want to do next after the event? "
}

# ''' uncomment this line to not run main
# helper functions
def remove(text):
    remove_chars = '[0-9’!"#$%&\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    return re.sub(remove_chars, '', text)

# using nltk.wordnet to find antonyms and similarity
def find_antonyms(word):
    anto = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            if lemma.antonyms():
                anto.append(lemma.antonyms()[0].name())
    return anto

def similarity(first_word, second_word):
    if len(wordnet.synsets(first_word)) > 0 and len(wordnet.synsets(second_word)) > 0:
        first_word = wordnet.synsets(first_word)[0]
        second_word = wordnet.synsets(second_word)[0]
        simi = first_word.wup_similarity(second_word)
        return simi
    else:
        return 1 # error value to re-generate random
    
def get_random_attr(n, attr_name):
    events = dataset.keys()
    attributes = []
    while(len(attributes) < n):
        random_events = random.sample(events, n)
        # print("random_events", random_events)
        for e in random_events:
            attributes = attributes + dataset[e][attr_name] 
    return random.sample(attributes, n)

### ATOMIC SELF QA CONSTRUCTOR
random.seed(666)

# Step 1: get all data
dir_path = os.path.dirname(os.path.realpath(__file__))
# not working in interactive python... No __file__!
# should be:
# dir_path = os.getcwd()
file_name = "v4_atomic_all.csv"
df = pd.read_csv(os.path.join(dir_path, file_name))
data = df.values[:, :-2] # don't need "prefix" and "split" col
cols = df.columns.values
attributes = cols[1:-2] # "event", "prefix", "split" are not attr
# print(data.shape)
# print(cols.shape)
# print(attributes)

# Step 2: put all these data into a dictionary of dictionaries, 

# Step 2a: clean up, from csv to dict of dicts
# Step 2b: remove triangles? (PersonX, Y and Z all appeared in one sentence) 
# to avoid ambiguity on what person o-attributes are about
def contains_persons(sentence):
    # return a triplet of bool values
    # contains PersonX, PersonY, PersonZ
    pattern1 = re.compile("personx", re.IGNORECASE)
    pattern2 = re.compile("person x", re.IGNORECASE)
    pattern3 = re.compile("persony", re.IGNORECASE)
    pattern4 = re.compile("person y", re.IGNORECASE)
    pattern5 = re.compile("personz", re.IGNORECASE)
    pattern6 = re.compile("person z", re.IGNORECASE)
    return (
        re.search(pattern1, sentence) or re.search(pattern2, sentence), 
        re.search(pattern3, sentence) or re.search(pattern4, sentence),
        re.search(pattern5, sentence) or re.search(pattern6, sentence)
    )

dataset = {} # {"event": {"attr": [] }}
for i in range(data.shape[0]):
# for i in range(21000):
    event = data[i, 0]
    if "___" in event or all(contains_persons(event)):
        continue
    if event not in dataset:
        dataset[event] = {attr: [] for attr in attributes}
        # dataset[event] = {}
    for j in range(attributes.shape[0]):
        attr = cols[j+1]
        cell_answers = data[i, j+1][1:-1].lower().split(", ") # remove brakets, lowercase
        for answer in cell_answers:
            if answer != "" and answer != " " and 'none' not in answer:
                answer = remove(answer)
                dataset[event][attr].append(answer)

# for key in dataset.keys():
#     print(key)
print(len(dataset.keys()))
# print(data.shape)
# Step 2c: pickle load and dump
## TODO (not saving too much time, maybe later)

# Step 5: construct the which_person qa dataset

# Question types for which_peron qa
# Group One (vise versa)
# xReact: How does PersonX feel after the event? (x 1 correct)
# oReact: How do others feel after the event? (x 2 incorrect)
# Group Two (vise versa)
# xWant: What does PersonX want to do next after the event? x 1
# oWant: What would others want to do next after the event? x 2
# Group Three (vise versa)
# xEffect: What effects does the event have on X? x 1
# oEffect: What effects does the event have on others? x 2

# skip events which does not have at least one answer for each x-o attr pair

# Addendum: for the two incorrect answers:

# 1. if one event has only one answer for the opposite/incorrect attr: 
#      randomly select from same attr of another event 
# 2. if one event indeed has 2+ incorrect answers:
#      50% of the time still do a random selection

ATOMIC_WHICH_PERSON_QUESTIONS = {
    'oEffect': "What effects does the event have on others?",
    'oReact':  "How do others feel after the event?",
    'oWant':   "What would others want to do next after the event?",
    'xEffect': "What effects does the event have on X?",
    'xReact':  "How does PersonX feel after the event?",
    'xWant':   "What does PersonX want to do next after the event? "
}

random.seed(666)
temporal_attr_pairs = [
    ('oEffect', 'xEffect'),
    ('oReact',  'xReact'),
    ('oWant',   'xWant'),
    ('xEffect', 'oEffect'),
    ('xReact',  'oReact'),
    ('xWant',   'oWant')
]
QA_WHICH_PERSON_dataset = []
for event in dataset:
    for pair in temporal_attr_pairs:
        correct_attr = pair[0]
        wrong_attr = pair[1]
        if len(dataset[event][correct_attr]) == 0 \
            or len(dataset[event][wrong_attr]) == 0:
            continue
        if random.choice([0, 1]) == 0: # too much data (60k+ entries), randomly select half of it
            correct_ans = random.choice(dataset[event][correct_attr])
            first_wrong_ans = second_wrong_ans = ""
            if len(dataset[event][wrong_attr]) >= 2 and random.choice([0, 1]) == 0:
                first_wrong_ans, second_wrong_ans = random.sample(dataset[event][wrong_attr], 2)
            else:
                first_wrong_ans = random.choice(dataset[event][wrong_attr])
                second_wrong_ans = random.choice(get_random_attr(1, correct_attr) \
                                    + get_random_attr(1, wrong_attr))
            question = ATOMIC_WHICH_PERSON_QUESTIONS[correct_attr]
            print("event: {}\n\tquestion of type {}: {}\n\tcorrect answer: {}\n\tfirst wrong: {}\n\tsecond wrong: {}" \
                .format(event, correct_attr, question, correct_ans, first_wrong_ans, second_wrong_ans))
            answers = [correct_ans, first_wrong_ans, second_wrong_ans]
            order = [0, 1, 2]
            random.shuffle(order)
            which_is_correct = order.index(0) + 1
            answers = [answers[i] for i in order]
            QA_WHICH_PERSON_dataset.append((
                answers, 
                which_is_correct, 
                question, 
                event))
            print(answers, which_is_correct)

jsonl_writer = jsonlines.open('atomic_which_one_qa.jsonl', mode='w')
label_writer = open("atomic_which_one_qa-labels.lst", "w")
for entry in QA_WHICH_PERSON_dataset:
    json_obj = {
        "content": entry[3],
        "question": entry[2],
        "answerA": entry[0][0],
        "answerB": entry[0][1],
        "answerC": entry[0][2]
    }
    label = entry[1]
    jsonl_writer.write(json_obj)
    label_writer.write(str(label)+'\n')
jsonl_writer.close()
label_writer.close()
print("which person size:",len(QA_WHICH_PERSON_dataset))

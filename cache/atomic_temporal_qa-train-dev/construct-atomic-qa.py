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

''' uncomment this line to not run playground
# first_word = wordnet.synsets("Good.n.01")
# second_word = wordnet.synset("zebra.n.01")
first_word = wordnet.synsets("Good")[0]
second_word = wordnet.synsets("zebra")[0]
print('Similarity: ' + str(first_word.wup_similarity(second_word)))
# '''

# ''' uncomment this line to not run main
random.seed(666)

def remove(text):
    remove_chars = '[0-9’!"#$%&\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    return re.sub(remove_chars, '', text)

### ATOMIC SELF QA CONSTRUCTOR

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

# -------------------------------------------------------------------
# Step 3: construct the attr qa dataset

# Step 3a, construct the qa for xAttr
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
    
def get_random_attr(number):
    events = dataset.keys()
    attributes = []
    while(len(attributes) < number):
        random_events = random.sample(events, number)
        # print("random_events", random_events)
        for e in random_events:
            attributes = attributes + dataset[e]['xAttr'] 
    return random.sample(attributes, number)

QA_ATTR_dataset = {}
xAttr_q = "How would you describe PersonX?"
for event in dataset:
    count = 0 # count the number of times fetching a random attr from dataset
    stop_searching_thread = 100 # for some words, it is hard to find their not so similar words, probably caused by mis-spelling
    if len(dataset[event]['xAttr']) > 0:
        correct_ans = random.choice(dataset[event]['xAttr'])
        if len(correct_ans.split()) > 1: # don't bother try to find word similarity for phrases...
            continue
        anto = find_antonyms(correct_ans)
        first_wrong = second_wrong = ""
        
        if len(anto) >= 1: # antonym found, just need one random xAttr field
            first_wrong = anto[0]
            second_wrong = get_random_attr(1)[0]
            count += 1
            simi_with_correct = similarity(second_wrong, correct_ans)
            simi_with_first = similarity(second_wrong, first_wrong)
            while count < stop_searching_thread and \
                ((simi_with_correct and simi_with_correct > 0.4) \
                    or (simi_with_first and simi_with_first > 0.4)):
                second_wrong = get_random_attr(1)[0]
                count += 1
                simi_with_correct = similarity(second_wrong, correct_ans)
                simi_with_first = similarity(second_wrong, first_wrong)

        else: # no antonym found, need two random xAttr field
            first_wrong, second_wrong = get_random_attr(2)
            count += 2
            simi_1 = similarity(first_wrong, correct_ans)
            simi_2 = similarity(second_wrong, correct_ans)
            while count < stop_searching_thread and \
                ((simi_1 and simi_1 > 0.4) or ((simi_2 and simi_2 > 0.4))):
                first_wrong, second_wrong = get_random_attr(2)
                count += 2
                # print("correct_ans {} first_wrong {} second_wrong {}".format(correct_ans, first_wrong, second_wrong))
                simi_1 = similarity(first_wrong, correct_ans)
                simi_2 = similarity(second_wrong, correct_ans)
        
        if count < stop_searching_thread:
            print("event: {}\n\tcorrect answer: {}\n\tfirst wrong: {}\n\tsecond wrong: {}" \
                .format(event, correct_ans, first_wrong, second_wrong))
            answers = [correct_ans, first_wrong, second_wrong]
            order = [0, 1, 2]
            random.shuffle(order)
            which_is_correct = order.index(0) + 1
            answers = [answers[i] for i in order]
            QA_ATTR_dataset[event] = (answers, which_is_correct, xAttr_q)
            print(answers, which_is_correct)

jsonl_writer = jsonlines.open('atomic_attr_qa.jsonl', mode='w')
label_writer = open("atomic_attr_qa-labels.lst", "w")
for event in QA_ATTR_dataset:
    json_obj = {
        "content": event,
        "question": QA_ATTR_dataset[event][2],
        "answerA": QA_ATTR_dataset[event][0][0],
        "answerB": QA_ATTR_dataset[event][0][1],
        "answerC": QA_ATTR_dataset[event][0][2]
    }
    label = QA_ATTR_dataset[event][1]
    jsonl_writer.write(json_obj)
    label_writer.write(str(label)+'\n')

# ---------------------------------------------------------

# Step 4: construct the temporal qa dataset
# Question types for temporal qa
# Only group (vise versa)
# xNeed: What does personX need to do before the event?
# xWant: What does PersonX want to do next after the event?
# oWant: What would others want to do next after the event?
# should we add oWant? 
# pro: socialIQA wants-type questions account for 29% of all questions
#      so add more Want-type answers would help
# con: not purely temporal any more:
#      it would add person difference into temporal qa
#      (I think it can be arguably beneficial)
# the discussion result is yes.

random.seed(666)
ATOMIC_TEMPORAL_QUESTION = {
    'xNeed':   "What does personX need to do before the event?",
    'xWant':   "What does PersonX want to do next after the event? ",
    'oWant':   "What would others want to do next after the event?",
}

QA_TEMPO_dataset = {}
for event in dataset:
    if len(dataset[event]['xNeed']) == 0 \
        or len(dataset[event]['xWant']) == 0 \
        or len(dataset[event]['oWant']) == 0:
        continue
    # avoid using set
    # temporal_attrs = set(['xNeed', 'xWant', 'oWant'])
    temporal_attrs = ['xNeed', 'xWant', 'oWant']
    correct_ans = first_wrong_ans = second_wrong_ans = ""
    for i in range(len(temporal_attrs)):
        attr = temporal_attrs[i]
        correct_ans = random.choice(dataset[event][attr])
        # avoid using set
        # other_two_attrs = temporal_attrs - set(attr)
        # first_wrong_attr = other_two_attrs.pop()
        # second_wrong_attr = other_two_attrs.pop()
        first_wrong_attr = temporal_attrs[(i + 1) % 3]
        second_wrong_attr = temporal_attrs[(i + 2) % 3]
        first_wrong_ans = random.choice(dataset[event][first_wrong_attr])
        second_wrong_ans = random.choice(dataset[event][second_wrong_attr])

        print("event: {}\n\tquestion: {}: {}\n\tcorrect answer: {}\n\tfirst wrong from attr {}: {}\n\tsecond wrong from attr {}: {}" \
            .format(event, attr, ATOMIC_TEMPORAL_QUESTION[attr], correct_ans, first_wrong_attr, first_wrong_ans, second_wrong_attr, second_wrong_ans))
        answers = [correct_ans, first_wrong_ans, second_wrong_ans]
        order = [0, 1, 2]
        random.shuffle(order)
        which_is_correct = order.index(0) + 1
        answers = [answers[i] for i in order]
        question = ATOMIC_TEMPORAL_QUESTION[attr]
        QA_TEMPO_dataset[event] = (answers, which_is_correct, question)
        print(answers, which_is_correct)

jsonl_writer = jsonlines.open('atomic_temporal_qa.jsonl', mode='w')
label_writer = open("atomic_temporal_qa-labels.lst", "w")
for event in QA_TEMPO_dataset:
    json_obj = {
        "content": event,
        "question": QA_TEMPO_dataset[event][2],
        "answerA": QA_TEMPO_dataset[event][0][0],
        "answerB": QA_TEMPO_dataset[event][0][1],
        "answerC": QA_TEMPO_dataset[event][0][2]
    }
    label = QA_TEMPO_dataset[event][1]
    jsonl_writer.write(json_obj)
    label_writer.write(str(label)+'\n')

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

ATOMIC_QUESTIONS = {
    'oEffect': "What effects does the event have on others?",
    'oReact':  "How do others feel after the event?",
    'oWant':   "What would others want to do next after the event?",
    'xEffect': "What effects does the event have on X?",
    'xReact':  "How does PersonX feel after the event?",
    'xWant':   "What does PersonX want to do next after the event? "
}

random.seed(666)
# TODO
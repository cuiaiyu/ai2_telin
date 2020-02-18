import jsonlines
import re
# resulted jsonl looks like (notice that 
# either all beforeABC or all afterABC are empty
# which will be ignored later in the dataset/dataloader):
# {  
#    content: "",
#    beforeA: "",
#    beforeB:"",
#    beforeC: "",
#    afterA: "",
#    afterB: "",
#    afterC: ""
# }

def get_name_at_word_index(sentence, word_index):
    # precondition: the word @ word_index is a name
    words = sentence.split()
    name = ""
    if len(words) <= word_index:
        return ""
    elif "'" in words[word_index] and len(words) > (word_index + 1):
        name = words[word_index] + ' ' + words[word_index + 1] # e.g. Jesus' mother
    elif words[word_index] in ["the", "his", "her", "their", "this", "that"]:
        name = words[word_index] + ' ' + words[word_index + 1] # e.g. the kid, her son
    else:
        name = words[word_index]
    return re.sub(r'[^\w\s]','',name)

# (the format of different socialIQA question types)
# name should be right after the first element (question head)
QUESTION_FORMATS = {
    0: ["How would", "feel"],
    1: ["How would you describe", ""],
    2: ["What will", "want to do next"],
    3: ["What will happen to", ""],
    4: ["What will", "do after"],
    5: ["What does", "need to do before this"],
    6: ["Why did", "do this"]
}
QUESTION_TYPES = list(range(7)) # see indexing in file: question-types-examples

# length of first element = 2 ==> name at index 2  # [0, 2, 4, 5, 6]
q_type_list_names_at_index2 = [q_type for q_type in QUESTION_FORMATS \
    if len(QUESTION_FORMATS[q_type][0].split()) == 2 ]
print(q_type_list_names_at_index2)  # [0, 2, 4, 5, 6]

# length of first element = 4 ==> name at index 4  # [1, 3]
q_type_list_names_at_index4 = [q_type for q_type in QUESTION_FORMATS \
    if len(QUESTION_FORMATS[q_type][0].split()) == 4 ]
print(q_type_list_names_at_index4)  # [1, 3]

def preprocess(preprocess_train=True):    
    # fp = open("social_before_after-train-dev/train.jsonl", 'w+') if preprocess_train \
    #     else open("social_before_after-train-dev/dev.jsonl", 'w+')
    # jsonlWriter = jsonlines.Writer(fp)
    # labelWriter = open("social_before_after-train-dev/train-labels.lst", "w+") if preprocess_train \
    #     else open("social_before_after-train-dev/dev-labels.lst", "w+")
    fp = open("social_before_after_still_qa-train-dev/train.jsonl", 'w+') if preprocess_train \
        else open("social_before_after_still_qa-train-dev/dev.jsonl", 'w+')
    jsonlWriter = jsonlines.Writer(fp)
    labelWriter = open("social_before_after_still_qa-train-dev/train-labels.lst", "w+") if preprocess_train \
        else open("social_before_after_still_qa-train-dev/dev-labels.lst", "w+")
    labelReader = open("../socialiqa-train-dev/socialiqa-train-dev/train-labels.lst", "r") if preprocess_train \
        else open("../socialiqa-train-dev/socialiqa-train-dev/dev-labels.lst", "r")
    jsonlReader = jsonlines.open('../socialiqa-train-dev/socialiqa-train-dev/train.jsonl') if preprocess_train \
        else jsonlines.open('../socialiqa-train-dev/socialiqa-train-dev/dev.jsonl')

    count_irregular = 0
    count_regular = 0

    for obj in jsonlReader:
        this_question = obj["question"]
        this_label = int(labelReader.readline())
        type_matches = []
        for q_type in QUESTION_TYPES:
            if QUESTION_FORMATS[q_type][0] in this_question and \
                QUESTION_FORMATS[q_type][1] in this_question:
                type_matches.append(True)
            else:
                type_matches.append(False)
        
        # handling irregular questions
        if sum(type_matches) != 1:
            count_irregular += 1
            # print(this_question, type_matches)
            # if "before": put the answer at front
            # else: put the answer after content

        # handling regular questions
        else:
            count_regular += 1
            # 1. get type
            # which_type = type_matches.index(True)
            # print(obj)
            # print(this_question)

            # # 2. get name
            # name = ""
            # if which_type in q_type_list_names_at_index2: # [0, 2, 4, 5, 6]
            #     name = get_name_at_word_index(this_question, 2)
            # elif which_type in q_type_list_names_at_index4: # [1, 3]
            #     name = get_name_at_word_index(this_question, 4)
            # assert(len(name) > 0)
            # name = name[0].upper() + name[1:]

            # # 3. generate new jsonl according to question type:
            # new_obj = {
            #     "context": obj["context"],
            #     "beforeA": "",
            #     "beforeB":"",
            #     "beforeC": "",
            #     "afterA": "",
            #     "afterB": "",
            #     "afterC": ""
            # }
            # ansA = obj["answerA"][0].lower() + obj["answerA"][1:]
            # ansB = obj["answerB"][0].lower() + obj["answerB"][1:]
            # ansC = obj["answerC"][0].lower() + obj["answerC"][1:]

            # if which_type == 0:
            #     # Q: How would someone feel [restriction e.g. as a result, afterwards]?
            #     new_obj["afterA"] = name + ' would feel ' + ansA
            #     new_obj["afterB"] = name + ' would feel ' + ansB
            #     new_obj["afterC"] = name + ' would feel ' + ansC
            # elif which_type == 1:
            #     # Q: How would you describe someone?
            #     new_obj["afterA"] = name + ' is ' + ansA
            #     new_obj["afterB"] = name + ' is ' + ansB
            #     new_obj["afterC"] = name + ' is ' + ansC
            # elif which_type == 2:
            #     # Q: How will someone want to do next?
            #     new_obj["afterA"] = name + ' wants to ' + ansA
            #     new_obj["afterB"] = name + ' wants to ' + ansB
            #     new_obj["afterC"] = name + ' wants to ' + ansC
            # elif which_type == 3:
            #     # Q: What will happen to someone?
            #     new_obj["afterA"] = name + ' may ' + ansA
            #     new_obj["afterB"] = name + ' may ' + ansB
            #     new_obj["afterC"] = name + ' may ' + ansC              
            # elif which_type == 4:
            #     # Q: What will someone do after?
            #     new_obj["afterA"] = 'Next, ' + name + ' will ' + ansA
            #     new_obj["afterB"] = 'Next, ' + name + ' will ' + ansB
            #     new_obj["afterC"] = 'Next, ' + name + ' will ' + ansC
            # elif which_type == 5:
            #     # Q: What does someone need to do before this?
            #     new_obj["beforeA"] = name + ' ' + ansA # Or 'First, ' + name + ' ' + ansA?
            #     new_obj["beforeB"] = name + ' ' + ansB
            #     new_obj["beforeC"] = name + ' ' + ansC
            # elif which_type == 6:
            #     # Q: Why did someone do this?
            #     # formats of answers for this question varied!! TODO
            #     new_obj["afterA"] = name + ' did this to ' + ansA
            #     new_obj["afterB"] = name + ' did this to ' + ansB
            #     new_obj["afterC"] = name + ' did this to ' + ansC
            # else:
            #     exit(-1)

            # if count_regular % ( 5000 if preprocess_train else 1000) == 0:
            #     print("----------------------------------")
            #     print(obj)
            #     print()
            #     print(new_obj)

            # jsonlWriter.write(new_obj)
            jsonlWriter.write(obj)
            labelWriter.write(str(this_label)+"\n")
    print("----------------------------------")

    print(count_irregular) # 1180  in train;   49 in dev
    print(count_regular)   # 32230 in train; 1905 in dev

    jsonlReader.close()
    labelReader.close()
    jsonlWriter.close()
    labelWriter.close()

if __name__ == "__main__":
    preprocess(preprocess_train=True)
    preprocess(preprocess_train=False)
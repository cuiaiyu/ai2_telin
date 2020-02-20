import jsonlines
import re

"""
resulted jsonl format:
    {  
        content: "",
        beforeA: "",
        beforeB:"",
        beforeC: "",
        afterA: "",
        afterB: "",
        afterC: ""
    }
"""

# (the format of different socialIQA question types)
# name should be right after the first element (question head)
QUESTION_FORMATS = {
    "feel": {
        "find_name_after": ["How would", "How will", "What will", "What would", "How did", "How does", "How is",  "How do you think"],
        "must_include": ["feel"]
    },
    "describe": {
        "find_name_after": ["How would you describe", "What will people think about", "What kind of person is"],
        "must_include": [""]
    },
    "next_action": {
        "find_name_after": ["What will", "What does", "What should", "What would", "What is", "What may", "What did"],
        "must_include": ["want to", "do after", "do next", "going to do", "need to do next", "next", "need to do now"]
    },
    "happen_to": {
        "find_name_after": ["What will happen to", "What is happening to"],
        "must_include": [""]
    },
    "action_before": {
        "find_name_after": ["What does", "What did", "What do", "What will", "What would", "What should"], 
        "must_include": ["before"]
    },
    "reason": {
        "find_name_after": ["Why did", "Why would", "Why does", "Why is"],
        "must_include": ["do this", ""]
    }
}

# question types with name starting at the third word
# ['feel', 'next_action', 'action_before', 'reason']
ques_types_with_name_at_2 = [q_type for q_type in QUESTION_FORMATS \
    if len(QUESTION_FORMATS[q_type]["find_name_after"][0].split()) == 2 ]

# question types with name starting at the fifth word
# ['describe', 'happen_to']
ques_types_with_name_at_4 = [q_type for q_type in QUESTION_FORMATS \
    if len(QUESTION_FORMATS[q_type]["find_name_after"][0].split()) == 4 ]

def is_of_type(q_str, q_type):
    """
        determine if q_str matches q_type or not
        :param: q_str: str, question string
        :param: q_type: dict, 
            e.g. {"find_name_after": ["How would", "How will"],"must_include": ["feel"]}
    """
    if any( [ (phrase in q_str) for phrase in q_type["find_name_after"] ]):
        if any( [ (phrase in q_str) for phrase in q_type["must_include"] ]):
            return True
    return False


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



def preprocess(preprocess_train=True):    
    fp = open("social_before_after_v2-train-dev/train.jsonl", 'w+') if preprocess_train \
        else open("social_before_after_v2-train-dev/dev.jsonl", 'w+')
    jsonlWriter = jsonlines.Writer(fp)
    labelWriter = open("social_before_after_v2-train-dev/train-labels.lst", "w+") if preprocess_train \
        else open("social_before_after_v2-train-dev/dev-labels.lst", "w+")
    
    labelReader = open("../socialiqa-train-dev/socialiqa-train-dev/train-labels.lst", "r") if preprocess_train \
        else open("../socialiqa-train-dev/socialiqa-train-dev/dev-labels.lst", "r")
    jsonlReader = jsonlines.open('../socialiqa-train-dev/socialiqa-train-dev/train.jsonl') if preprocess_train \
        else jsonlines.open('../socialiqa-train-dev/socialiqa-train-dev/dev.jsonl')

    count_irregular = 0
    total_count = 0

    for obj in jsonlReader:
        this_question = obj["question"]
        this_label = int(labelReader.readline())
        total_count += 1

        # 1. get type
        # match question format with types
        matched_types = []
        for which_type in QUESTION_FORMATS:
            q_type = QUESTION_FORMATS[which_type]
            if is_of_type(this_question, q_type):
                matched_types.append(which_type)
        
        # handling non-standard questions
        if len(matched_types) != 1:
            # matching more than one type of questions: 
            # happen_to > action+before > feel > other types
            if  len(matched_types) > 1:
                if "happen_to" in matched_types:
                    matched_types = ["happen_to"]
                elif "action_before" in matched_types:
                    matched_types = ["action_before"]
                elif "feel" in matched_types:
                    matched_types = ["feel"]
                else:
                    matched_types = [matched_types[0]]
            # no matching question types --> irregular questions, which_type = None
            elif len(matched_types) == 0:
                count_irregular += 1
                # print(this_question)
                matched_types = [None]
        # for all
        which_type = matched_types[0]

        # 2. get name
        if which_type:
            idx = len(QUESTION_FORMATS[which_type]["find_name_after"][0].split())
            name = get_name_at_word_index(this_question, idx)
            assert(len(name) > 0)
            name = name[0].upper() + name[1:]
        else: # for the most irregular cases
            name = ""

        # 3. generate new jsonl according to question type:
        new_obj = {
            "context": obj["context"],
            "beforeA": "",
            "beforeB":"",
            "beforeC": "",
            "afterA": "",
            "afterB": "",
            "afterC": ""
        }
        ansA = obj["answerA"][0].lower() + obj["answerA"][1:]
        ansB = obj["answerB"][0].lower() + obj["answerB"][1:]
        ansC = obj["answerC"][0].lower() + obj["answerC"][1:]

        if which_type == "feel":
            # Q: How would someone feel?
            new_obj["afterA"] = name + ' would feel ' + ansA
            new_obj["afterB"] = name + ' would feel ' + ansB
            new_obj["afterC"] = name + ' would feel ' + ansC
        elif which_type == "describe":
            # Q: How would you describe someone?
            new_obj["afterA"] = name + ' is ' + ansA
            new_obj["afterB"] = name + ' is ' + ansB
            new_obj["afterC"] = name + ' is ' + ansC
        elif which_type == "next_action":
            # Q: How will someone want to do next?
            new_obj["afterA"] = name + ' wants to ' + ansA
            new_obj["afterB"] = name + ' wants to ' + ansB
            new_obj["afterC"] = name + ' wants to ' + ansC
        elif which_type == "happen_to":
            # Q: What will happen to someone?
            new_obj["afterA"] = name + ' may ' + ansA
            new_obj["afterB"] = name + ' may ' + ansB
            new_obj["afterC"] = name + ' may ' + ansC              
        elif which_type == "action_before":
            # Q: What does someone need to do before this?
            new_obj["beforeA"] = name + ' ' + ansA
            new_obj["beforeB"] = name + ' ' + ansB
            new_obj["beforeC"] = name + ' ' + ansC
        elif which_type == "reason":
            # Q: Why did someone do this?
            new_obj["afterA"] = name + ' did this to ' + ansA
            new_obj["afterB"] = name + ' did this to ' + ansB
            new_obj["afterC"] = name + ' did this to ' + ansC
        else: # the irregular questions
            new_obj["afterA"] = this_question[:-1] + " is " + ansA
            new_obj["afterB"] = this_question[:-1] + " is " + ansB
            new_obj["afterC"] = this_question[:-1] + " is " + ansC

        if total_count % ( 5000 if preprocess_train else 1000) == 0 or which_type is None:
            print("----------------------------------")
            print(obj)
            print(which_type)
            print(new_obj)

        jsonlWriter.write(new_obj)
        labelWriter.write(str(this_label)+"\n")
    print("----------------------------------")

    print(count_irregular) #   374 in train;   28 in dev
    print(total_count)     # 33410 in train; 1954 in dev

    jsonlReader.close()
    labelReader.close()
    jsonlWriter.close()
    labelWriter.close()

if __name__ == "__main__":
    preprocess(preprocess_train=True)
    preprocess(preprocess_train=False)
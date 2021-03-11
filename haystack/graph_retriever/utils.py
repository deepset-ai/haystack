import re
import numpy as np
import string


# def _fuzzymatch(answer1, answer2):
#     ed = editdistance.eval(answer1, answer2)
#     maxlen = max([len(answer1),len(answer2)])
#     fuzzy = (maxlen-ed) / maxlen
#     return fuzzy

def _qa_f1(answer1, answer2):
    # remove a, an, and
    answer1 = re.sub(r'\b(a|an|the)\b', ' ', answer1)
    answer2 = re.sub(r'\b(a|an|the)\b', ' ', answer2)
    #remove punctuation
    answer1 = ''.join(ch for ch in answer1 if ch not in set(string.punctuation))
    answer2 = ''.join(ch for ch in answer2 if ch not in set(string.punctuation))

    ans1_tokens = answer1.split()
    ans2_tokens = answer2.split()
    n_overlap = len([x for x in ans1_tokens if x in ans2_tokens])
    if n_overlap == 0:
        return 0.0
    precision = n_overlap / len(ans1_tokens)
    recall = n_overlap / len(ans2_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compare_answers_fuzzy(answer1, answer2, question_type, extractive=True, list_f1_treshold = 0.7):
    # question type: Short, Multi Fact, Numeric, Boolean, Open-Ended
    result = {}



    if question_type == "Short" or question_type == "Open-Ended":
        answer1 = answer1.strip().lower()
        answer2 = answer2.strip().lower()
        result["em"] = int(answer1 == answer2)
        result["f1"] = _qa_f1(answer1, answer2)

    if question_type == "Multi Fact":
        answer1 = answer1.strip().lower()
        answer2 = answer2.strip().lower()
        if extractive:
            list1 = re.split('and |, |\n',answer1)
            list2 = re.split('and |, |\n',answer2)
            set1 = set([x.strip() for x in list1])
            set2 = set([x.strip() for x in list2])
        else:
            set1 = set([x.strip() for x in answer1.split("\n")])
            set2 = set([x.strip() for x in answer2.split("\n")])

        result["em"] = int(set1 == set2)

        bestmatch = np.zeros(len(set1))
        for i,e1 in enumerate(list(set1)):
            for e2 in list(set2):
                f1 = _qa_f1(e1,e2)
                if bestmatch[i] < f1:
                    bestmatch[i] = f1
        bestmatch = bestmatch > list_f1_treshold
        result["f1"] = sum(bestmatch)/len(set1.union(set2))

    if question_type == "Boolean":
        try:
            answer1 = bool(answer1)
            answer2 = bool(answer2)
        except Exception:
            result["em"] = 0
            result["f1"] = 0
        result["em"] = int(answer1 == answer2)
        result["f1"] = int(answer1 == answer2)

    if question_type == "Numeric":
        try:
            answer1 = int(answer1)
            answer2 = int(answer2)
            result["em"] = int(answer1 == answer2)
            result["f1"] = 1 - (abs(answer1 - answer2) / max([answer1, answer2]))
        except Exception:
            result["em"] = 0
            result["f1"] = 0

    return result


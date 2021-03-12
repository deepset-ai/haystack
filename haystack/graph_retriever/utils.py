import re
import numpy as np
import string
import pandas as pd
import ast
import logging

logger = logging.getLogger(__name__)

def eval_on_all_data(pipeline, top_k_graph, input_df):
    # iterating over each Question
    for index, row in input_df.iterrows():

        prediction, _ = pipeline.run(query=row['Question Text'], top_k_graph=top_k_graph)

        if not pd.isna(row["Short"]):
            question_type = "Short"
        elif not pd.isna(row["Multi Fact"]):
            question_type = "Multi Fact"
        elif not pd.isna(row["Numeric"]):
            question_type = "Numeric"
        elif not pd.isna(row["Boolean"]):
            question_type = "Boolean"
        elif not pd.isna(row["Open-Ended"]):
            question_type = "Open-Ended"
        else:
            raise NotImplementedError

        best_res = {"em":0, "f1":0}
        for i, p in enumerate(prediction["answers"]):
            if not pd.isna(row["feedback_labels"]):
                try:
                    ground_truths = ast.literal_eval(row["feedback_labels"])
                    if not isinstance(ground_truths, list):
                        ground_truths = []
                        logger.warning(f"Could not convert feedback into list: {row['feedback_labels']}")
                except Exception:
                    logger.warning(f"Could not convert feedback into list: {row['feedback_labels']}")
                    ground_truths = []
                ground_truths.append(row["Answer"])
                ground_truths = list(set(ground_truths))
            else:
                ground_truths = [row["Answer"]]
            res = compare_answers_fuzzy(ground_truths=ground_truths,
                                        prediction=p["answer"],
                                        question_type=question_type)
            if res["f1"] > best_res["f1"]:
                best_res = res

        input_df.loc[index, f"best_prediction"] = prediction["answers"][0]["answer"]
        input_df.loc[index, f'em_top{top_k_graph}'] = best_res["em"]
        input_df.loc[index, f'f1_top{top_k_graph}'] = best_res["f1"]
        if index % 10 == 0:
            print(f"Predicting the {index} item")
            input_df.to_csv("tempcache.csv",index=False)

    return input_df


def compare_answers_fuzzy(ground_truths, prediction, question_type, extractive=True, list_f1_treshold = 0.7):
    # question type: Short, Multi Fact, Numeric, Boolean, Open-Ended
    best_result = {"em": 0, "f1": 0}
    for gt in ground_truths:
        result = {"em": 0, "f1": 0}
        gt = str(gt)
        prediction = str(prediction)

        if question_type == "Short" or question_type == "Open-Ended":
            gt = gt.strip().lower()
            prediction = prediction.strip().lower()
            result["em"] = int(gt == prediction)
            result["f1"] = _qa_f1(gt, prediction)
        elif question_type == "Multi Fact":
            gt = gt.strip().lower()
            prediction = prediction.strip().lower()
            if extractive:
                list1 = re.split('and |, |\n', gt)
                list2 = re.split('and |, |\n', prediction)
                set1 = set([x.strip() for x in list1])
                set2 = set([x.strip() for x in list2])
            else:
                set1 = set([x.strip() for x in gt.split("\n")])
                set2 = set([x.strip() for x in prediction.split("\n")])

            result["em"] = int(set1 == set2)

            bestmatch = np.zeros(len(set1))
            for i,e1 in enumerate(list(set1)):
                for e2 in list(set2):
                    f1 = _qa_f1(e1,e2)
                    if bestmatch[i] < f1:
                        bestmatch[i] = f1
            bestmatch = bestmatch > list_f1_treshold
            result["f1"] = sum(bestmatch)/len(set1.union(set2))
        elif question_type == "Boolean":
            gt = str(gt).lower()
            prediction = str(prediction).lower()
            result["em"] = int(gt == prediction)
            result["f1"] = int(gt == prediction)
        elif question_type == "Numeric":
            try:
                gt = int(gt)
                prediction = int(prediction)
                result["em"] = int(gt == prediction)
                result["f1"] = 1 - (abs(gt - prediction) / max([gt, prediction]))
            except ValueError:
                result["em"] = 0
                result["f1"] = 0
        else:
            logger.warning(f"could not understand question type {question_type}")

        if result["f1"] > best_result["f1"]:
            best_result = result
    return best_result

def _qa_f1(answer1, answer2):
    # remove a, an, and
    answer1 = re.sub(r'\b(a|an|the)\b', ' ', answer1)
    answer2 = re.sub(r'\b(a|an|the)\b', ' ', answer2)
    #remove punctuation
    answer1 = ' '.join(ch for ch in answer1 if ch not in set(string.punctuation))
    answer2 = ' '.join(ch for ch in answer2 if ch not in set(string.punctuation))

    ans1_tokens = answer1.split()
    ans2_tokens = answer2.split()
    n_overlap = len([x for x in ans1_tokens if x in ans2_tokens])
    if n_overlap == 0:
        return 0.0
    precision = n_overlap / len(ans1_tokens)
    recall = n_overlap / len(ans2_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
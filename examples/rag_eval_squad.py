import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def aggregate_wiki_title(data, agg_wiki_title):
    for idx, x in enumerate(data.iter(batch_size=1)):
        if x["context"] not in agg_wiki_title[x["title"][0]]["context"]:
            agg_wiki_title[x["title"][0]]["context"].append(x["context"])
        agg_wiki_title[x["title"][0]]["question_answers"].append({"question": x["question"], "answers": x["answers"]})


def main():
    data_train = load_dataset("squad", split="train")
    data_validation = load_dataset("squad", split="validation")
    agg_wiki_title = defaultdict(lambda: {"context": [], "question_answers": [], "text": ""})
    aggregate_wiki_title(data_train, agg_wiki_title)
    aggregate_wiki_title(data_validation, agg_wiki_title)

    # merge the context into a single document
    for article in tqdm(agg_wiki_title.keys()):
        agg_wiki_title[article]["text"] = "\n".join([x[0] for x in agg_wiki_title[article]["context"]])

    # create documents
    for article in agg_wiki_title.keys():
        out_path = Path("transformed_squad/articles/")
        out_path.mkdir(parents=True, exist_ok=True)
        with open(f"{str(out_path)}/{article}.txt", "w") as f:
            f.write(agg_wiki_title[article]["text"])

    # create question/answers
    questions = Path("transformed_squad/")
    questions.mkdir(parents=True, exist_ok=True)
    with open(f"{str(questions)}/questions.jsonl", "w") as f:
        for article in agg_wiki_title.keys():
            for entry in agg_wiki_title[article]["question_answers"]:
                f.write(
                    json.dumps({"question": entry["question"][0], "document": article, "answers": entry["answers"][0]})
                    + "\n"
                )


if __name__ == "__main__":
    main()

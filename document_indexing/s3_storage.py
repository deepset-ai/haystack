from dataclasses import dataclass
from typing import List

import pandas as pd
from smart_open import open
import json
import boto3

import sys
import os
# this is horrible until I find a prettier solution
myDir = os.getcwd()
sys.path.append(myDir)
from pathlib import Path
path = Path(myDir)
sys.path.append(str(path.parent.absolute()))

from rest_api.rest_api.schema import QuestionAnswerPair

@dataclass
class S3Storage:
    """
    structure of the storage:

    board-games-rules-explainer/
        faq/
            game1.txt # hold json Q&A objects
            game2.txt

        raw/
            game1/
                doc1.json
                doc2.json
            game2/
                doc3.json

        processed/
            game1/
                doc1.json
                doc2.json
            game2/
                doc3.json

        embeddings/
            game1/
                doc1.json
                doc2.json
            game2/
                doc3.json

    """

    bucket = "board-games-rules-explainer"

    def __post_init__(self):
        self.session = boto3.Session(profile_name="fsdl22")
        self.client = self.session.client("s3")

    def upload_qa_pairs(self, qa_pairs: List[QuestionAnswerPair]):

        games = list(set([qa_pair.game for qa_pair in qa_pairs]))
        if len(games) > 1:
            raise ValueError(
                f"upload_qa_pairs can only process qa pairs for the same game, "
                f"received pair for games {games}"
            )
        game = games[0]

        # smart-open has no append mode so first read existing pairs if exists
        try:
            with open(
                f"s3://{self.bucket}/{game}.txt",
                "r",
                transport_params={"client": self.client},
            ) as fin:
                existing_pairs = fin.readlines()
        except OSError:
            print(f"There are no existing Q&A pairs for game {game}")
            existing_pairs = []

        all_pairs = set(
            [json.dumps(qa_pair.dict()) for qa_pair in qa_pairs] + existing_pairs
        )
        with open(
            f"s3://{self.bucket}/{game}.txt",
            "w",
            transport_params={"client": self.client},
        ) as fin:
            for s in all_pairs:
                fin.write(s.replace("\n", ""))
                fin.write("\n")

    def load_qa_pairs(self, game: str) -> List[QuestionAnswerPair]:
        with open(
            f"s3://{self.bucket}/{game}.txt",
            "r",
            transport_params={"client": self.client},
        ) as fin:
            serialized = fin.readlines()
        return [QuestionAnswerPair(**json.loads(s)) for s in serialized]

    def _download_rulebook_from_s3(self, game: str, local_path: str) -> None:
        if not os.path.exists(local_path):
            os.makedirs('./tmp/', exist_ok=True)
            self.client.download_file(
                self.bucket,
                f'{game}.pdf',
                local_path
            )

    def load_rulebook_path(self, game: str) -> str:
        local_download_path = f'./tmp/{game}.pdf'
        self._download_rulebook_from_s3(game, local_download_path)
        return local_download_path

    def upload_documents(self):
        raise NotImplementedError

    def upload_embeddings(self):
        raise NotImplementedError


def upload_monopoly_google_faq_sheet(path_to_csv_dump: str):
    df = pd.read_csv(path_to_csv_dump)
    df.columns = df.columns.str.lower()
    df["game"] = "monopoly"
    df["approved"] = True
    records = df[["game", "answer", "question", "approved"]].to_dict(orient="records")
    qa_pairs = [QuestionAnswerPair(**record) for record in records]
    S3Storage().upload_qa_pairs(qa_pairs)


if __name__ == "__main__":
    upload_monopoly_google_faq_sheet("Q&A - Sheet1.csv")
    result = S3Storage().load_qa_pairs("monopoly")
    print(result)

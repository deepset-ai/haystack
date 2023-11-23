from typing import List
from unittest.mock import patch, Mock
from uuid import UUID

from numpy import loadtxt

import pytest

from haystack.schema import Document
from haystack.nodes import SentenceTransformersRanker, TopPSampler, TransformersDocumentClassifier


@pytest.fixture
def docs_with_true_emb(test_rootdir):
    return [
        Document(
            content="The capital of Germany is the city state of Berlin.",
            embedding=loadtxt(test_rootdir / "samples" / "embeddings" / "embedding_1.txt"),
        ),
        Document(
            content="Berlin is the capital and largest city of Germany by both area and population.",
            embedding=loadtxt(test_rootdir / "samples" / "embeddings" / "embedding_2.txt"),
        ),
    ]


@pytest.fixture
def docs_with_ids(docs) -> List[Document]:
    # Should be already sorted
    uuids = [
        UUID("190a2421-7e48-4a49-a639-35a86e202dfb"),
        UUID("20ff1706-cb55-4704-8ae8-a3459774c8dc"),
        UUID("5078722f-07ae-412d-8ccb-b77224c4bacb"),
        UUID("81d8ca45-fad1-4d1c-8028-d818ef33d755"),
        UUID("f985789f-1673-4d8f-8d5f-2b8d3a9e8e23"),
    ]
    uuids.sort()
    for doc, uuid in zip(docs, uuids):
        doc.id = str(uuid)
    return docs


@pytest.fixture
def ranker_two_logits():
    return SentenceTransformersRanker(model_name_or_path="deepset/gbert-base-germandpr-reranking")


@pytest.fixture
def ranker():
    return SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")


@pytest.fixture
def top_p_sampler():
    return TopPSampler()


@pytest.fixture
def document_classifier():
    return TransformersDocumentClassifier(
        model_name_or_path="bhadresh-savani/distilbert-base-uncased-emotion", use_gpu=False, top_k=2
    )


@pytest.fixture
def zero_shot_document_classifier():
    return TransformersDocumentClassifier(
        model_name_or_path="cross-encoder/nli-distilroberta-base",
        use_gpu=False,
        task="zero-shot-classification",
        labels=["negative", "positive"],
    )


@pytest.fixture
def batched_document_classifier():
    return TransformersDocumentClassifier(
        model_name_or_path="bhadresh-savani/distilbert-base-uncased-emotion", use_gpu=False, batch_size=16
    )


@pytest.fixture
def indexing_document_classifier():
    return TransformersDocumentClassifier(
        model_name_or_path="bhadresh-savani/distilbert-base-uncased-emotion",
        use_gpu=False,
        batch_size=16,
        classification_field="class_field",
    )


example_serperdev_response = {
    "searchParameters": {
        "q": "Who is the boyfriend of Olivia Wilde?",
        "gl": "us",
        "hl": "en",
        "autocorrect": True,
        "type": "search",
    },
    "organic": [
        {
            "title": "Olivia Wilde embraces Jason Sudeikis amid custody battle, Harry Styles split - Page Six",
            "link": "https://pagesix.com/2023/01/29/olivia-wilde-hugs-it-out-with-jason-sudeikis-after-harry-styles-split/",
            "snippet": "Looks like Olivia Wilde and Jason Sudeikis are starting 2023 on good terms. Amid their highly publicized custody battle – and the actress' ...",
            "date": "Jan 29, 2023",
            "position": 1,
        },
        {
            "title": "Olivia Wilde Is 'Quietly Dating' Again Following Harry Styles Split: 'He Makes Her Happy'",
            "link": "https://www.yahoo.com/now/olivia-wilde-quietly-dating-again-183844364.html",
            "snippet": "Olivia Wilde is “quietly dating again” following her November 2022 split from Harry Styles, a source exclusively tells Life & Style.",
            "date": "Feb 10, 2023",
            "position": 2,
        },
        {
            "title": "Olivia Wilde and Harry Styles' Relationship Timeline: The Way They Were - Us Weekly",
            "link": "https://www.usmagazine.com/celebrity-news/pictures/olivia-wilde-and-harry-styles-relationship-timeline/",
            "snippet": "Olivia Wilde started dating Harry Styles after ending her years-long engagement to Jason Sudeikis — see their relationship timeline.",
            "date": "Mar 10, 2023",
            "imageUrl": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSgTcalNFvptTbYBiDXX55s8yCGfn6F1qbed9DAN16LvynTr9GayK5SPmY&s",
            "position": 3,
        },
        {
            "title": "Olivia Wilde Is 'Ready to Date Again' After Harry Styles Split - Us Weekly",
            "link": "https://www.usmagazine.com/celebrity-news/news/olivia-wilde-is-ready-to-date-again-after-harry-styles-split/",
            "snippet": "Ready for love! Olivia Wilde is officially back on the dating scene following her split from her ex-boyfriend, Harry Styles.",
            "date": "Mar 1, 2023",
            "imageUrl": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRCRAeRy5sVE631ZctzbzuOF70xkIOHaTvh2K7dYvdiVBwALiKrIjpscok&s",
            "position": 4,
        },
        {
            "title": "Harry Styles and Olivia Wilde's Definitive Relationship Timeline - Harper's Bazaar",
            "link": "https://www.harpersbazaar.com/celebrity/latest/a35172115/harry-styles-olivia-wilde-relationship-timeline/",
            "snippet": "November 2020: News breaks about Olivia splitting from fiancé Jason Sudeikis. ... In mid-November, news breaks of Olivia Wilde's split from Jason ...",
            "date": "Feb 23, 2023",
            "imageUrl": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRRqw3fvZOIGHEepxCc7yFAWYsS_v_1H6X-4nxyFJxdfRuFQw_BrI6JVzI&s",
            "position": 5,
        },
        {
            "title": "Harry Styles and Olivia Wilde's Relationship Timeline - People",
            "link": "https://people.com/music/harry-styles-olivia-wilde-relationship-timeline/",
            "snippet": "Harry Styles and Olivia Wilde first met on the set of Don't Worry Darling and stepped out as a couple in January 2021. Relive all their biggest relationship ...",
            "position": 6,
        },
        {
            "title": "Jason Sudeikis and Olivia Wilde's Relationship Timeline - People",
            "link": "https://people.com/movies/jason-sudeikis-olivia-wilde-relationship-timeline/",
            "snippet": "Jason Sudeikis and Olivia Wilde ended their engagement of seven years in 2020. Here's a complete timeline of their relationship.",
            "date": "Mar 24, 2023",
            "imageUrl": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSleZoXusQyJJe2WMgIuck_cVaJ8AE0_hU2QxsXzYvKANi55UQlv82yAVI&s",
            "position": 7,
        },
        {
            "title": "Olivia Wilde's anger at ex-boyfriend Harry Styles: She resents him and thinks he was using her | Marca",
            "link": "https://www.marca.com/en/lifestyle/celebrities/2023/02/23/63f779a4e2704e8d988b4624.html",
            "snippet": "The two started dating after Wilde split up with actor Jason Sudeikisin 2020. However, their relationship came to an end last November.",
            "date": "Feb 23, 2023",
            "imageUrl": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQBgJF2mSnIWCvPrqUqM4WTI9xPNWPyLvHuune85swpB1yE_G8cy_7KRh0&s",
            "position": 8,
        },
        {
            "title": "Olivia Wilde's dating history: Who has the actress dated? | The US Sun",
            "link": "https://www.the-sun.com/entertainment/5221040/olivia-wildes-dating-history/",
            "snippet": "AMERICAN actress Olivia Wilde started dating Harry Styles in January 2021 after breaking off her engagement the year prior.",
            "date": "Nov 19, 2022",
            "imageUrl": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTpm8BToVFHJoH6yRggg0fLocLT9mt6lwsnRxFFDNdDGhDydzQiSKZ9__g&s",
            "position": 9,
        },
    ],
    "relatedSearches": [
        {"query": "Harry Styles girlfriends in order"},
        {"query": "Harry Styles and Olivia Wilde engaged"},
        {"query": "Harry Styles and Olivia Wilde wedding"},
        {"query": "Who is Harry Styles married to"},
        {"query": "Jason Sudeikis Olivia Wilde relationship"},
        {"query": "Olivia Wilde and Jason Sudeikis kids"},
        {"query": "Olivia Wilde children"},
        {"query": "Harry Styles and Olivia Wilde age difference"},
        {"query": "Jason Sudeikis Olivia Wilde, Harry Styles"},
    ],
}


@pytest.fixture
def mock_web_search():
    with patch("haystack.nodes.search_engine.providers.requests") as mock_run:
        mock_run.request.return_value = Mock(status_code=200, json=lambda: example_serperdev_response)
        yield mock_run

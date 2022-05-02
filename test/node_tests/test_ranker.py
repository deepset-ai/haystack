from haystack.schema import Document
from haystack.nodes.ranker.base import BaseRanker
from haystack.nodes.ranker.sentence_transformers import SentenceTransformersRanker


def test_ranker(ranker):
    assert isinstance(ranker, BaseRanker)
    assert isinstance(ranker, SentenceTransformersRanker)
    query = "What is the most important building in King's Landing that has a religious background?"
    docs = [
        Document(
            content="""Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother's spokesman (""prophet"") to the Pharaoh. Part of the Law (Torah) that Moses received from""",
            meta={"name": "0"},
            id="1",
        ),
        Document(
            content="""Democratic Republic of the Congo to the south. Angola's capital, Luanda, lies on the Atlantic coast in the northwest of the country. Angola, although located in a tropical zone, has a climate that is not characterized for this region, due to the confluence of three factors: As a result, Angola's climate is characterized by two seasons: rainfall from October to April and drought, known as ""Cacimbo"", from May to August, drier, as the name implies, and with lower temperatures. On the other hand, while the coastline has high rainfall rates, decreasing from North to South and from to , with""",
            id="2",
        ),
        Document(
            content="""Schopenhauer, describing him as an ultimately shallow thinker: ""Schopenhauer has quite a crude mind ... where real depth starts, his comes to an end."" His friend Bertrand Russell had a low opinion on the philosopher, and attacked him in his famous ""History of Western Philosophy"" for hypocritically praising asceticism yet not acting upon it. On the opposite isle of Russell on the foundations of mathematics, the Dutch mathematician L. E. J. Brouwer incorporated the ideas of Kant and Schopenhauer in intuitionism, where mathematics is considered a purely mental activity, instead of an analytic activity wherein objective properties of reality are""",
            meta={"name": "1"},
            id="3",
        ),
        Document(
            content="""The Dothraki vocabulary was created by David J. Peterson well in advance of the adaptation. HBO hired the Language Creatio""",
            meta={"name": "2"},
            id="4",
        ),
        Document(
            content="""The title of the episode refers to the Great Sept of Baelor, the main religious building in King's Landing, where the episode's pivotal scene takes place. In the world created by George R. R. Martin""",
            meta={},
            id="5",
        ),
    ]
    results = ranker.predict(query=query, documents=docs)
    assert results[0] == docs[4]


def test_ranker_two_logits(ranker_two_logits):
    assert isinstance(ranker_two_logits, BaseRanker)
    assert isinstance(ranker_two_logits, SentenceTransformersRanker)
    query = "Welches ist das wichtigste Gebäude in Königsmund, das einen religiösen Hintergrund hat?"
    docs = [
        Document(
            content="""Aaron Aaron (oder ; "Ahärôn") ist ein Prophet, Hohepriester und der Bruder von Moses in den abrahamitischen Religionen. Aaron ist ebenso wie sein Bruder Moses ausschließlich aus religiösen Texten wie der Bibel und dem Koran bekannt. Die hebräische Bibel berichtet, dass Aaron und seine ältere Schwester Mirjam im Gegensatz zu Mose, der am ägyptischen Königshof aufwuchs, bei ihren Verwandten im östlichen Grenzland Ägyptens (Goschen) blieben. Als Mose den ägyptischen König zum ersten Mal mit den Israeliten konfrontierte, fungierte Aaron als Sprecher ("Prophet") seines Bruders gegenüber dem Pharao. Ein Teil des Gesetzes (Tora), das Mose von""",
            meta={"name": "0"},
            id="1",
        ),
        Document(
            content="""Demokratische Republik Kongo im Süden. Die angolanische Hauptstadt Luanda liegt an der Atlantikküste im Nordwesten des Landes. Angola liegt zwar in einer tropischen Zone, hat aber ein Klima, das aufgrund des Zusammenwirkens von drei Faktoren nicht für diese Region typisch ist: So ist das Klima Angolas durch zwei Jahreszeiten gekennzeichnet: Regenfälle von Oktober bis April und die als "Cacimbo" bezeichnete Dürre von Mai bis August, die, wie der Name schon sagt, trockener ist und niedrigere Temperaturen aufweist. Andererseits sind die Niederschlagsmengen an der Küste sehr hoch und nehmen von Norden nach Süden und von Süden nach Süden ab, mit""",
            id="2",
        ),
        Document(
            content="""Schopenhauer, indem er ihn als einen letztlich oberflächlichen Denker beschreibt: ""Schopenhauer hat einen ziemlich groben Verstand ... wo wirkliche Tiefe beginnt, hört seine auf."" Sein Freund Bertrand Russell hatte eine schlechte Meinung von dem Philosophen und griff ihn in seiner berühmten "Geschichte der westlichen Philosophie" an, weil er heuchlerisch die Askese lobte, aber nicht danach handelte. Der holländische Mathematiker L. E. J. Brouwer, der auf der gegenüberliegenden Insel von Russell über die Grundlagen der Mathematik sprach, nahm die Ideen von Kant und Schopenhauer in den Intuitionismus auf, in dem die Mathematik als eine rein geistige Tätigkeit betrachtet wird und nicht als eine analytische Tätigkeit, bei der die objektiven Eigenschaften der Realität berücksichtigt werden.""",
            meta={"name": "1"},
            id="3",
        ),
        Document(
            content="""Das dothrakische Vokabular wurde von David J. Peterson lange vor der Verfilmung erstellt. HBO beauftragte das Language Creatio""",
            meta={"name": "2"},
            id="4",
        ),
        Document(
            content="""Der Titel der Episode bezieht sich auf die Große Septe von Baelor, das wichtigste religiöse Gebäude in Königsmund, in dem die Schlüsselszene der Episode stattfindet. In der von George R. R. Martin geschaffenen Welt""",
            meta={},
            id="5",
        ),
    ]
    results = ranker_two_logits.predict(query=query, documents=docs)
    assert results[0] == docs[4]

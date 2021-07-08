from haystack import Document
from haystack.ranker import FARMRanker
from haystack.ranker.base import BaseRanker
from haystack.ranker.sentence_transformers import SentenceTransformersRanker


def test_ranker(ranker):
    assert isinstance(ranker, BaseRanker)

    if isinstance(ranker, FARMRanker):
        query = "Welches ist die zweitgrößte Stadt in den Alpen?"
        docs = [
            Document(
                text="""Deçan\n\n== Geographie ==\nDeçan liegt im Westen des Kosovo auf etwa 550 Meter über Meer nahe den Grenzen zu Montenegro und Albanien. Westlich der Stadt liegt das Prokletije (auch ''Albanische Alpen'' genannt). Etwas nordwestlich  tritt der Fluss Bistrica e Deçanit aus dem Gebirge, der Deçan nördlich des Zentrums passiert. Etwa zehn Kilometer im Südosten befindet sich der Radoniq-Stausee, welcher der zweitgrößte See im Land ist. Deçan befindet sich zirka auf halbem Weg zwischen Gjakova und Peja. Die Hauptstadt Pristina liegt rund 70 Kilometer im Osten.""",
                meta={"name": "0"},
                id="1",
            ),
            Document(
                text="""Alpen\n\n=== Städte ===\nInnerhalb der Alpen ist das französische Grenoble die größte Stadt, gefolgt von Innsbruck in Österreich sowie von Trient und Bozen in Italien. In der Schweiz liegen Chur, Thun und Lugano in den Alpen. Weitere Alpenstädte in Österreich sind Klagenfurt und Villach, sowie im Rheintal Bregenz, Dornbirn und Feldkirch. Ferner zu nennen ist Vaduz, die Hauptstadt Liechtensteins. Die höchste Stadt der Alpen (und Europas) ist das schweizerische Davos.\nIn direkter Alpenrandlage ist Wien die weitaus größte Stadt, gefolgt von Genf (Schweiz) und Nizza (Frankreich). Weitere wichtige Städte sind – von Ost nach West – Maribor (Slowenien), Graz (Österreich), Ljubljana (Slowenien), Udine (Italien), Salzburg (Österreich), Vicenza (Italien), Verona (Italien), Brescia (Italien), Bergamo (Italien), St. Gallen (Schweiz), Lecco (Italien), Como (Italien), Varese (Italien), Luzern (Schweiz), Savona (Italien), Biella (Italien), San Remo (Italien), Cuneo (Italien), Bern (Schweiz) und Monaco.""",
                meta={"name": "1"},
                id="2",
            ),
            Document(
                text="""Latumer_Bruch\nDer Latumer Bruch, lokal auch ''Lohbruch'' genannt, ist ein Bruchwald- und Feuchtgebiet im Südosten der Stadt Krefeld, welches unter gleichem Namen das zweitgrößte Naturschutzgebiet der Stadt bildet (Nr. ''KR-001'').\nDer Bruch liegt am südlichen Rand des Krefelder Stadtteils Linn. Im Nordwesten grenzt das Gebiet an Oppum, im Nordosten an Gellep-Stratum, im Südwesten und Südosten liegen die Meerbuscher Stadtteile Ossum-Bösinghoven und Lank-Latum. Benannt ist der Latumer Bruch nach dem Haus Latum, einem Gutshof am Ortsrand von Lank-Latum, zu dessen Ländereien das Gebiet historisch gehörte.""",
                meta={"name": "2"},
                id="3",
            ),
            Document(
                text="""Großglockner\n\n=== Lage und Umgebung ===\nDer Großglockner ist Teil des ''Glocknerkamms'', eines Gebirgskamms der Glocknergruppe (Österreichische Zentralalpen), der am Eiskögele in südöstlicher Richtung vom Alpenhauptkamm abzweigt und dort die Grenze zwischen den Bundesländern Tirol im Südwesten und Kärnten im Nordosten bildet. Diese Grenze ist auch die Wasserscheide zwischen dem Kalser Tal mit seinen Seitentälern auf der Osttiroler und dem Mölltal mit der Pasterze auf der Kärntner Seite. Die Gegend um den Berg ist außerdem seit 1986 Bestandteil des ''Sonderschutzgebietes Großglockner-Pasterze'' innerhalb des Nationalparks Hohe Tauern.\nDer Großglockner ist der höchste Berg der Alpen östlich der 175 km entfernten Ortlergruppe und weist damit nach dem Mont Blanc die zweitgrößte geografische Dominanz aller Berge der Alpen auf. Auch seine Schartenhöhe ist mit 2.424 Metern nach dem Montblanc die zweitgrößte aller Alpengipfel. Somit ist der Berg eine der eigenständigsten Erhebungen der Alpen. Die Aussicht vom Großglockner gilt als die weiteste aller Berge der Ostalpen, sie reicht 220 Kilometer weit, unter Berücksichtigung der terrestrischen Refraktion fast 240 Kilometer. Der Blick über mehr als 150.000 Quadratkilometer Erdoberfläche reicht bis zur Schwäbisch-Bayerischen Ebene im Nordwesten, bis Regensburg und zum Böhmerwald im Norden, zum Ortler im Westen, zur Poebene im Süden, zum Triglav und zum Toten Gebirge im Osten.\nDie bedeutendsten Orte in der Umgebung des Berges sind Kals am Großglockner () im Kalser Tal in Osttirol, vom Gipfel aus ungefähr acht Kilometer in südwestlicher Richtung gelegen, und Heiligenblut am Großglockner () im Mölltal in Kärnten, vom Gipfel aus ca. zwölf Kilometer in südöstlicher Richtung.""",
                meta={"name": "3"},
                id="4",
            ),
        ]
        results = ranker.predict(query=query, documents=docs)
        assert results[0] == docs[1]
    elif isinstance(ranker, SentenceTransformersRanker):
        query = "What is the most important building in King's Landing that has a religious background?"
        docs = [
            Document(
                text="""Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother's spokesman (""prophet"") to the Pharaoh. Part of the Law (Torah) that Moses received from""",
                meta={"name": "0"},
                id="1",
            ),
            Document(
                text="""Democratic Republic of the Congo to the south. Angola's capital, Luanda, lies on the Atlantic coast in the northwest of the country. Angola, although located in a tropical zone, has a climate that is not characterized for this region, due to the confluence of three factors: As a result, Angola's climate is characterized by two seasons: rainfall from October to April and drought, known as ""Cacimbo"", from May to August, drier, as the name implies, and with lower temperatures. On the other hand, while the coastline has high rainfall rates, decreasing from North to South and from to , with""",
                id="2",
            ),
            Document(
                text="""Schopenhauer, describing him as an ultimately shallow thinker: ""Schopenhauer has quite a crude mind ... where real depth starts, his comes to an end."" His friend Bertrand Russell had a low opinion on the philosopher, and attacked him in his famous ""History of Western Philosophy"" for hypocritically praising asceticism yet not acting upon it. On the opposite isle of Russell on the foundations of mathematics, the Dutch mathematician L. E. J. Brouwer incorporated the ideas of Kant and Schopenhauer in intuitionism, where mathematics is considered a purely mental activity, instead of an analytic activity wherein objective properties of reality are""",
                meta={"name": "1"},
                id="3",
            ),
            Document(
                text="""The Dothraki vocabulary was created by David J. Peterson well in advance of the adaptation. HBO hired the Language Creatio""",
                meta={"name": "2"},
                id="4",
            ),
            Document(
                text="""The title of the episode refers to the Great Sept of Baelor, the main religious building in King's Landing, where the episode's pivotal scene takes place. In the world created by George R. R. Martin""",
                meta={},
                id="5",
            ),
        ]
        results = ranker.predict(query=query, documents=docs)
        assert results[0] == docs[4]

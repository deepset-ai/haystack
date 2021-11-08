import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import SequentialSampler
from tqdm import tqdm

from haystack.modeling.data_handler.dataloader import NamedDataLoader
from haystack.modeling.data_handler.processor import TextSimilarityProcessor
from haystack.modeling.model.biadaptive_model import BiAdaptiveModel
from haystack.modeling.model.language_model import LanguageModel, DPRContextEncoder, DPRQuestionEncoder
from haystack.modeling.model.prediction_head import TextSimilarityHead
from haystack.modeling.model.tokenization import Tokenizer
from haystack.modeling.utils import set_all_seeds, initialize_device_settings


def test_dpr_modules(caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    devices, n_gpu = initialize_device_settings(use_cuda=True)

    # 1.Create question and passage tokenizers
    query_tokenizer = Tokenizer.load(pretrained_model_name_or_path="facebook/dpr-question_encoder-single-nq-base",
                                     do_lower_case=True, use_fast=True)
    passage_tokenizer = Tokenizer.load(pretrained_model_name_or_path="facebook/dpr-ctx_encoder-single-nq-base",
                                       do_lower_case=True, use_fast=True)

    processor = TextSimilarityProcessor(
        query_tokenizer=query_tokenizer,
        passage_tokenizer=passage_tokenizer,
        max_seq_len_query=256,
        max_seq_len_passage=256,
        label_list=["hard_negative", "positive"],
        metric="text_similarity_metric",
        data_dir="data/retriever",
        train_filename="nq-train.json",
        dev_filename="nq-dev.json",
        test_filename="nq-dev.json",
        embed_title=True,
        num_hard_negatives=1
    )

    question_language_model = LanguageModel.load(pretrained_model_name_or_path="bert-base-uncased",
                                                 language_model_class="DPRQuestionEncoder",
                                                 hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    passage_language_model = LanguageModel.load(pretrained_model_name_or_path="bert-base-uncased",
                                                language_model_class="DPRContextEncoder",
                                                hidden_dropout_prob=0, attention_probs_dropout_prob=0)

    prediction_head = TextSimilarityHead(similarity_function="dot_product")

    model = BiAdaptiveModel(
        language_model1=question_language_model,
        language_model2=passage_language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.0,
        lm1_output_types=["per_sequence"],
        lm2_output_types=["per_sequence"],
        device=devices[0],
    )

    model.connect_heads_with_processor(processor.tasks)

    assert type(model) == BiAdaptiveModel
    assert type(processor) == TextSimilarityProcessor
    assert type(question_language_model) == DPRQuestionEncoder
    assert type(passage_language_model) == DPRContextEncoder

    # check embedding layer weights
    assert list(model.named_parameters())[0][1][0, 0].item() - -0.010200000368058681 < 0.0001

    d = {'query': 'big little lies season 2 how many episodes',
         'passages': [
                         {'title': 'Big Little Lies (TV series)',
                          'text': 'series garnered several accolades. It received 16 Emmy Award nominations and won eight, including Outstanding Limited Series and acting awards for Kidman, Skarsgård, and Dern. The trio also won Golden Globe Awards in addition to a Golden Globe Award for Best Miniseries or Television Film win for the series. Kidman and Skarsgård also received Screen Actors Guild Awards for their performances. Despite originally being billed as a miniseries, HBO renewed the series for a second season. Production on the second season began in March 2018 and is set to premiere in 2019. All seven episodes are being written by Kelley',
                          'label': 'positive',
                          'external_id': '18768923'},
                         {'title': 'Little People, Big World',
                          'text': 'final minutes of the season two-A finale, "Farm Overload". A crowd had gathered around Jacob, who was lying on the ground near the trebuchet. The first two episodes of season two-B focus on the accident, and how the local media reacted to it. The first season of "Little People, Big World" generated solid ratings for TLC (especially in the important 18–49 demographic), leading to the show\'s renewal for a second season. Critical reviews of the series have been generally positive, citing the show\'s positive portrayal of little people. Conversely, other reviews have claimed that the show has a voyeuristic bend',
                          'label': 'hard_negative',
                          'external_id': '7459116'},
                         {'title': 'Cormac McCarthy',
                          'text': 'chores of the house, Lee was asked by Cormac to also get a day job so he could focus on his novel writing. Dismayed with the situation, she moved to Wyoming, where she filed for divorce and landed her first job teaching. Cormac McCarthy is fluent in Spanish and lived in Ibiza, Spain, in the 1960s and later settled in El Paso, Texas, where he lived for nearly 20 years. In an interview with Richard B. Woodward from "The New York Times", "McCarthy doesn\'t drink anymore – he quit 16 years ago in El Paso, with one of his young',
                          'label': 'negative',
                          'passage_id': '2145653'}
                     ]
         }

    dataset, tensor_names, _ = processor.dataset_from_dicts(dicts=[d], return_baskets=False)
    features = {key: val.unsqueeze(0).to(devices[0]) for key, val in zip(tensor_names, dataset[0])}

    # test features
    assert torch.all(torch.eq(features["query_input_ids"][0][:10].cpu(),
                              torch.tensor([101, 2502, 2210, 3658, 2161, 1016, 2129, 2116, 4178, 102])))
    assert torch.all(torch.eq(features["passage_input_ids"][0][0][:10].cpu(),
                              torch.tensor([101,  2502,  2210,  3658,  1006,  2694,  2186,  1007,   102,  2186])))
    assert len(features["query_segment_ids"][0].nonzero()) == 0
    assert len(features["passage_segment_ids"][0].nonzero()) == 0
    assert torch.all(torch.eq(features["query_attention_mask"].nonzero()[:, 1].cpu(), torch.tensor(list(range(10)))))
    assert torch.all(torch.eq(features["passage_attention_mask"][0][0].nonzero().cpu().squeeze(), torch.tensor(list(range(127)))))
    assert torch.all(torch.eq(features["passage_attention_mask"][0][1].nonzero().cpu().squeeze(), torch.tensor(list(range(143)))))

    # test model encodings
    query_vector = model.language_model1(**features)[0]
    passage_vector = model.language_model2(**features)[0]
    assert torch.all(torch.le(query_vector[0, :10].cpu() - torch.tensor([-0.2135, -0.4748, 0.0501, -0.0430, -0.1747, -0.0441, 0.5638, 0.1405,
                                                                         0.2285, 0.0893]), torch.ones((1, 10))*0.0001))
    assert torch.all(torch.le(passage_vector[0, :10].cpu() - torch.tensor([0.0557, -0.6836, -0.3645, -0.5566,  0.2034, -0.3656,  0.2969, -0.0555,
                                                                          0.3405, -0.8691]), torch.ones((1, 10))*0.0001))
    assert torch.all(torch.le(passage_vector[1, :10].cpu() - torch.tensor([-0.2006, -1.5002, -0.1897, -0.3421, -0.0405, -0.0471, -0.0306,  0.1156,
                                                                           0.3350, -0.3412]), torch.ones((1, 10)) * 0.0001))

    # test logits and loss
    embeddings = model(**features)
    query_emb, passage_emb = embeddings[0]
    assert torch.all(torch.eq(query_emb.cpu(), query_vector.cpu()))
    assert torch.all(torch.eq(passage_emb.cpu(), passage_vector.cpu()))

    loss = model.logits_to_loss_per_head(embeddings, **features)
    similarity_scores = model.prediction_heads[0]._embeddings_to_scores(query_emb, passage_emb).cpu()
    assert torch.all(torch.le(similarity_scores - torch.tensor([[-1.8311e-03, -6.3016e+00]]), torch.ones((1, 2)) * 0.0001))
    assert (loss[0].item() - 0.0018) <= 0.0001



query_input_ids = [torch.tensor([101, 2073, 2003, 3317, 2006, 1996, 2940, 2241, 2006, 102]),
                   torch.tensor([101, 2043, 2106, 1996, 2548, 2155, 11092, 1996, 2171, 10064]),
                   torch.tensor([101, 2054, 2003, 1037, 4937,  102,    0,    0,    0,    0])]
query_attention_mask = [torch.tensor(range(10)).unsqueeze(-1), torch.tensor(range(11)).unsqueeze(-1), torch.tensor(range(6)).unsqueeze(-1)]
passage_ids = {
            'titled': [torch.tensor([[101, 3317, 2006, 1996, 2940,  102, 3317, 2006, 1996, 2940],
                                     [101, 3317, 2940, 1010, 2047, 2148, 3575,  102, 8765, 2061],
                                     [101, 3317, 2940, 1010, 27492, 102, 3419, 18874, 3385, 1010]]),
                       torch.tensor([[101,  2160,  1997, 10064,   102,  2160,  1997, 10064,  1996,  2160],
                                     [101, 26902,  1010, 11017,  1997, 10387,   102,  2384,  1010,  1998],
                                     [101, 102, 102, 0, 0, 0, 0, 0, 0, 0]]),
                       torch.tensor([[101, 2516, 2007, 1000, 2569, 3494, 1000,  102, 2023, 2003],
                                     [101, 102, 102, 0, 0, 0, 0, 0, 0, 0],
                                     [101, 102, 102, 0, 0, 0, 0, 0, 0, 0]])
                       ],

            'untitled': [torch.tensor([[101, 3317, 2006, 1996, 2940, 1000, 3317, 2006, 1996, 2940],
                                       [101, 8765, 2061, 2004, 2000, 5438, 1037, 8084, 10527, 5701],
                                       [101, 3419, 18874, 3385, 1010, 3818, 1000, 1000, 2152, 2006]]),
                         torch.tensor([[101, 2160, 1997, 10064, 1996, 2160, 1997, 10064, 2003, 1996],
                                       [101, 2384, 1010, 1998, 2001, 2000, 2202, 2173, 1999, 1037],
                                       [101, 102, 102, 0, 0, 0, 0, 0, 0, 0]]),
                         torch.tensor([[101, 2023, 2003, 1037, 1026, 7308, 1028, 6251,  1012, 8870],
                                       [101, 102, 102, 0, 0, 0, 0, 0, 0, 0],
                                       [101, 102, 102, 0, 0, 0, 0, 0, 0, 0]]),
                         ]}

passage_attention = {
        'titled': [[torch.tensor(range(140)).unsqueeze(-1), torch.tensor(range(130)).unsqueeze(-1), torch.tensor(range(127)).unsqueeze(-1)],
                   [torch.tensor(range(132)).unsqueeze(-1),  torch.tensor(range(121)).unsqueeze(-1), torch.tensor(range(3)).unsqueeze(-1)],
                   [torch.tensor(range(22)).unsqueeze(-1), torch.tensor(range(3)).unsqueeze(-1),torch.tensor(range(3)).unsqueeze(-1)]],
'untitled': [[torch.tensor(range(135)).unsqueeze(-1), torch.tensor(range(123)).unsqueeze(-1), torch.tensor(range(122)).unsqueeze(-1)],
             [torch.tensor(range(128)).unsqueeze(-1), torch.tensor(range(115)).unsqueeze(-1), torch.tensor(range(3)).unsqueeze(-1)],
             [torch.tensor(range(15)).unsqueeze(-1), torch.tensor(range(3)).unsqueeze(-1),torch.tensor(range(3)).unsqueeze(-1)]]
                    }
labels1 = [[1,0], [1,0], [1,0]]
labels2 = [[1,0,0], [1,0,0], [1,0,0]]

@pytest.mark.parametrize("embed_title, passage_ids, passage_attns", [(True, passage_ids['titled'], passage_attention['titled']),  (False, passage_ids['untitled'], passage_attention['untitled'])])
@pytest.mark.parametrize("use_fast", [True, False])
@pytest.mark.parametrize("num_hard_negatives, labels", [(1, labels1),(2, labels2)])
def test_dpr_processor(embed_title, passage_ids, passage_attns, use_fast, num_hard_negatives, labels):
    dict = [{
             'query': 'where is castle on the hill based on',
             'answers': ['Framlingham Castle'],
             'passages': [{"text": 'Castle on the Hill "Castle on the Hill" is a song by English singer-songwriter Ed Sheeran. It was released as a digital download on 6 January 2017 as one of the double lead singles from his third studio album "÷" (2017), along with "Shape of You". "Castle on the Hill" was written and produced by Ed Sheeran and Benny Blanco. The song refers to Framlingham Castle in Sheeran\'s home town. Released on the same day as "Shape of You", "Castle on the Hill" reached number two in a number of countries, including the UK, Australia and Germany, while "Shape of',
                           "title": 'Castle on the Hill',
                           "label": "positive", "external_id": '19930582'},
                          {"text": 'crops so as to feed a struggling infant colony. Governor King began Government Farm 3 there on 8 July 1801, referring to it as "Castle Hill" on 1 March 1802. The majority of the convicts who worked the prison farm were Irish Catholics, many having been transported for seditious activity in 1798. The most notorious incident being the Battle of Vinegar Hill where around 39 were slaughtered. They were branded "politicals" and exiled for life, never to return. The first free settler in Castle Hill, a Frenchman Baron Verincourt de Clambe, in unusual circumstances received a grant of 200 acres',
                           "title": 'Castle Hill, New South Wales',
                           "label": "hard_negative", "external_id": '1977568'},
                          {
                              "text": 'Tom Gleeson, proposed ""high on the peak of Castle Hill, overlooking the harbour"" would be a suitable location for the monument. Having arrived in Townsville, the monument was then placed in storage for a number of years. It was not until October 1947 that the Council discussed where to place the monument. A number of locations were considered: Castle Hill, the Botanic Gardens, in front of the Queens Hotel, the Anzac Memorial Park and the Railway Oval, but Castle Hill was ultimately the council\'s choice. In February 1948, the Queensland Government gave its approval to the council to place the',
                              "title": 'Castle Hill, Townsville',
                              "label": "hard_negative", "external_id": '3643705'},
                          ]
            },

            {'query': 'when did the royal family adopt the name windsor',
                       'answers': ['in 1917'],
                       'passages': [{"text": 'House of Windsor The House of Windsor is the reigning royal house of the United Kingdom and the other Commonwealth realms. The dynasty is of German paternal descent and was originally a branch of the House of Saxe-Coburg and Gotha, itself derived from the House of Wettin, which succeeded the House of Hanover to the British monarchy following the death of Queen Victoria, wife of Albert, Prince Consort. The name was changed from "Saxe-Coburg and Gotha" to the English "Windsor" (from "Windsor Castle") in 1917 because of anti-German sentiment in the British Empire during World War I. There have been',
                                    "title": 'House of Windsor',
                                    "label": "positive", "external_id": '1478954'},
                                    {"text": "2005, and was to take place in a civil ceremony at Windsor Castle, with a subsequent religious service of blessing at St George's Chapel. However, to conduct a civil marriage at Windsor Castle would oblige the venue to obtain a licence for civil marriages, which it did not have. A condition of such a licence is that the licensed venue must be available for a period of one year to anyone wishing to be married there, and as the royal family did not wish to make Windsor Castle available to the public for civil marriages, even just for one year,",
                                    "title": 'Camilla, Duchess of Cornwall',
                                    "label": "hard_negative", "external_id": '1399730'}]
             },

            {'query': 'what is a cat?',
             'answers': ['animal', 'feline'],
             'passages': [{
                              "text":  'This is a <mask> sentence. Cats are good pets.',
                              "title": 'title with "special characters" ',
                              "label": "positive", "external_id": '0'},
                          {
                              "text": "2nd text => More text about cats is good",
                              "title": '2nd title \n',
                              "label": "positive", "external_id": '1'}]
             }]

    query_tok = "facebook/dpr-question_encoder-single-nq-base"
    query_tokenizer = Tokenizer.load(query_tok, use_fast=use_fast)
    passage_tok = "facebook/dpr-ctx_encoder-single-nq-base"
    passage_tokenizer = Tokenizer.load(passage_tok, use_fast=use_fast)
    processor = TextSimilarityProcessor(query_tokenizer=query_tokenizer,
                                        passage_tokenizer=passage_tokenizer,
                                        max_seq_len_query=256,
                                        max_seq_len_passage=256,
                                        data_dir="data/retriever",
                                        train_filename="nq-train.json",
                                        test_filename="nq-dev.json",
                                        embed_title=embed_title,
                                        num_hard_negatives=num_hard_negatives,
                                        label_list=["hard_negative", "positive"],
                                        metric="text_similarity_metric",
                                        shuffle_negatives=False)


    for i, d in enumerate(dict):
        dataset, tensor_names, _, baskets = processor.dataset_from_dicts(dicts=[d], return_baskets=True)
        feat = baskets[0].samples[0].features
        assert (torch.all(torch.eq(torch.tensor(feat[0]["query_input_ids"][:10]), query_input_ids[i])))
        assert (len(torch.tensor(feat[0]["query_segment_ids"]).nonzero()) == 0)
        assert (torch.all(torch.eq(torch.tensor(feat[0]["query_attention_mask"]).nonzero(), query_attention_mask[i])))

        positive_indices = np.where(np.array(feat[0]["label_ids"]) == 1)[0].item()
        assert (torch.all(torch.eq(torch.tensor(feat[0]["passage_input_ids"])[positive_indices, :10], passage_ids[i][positive_indices])))
        for j in range(num_hard_negatives+1):
            assert (torch.all(torch.eq(torch.tensor(feat[0]["passage_attention_mask"][j]).nonzero(), passage_attns[i][j])))
        assert (torch.all(torch.eq(torch.tensor(feat[0]["label_ids"]), torch.tensor(labels[i])[:num_hard_negatives+1])))
        assert (len(torch.tensor(feat[0]["passage_segment_ids"]).nonzero()) == 0)


@pytest.mark.parametrize("use_fast", [False])
@pytest.mark.parametrize("embed_title", [True, False])
def test_dpr_processor_empty_title(use_fast, embed_title):
        dict = {
            'query': 'what is a cat?',
            'passages': [{'title': '',
                               'text': 'Director Radio Iași); Dragoș-Liviu Vîlceanu; Mihnea-Adrian Vîlceanu; Nathalie-Teona',
                               'label': 'positive',
                               'external_id': 'b21eaeff-e08b-4548-b5e0-a280f6f4efef'}]}

        query_tok = "facebook/dpr-question_encoder-single-nq-base"
        query_tokenizer = Tokenizer.load(query_tok, use_fast=use_fast)
        passage_tok = "facebook/dpr-ctx_encoder-single-nq-base"
        passage_tokenizer = Tokenizer.load(passage_tok, use_fast=use_fast)
        processor = TextSimilarityProcessor(query_tokenizer=query_tokenizer,
                                            passage_tokenizer=passage_tokenizer,
                                            max_seq_len_query=256,
                                            max_seq_len_passage=256,
                                            data_dir="data/retriever",
                                            train_filename="nq-train.json",
                                            test_filename="nq-dev.json",
                                            embed_title=embed_title,
                                            num_hard_negatives=1,
                                            label_list=["hard_negative", "positive"],
                                            metric="text_similarity_metric",
                                            shuffle_negatives=False)
        _ = processor.dataset_from_dicts(dicts=[dict])

def test_dpr_problematic():
    erroneous_dicts = [{
             'query': [1],
             'answers': ['Framlingham Castle'],
             'passages': [{"text": 'Castle on the Hill "Castle on the Hill" is a song by English singer-songwriter Ed Sheeran. It was released as a digital download on 6 January 2017 as one of the double lead singles from his third studio album "÷" (2017), along with "Shape of You". "Castle on the Hill" was written and produced by Ed Sheeran and Benny Blanco. The song refers to Framlingham Castle in Sheeran\'s home town. Released on the same day as "Shape of You", "Castle on the Hill" reached number two in a number of countries, including the UK, Australia and Germany, while "Shape of',
                           "title": 'Castle on the Hill',
                           "label": "positive", "external_id": '19930582'},
                          {"text": 'crops so as to feed a struggling infant colony. Governor King began Government Farm 3 there on 8 July 1801, referring to it as "Castle Hill" on 1 March 1802. The majority of the convicts who worked the prison farm were Irish Catholics, many having been transported for seditious activity in 1798. The most notorious incident being the Battle of Vinegar Hill where around 39 were slaughtered. They were branded "politicals" and exiled for life, never to return. The first free settler in Castle Hill, a Frenchman Baron Verincourt de Clambe, in unusual circumstances received a grant of 200 acres',
                           "title": 'Castle Hill, New South Wales',
                           "label": "hard_negative", "external_id": '1977568'},
                          {
                              "text": 'Tom Gleeson, proposed ""high on the peak of Castle Hill, overlooking the harbour"" would be a suitable location for the monument. Having arrived in Townsville, the monument was then placed in storage for a number of years. It was not until October 1947 that the Council discussed where to place the monument. A number of locations were considered: Castle Hill, the Botanic Gardens, in front of the Queens Hotel, the Anzac Memorial Park and the Railway Oval, but Castle Hill was ultimately the council\'s choice. In February 1948, the Queensland Government gave its approval to the council to place the',
                              "title": 'Castle Hill, Townsville',
                              "label": "hard_negative", "external_id": '3643705'},
                          ]
            },

            {'query': 'when did the royal family adopt the name windsor',
                       'answers': ['in 1917'],
                       'passages': [{"text2": 'House of Windsor The House of Windsor is the reigning royal house of the United Kingdom and the other Commonwealth realms. The dynasty is of German paternal descent and was originally a branch of the House of Saxe-Coburg and Gotha, itself derived from the House of Wettin, which succeeded the House of Hanover to the British monarchy following the death of Queen Victoria, wife of Albert, Prince Consort. The name was changed from "Saxe-Coburg and Gotha" to the English "Windsor" (from "Windsor Castle") in 1917 because of anti-German sentiment in the British Empire during World War I. There have been',
                                    "title": 'House of Windsor',
                                    "label": "positive", "external_id": '1478954'},
                                    {"text2": "2005, and was to take place in a civil ceremony at Windsor Castle, with a subsequent religious service of blessing at St George's Chapel. However, to conduct a civil marriage at Windsor Castle would oblige the venue to obtain a licence for civil marriages, which it did not have. A condition of such a licence is that the licensed venue must be available for a period of one year to anyone wishing to be married there, and as the royal family did not wish to make Windsor Castle available to the public for civil marriages, even just for one year,",
                                    "title": 'Camilla, Duchess of Cornwall',
                                    "label": "hard_negative", "external_id": '1399730'}]
             },

            {'query': 'what is a cat?',
             'answers': ['animal', 'feline'],
             'passages': [{
                              "text":  'This is a <mask> sentence. Cats are good pets.',
                              "title": 'title with "special characters" ',
                              "label": "positive", "external_id": '0'},
                          {
                              "text": "2nd text => More text about cats is good",
                              "title": '2nd title \n',
                              "label": "positive", "external_id": '1'}]
             }]

    query_tok = "facebook/dpr-question_encoder-single-nq-base"
    query_tokenizer = Tokenizer.load(query_tok, use_fast=True)
    passage_tok = "facebook/dpr-ctx_encoder-single-nq-base"
    passage_tokenizer = Tokenizer.load(passage_tok, use_fast=True)
    processor = TextSimilarityProcessor(query_tokenizer=query_tokenizer,
                                        passage_tokenizer=passage_tokenizer,
                                        max_seq_len_query=256,
                                        max_seq_len_passage=256,
                                        data_dir="data/retriever",
                                        train_filename="nq-train.json",
                                        test_filename="nq-dev.json",
                                        embed_title=True,
                                        num_hard_negatives=1,
                                        label_list=["hard_negative", "positive"],
                                        metric="text_similarity_metric",
                                        shuffle_negatives=False)



    dataset, tensor_names, problematic_ids, baskets = processor.dataset_from_dicts(dicts=erroneous_dicts, return_baskets=True)
    assert problematic_ids == {0, 1}

def test_dpr_query_only():
    erroneous_dicts = [
        {
             'query': 'where is castle on the hill based on',
             'answers': ['Framlingham Castle'],
            },
        {
            'query': 'where is castle on the hill 2 based on',
            'answers': ['Framlingham Castle 2'],
        }]

    query_tok = "facebook/dpr-question_encoder-single-nq-base"
    query_tokenizer = Tokenizer.load(query_tok, use_fast=True)
    passage_tok = "facebook/dpr-ctx_encoder-single-nq-base"
    passage_tokenizer = Tokenizer.load(passage_tok, use_fast=True)
    processor = TextSimilarityProcessor(query_tokenizer=query_tokenizer,
                                        passage_tokenizer=passage_tokenizer,
                                        max_seq_len_query=256,
                                        max_seq_len_passage=256,
                                        data_dir="data/retriever",
                                        train_filename="nq-train.json",
                                        test_filename="nq-dev.json",
                                        embed_title=True,
                                        num_hard_negatives=1,
                                        label_list=["hard_negative", "positive"],
                                        metric="text_similarity_metric",
                                        shuffle_negatives=False)



    dataset, tensor_names, problematic_ids, baskets = processor.dataset_from_dicts(dicts=erroneous_dicts, return_baskets=True)
    assert len(problematic_ids) == 0
    assert tensor_names == ['query_input_ids', 'query_segment_ids', 'query_attention_mask']

def test_dpr_context_only():
    erroneous_dicts = [{
            'passages': [{"text": 'House of Windsor 2 The House of Windsor is the reigning royal house of the United',
            "title": 'House of Windsor',
            "label": "positive", "external_id": '1478954'},
            {"text": "2005, and was to take place in a civil ceremony at Windsor Castle, with a subsequent religious",
            "title": 'Camilla, Duchess of Cornwall',
            "label": "hard_negative", "external_id": '1399730'}]
             },
        {
            'passages': [{"text": 'House of Windsor The House of Windsor is the reigning royal house of the',
            "title": 'House of Windsor',
            "label": "positive", "external_id": '1478954'},
            {"text": "2005, and was to take place in a civil ceremony at Windsor Castle, with a subsequent",
            "title": 'Camilla, Duchess of Cornwall',
            "label": "hard_negative", "external_id": '1399730'}]
             }]

    query_tok = "facebook/dpr-question_encoder-single-nq-base"
    query_tokenizer = Tokenizer.load(query_tok, use_fast=True)
    passage_tok = "facebook/dpr-ctx_encoder-single-nq-base"
    passage_tokenizer = Tokenizer.load(passage_tok, use_fast=True)
    processor = TextSimilarityProcessor(query_tokenizer=query_tokenizer,
                                        passage_tokenizer=passage_tokenizer,
                                        max_seq_len_query=256,
                                        max_seq_len_passage=256,
                                        data_dir="data/retriever",
                                        train_filename="nq-train.json",
                                        test_filename="nq-dev.json",
                                        embed_title=True,
                                        num_hard_negatives=1,
                                        label_list=["hard_negative", "positive"],
                                        metric="text_similarity_metric",
                                        shuffle_negatives=False)



    dataset, tensor_names, problematic_ids, baskets = processor.dataset_from_dicts(dicts=erroneous_dicts, return_baskets=True)
    assert len(problematic_ids) == 0
    assert tensor_names == ['passage_input_ids', 'passage_segment_ids', 'passage_attention_mask', 'label_ids']


def test_dpr_processor_save_load():
    d = {'query': 'big little lies season 2 how many episodes ?',
         'passages': [
                         {'title': 'Big Little Lies (TV series)',
                          'text': 'series garnered several accolades. It received 16 Emmy Award nominations and won eight, including Outstanding Limited Series and acting awards for Kidman, Skarsgård, and Dern. The trio also won Golden Globe Awards in addition to a Golden Globe Award for Best Miniseries or Television Film win for the series. Kidman and Skarsgård also received Screen Actors Guild Awards for their performances. Despite originally being billed as a miniseries, HBO renewed the series for a second season. Production on the second season began in March 2018 and is set to premiere in 2019. All seven episodes are being written by Kelley',
                          'label': 'positive',
                          'external_id': '18768923'},
                         {'title': 'Little People, Big World',
                          'text': 'final minutes of the season two-A finale, "Farm Overload". A crowd had gathered around Jacob, who was lying on the ground near the trebuchet. The first two episodes of season two-B focus on the accident, and how the local media reacted to it. The first season of "Little People, Big World" generated solid ratings for TLC (especially in the important 18–49 demographic), leading to the show\'s renewal for a second season. Critical reviews of the series have been generally positive, citing the show\'s positive portrayal of little people. Conversely, other reviews have claimed that the show has a voyeuristic bend',
                          'label': 'hard_negative',
                          'external_id': '7459116'},
                         {'title': 'Cormac McCarthy',
                          'text': 'chores of the house, Lee was asked by Cormac to also get a day job so he could focus on his novel writing. Dismayed with the situation, she moved to Wyoming, where she filed for divorce and landed her first job teaching. Cormac McCarthy is fluent in Spanish and lived in Ibiza, Spain, in the 1960s and later settled in El Paso, Texas, where he lived for nearly 20 years. In an interview with Richard B. Woodward from "The New York Times", "McCarthy doesn\'t drink anymore – he quit 16 years ago in El Paso, with one of his young',
                          'label': 'negative',
                          'passage_id': '2145653'}
                     ]
         }

    query_tok = "facebook/dpr-question_encoder-single-nq-base"
    query_tokenizer = Tokenizer.load(query_tok, use_fast=True)
    passage_tok = "facebook/dpr-ctx_encoder-single-nq-base"
    passage_tokenizer = Tokenizer.load(passage_tok, use_fast=True)
    processor = TextSimilarityProcessor(query_tokenizer=query_tokenizer,
                                        passage_tokenizer=passage_tokenizer,
                                        max_seq_len_query=256,
                                        max_seq_len_passage=256,
                                        data_dir="data/retriever",
                                        train_filename="nq-train.json",
                                        test_filename="nq-dev.json",
                                        embed_title=True,
                                        num_hard_negatives=1,
                                        label_list=["hard_negative", "positive"],
                                        metric="text_similarity_metric",
                                        shuffle_negatives=False)
    processor.save(save_dir="testsave/dpr_processor")
    dataset, tensor_names, _ = processor.dataset_from_dicts(dicts=[d], return_baskets=False)
    loadedprocessor = TextSimilarityProcessor.load_from_dir(load_dir="testsave/dpr_processor")
    dataset2, tensor_names, _ = loadedprocessor.dataset_from_dicts(dicts=[d], return_baskets=False)
    assert np.array_equal(dataset.tensors[0],dataset2.tensors[0])

@pytest.mark.parametrize("query_and_passage_model", [{"query":"etalab-ia/dpr-question_encoder-fr_qa-camembert","passage":"etalab-ia/dpr-ctx_encoder-fr_qa-camembert"}, {"query":"deepset/gbert-base-germandpr-question_encoder","passage":"deepset/gbert-base-germandpr-ctx_encoder"},{"query":"facebook/dpr-question_encoder-single-nq-base", "passage":"facebook/dpr-ctx_encoder-single-nq-base"}])
def test_dpr_processor_save_load_non_bert_tokenizer(query_and_passage_model):
    """
    This test compares 1) a model that was loaded from model hub with
    2) a model from model hub that was saved to disk and then loaded from disk and
    3) a model in FARM style that was saved to disk and then loaded from disk
    """

    d = {'query': "Comment s'appelle le portail open data du gouvernement?",
         'passages': [
             {'title': 'Etalab',
              'text': "Etalab est une administration publique française qui fait notamment office de Chief Data Officer de l'État et coordonne la conception et la mise en œuvre de sa stratégie dans le domaine de la donnée (ouverture et partage des données publiques ou open data, exploitation des données et intelligence artificielle...). Ainsi, Etalab développe et maintient le portail des données ouvertes du gouvernement français data.gouv.fr. Etalab promeut également une plus grande ouverture l'administration sur la société (gouvernement ouvert) : transparence de l'action publique, innovation ouverte, participation citoyenne... elle promeut l’innovation, l’expérimentation, les méthodes de travail ouvertes, agiles et itératives, ainsi que les synergies avec la société civile pour décloisonner l’administration et favoriser l’adoption des meilleures pratiques professionnelles dans le domaine du numérique. À ce titre elle étudie notamment l’opportunité de recourir à des technologies en voie de maturation issues du monde de la recherche. Cette entité chargée de l'innovation au sein de l'administration doit contribuer à l'amélioration du service public grâce au numérique. Elle est rattachée à la Direction interministérielle du numérique, dont les missions et l’organisation ont été fixées par le décret du 30 octobre 2019.  Dirigé par Laure Lucchesi depuis 2016, elle rassemble une équipe pluridisciplinaire d'une trentaine de personnes.",
              'label': 'positive',
              'external_id': '1'},
         ]
         }

    # load model from model hub
    query_embedding_model = query_and_passage_model["query"]
    passage_embedding_model = query_and_passage_model["passage"]
    query_tokenizer = Tokenizer.load(pretrained_model_name_or_path=query_embedding_model)  # tokenizer class is inferred automatically
    query_encoder = LanguageModel.load(pretrained_model_name_or_path=query_embedding_model,
                                       language_model_class="DPRQuestionEncoder")
    passage_tokenizer = Tokenizer.load(pretrained_model_name_or_path=passage_embedding_model)
    passage_encoder = LanguageModel.load(pretrained_model_name_or_path=passage_embedding_model,
                                         language_model_class="DPRContextEncoder")

    processor = TextSimilarityProcessor(query_tokenizer=query_tokenizer,
                                        passage_tokenizer=passage_tokenizer,
                                        max_seq_len_passage=256,
                                        max_seq_len_query=256,
                                        label_list=["hard_negative", "positive"],
                                        metric="text_similarity_metric",
                                        embed_title=True,
                                        num_hard_negatives=0,
                                        num_positives=1)
    prediction_head = TextSimilarityHead(similarity_function="dot_product")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = BiAdaptiveModel(
        language_model1=query_encoder,
        language_model2=passage_encoder,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm1_output_types=["per_sequence"],
        lm2_output_types=["per_sequence"],
        device=device,
    )
    model.connect_heads_with_processor(processor.tasks, require_labels=False)

    # save model that was loaded from model hub to disk
    save_dir = "testsave/dpr_model"
    query_encoder_dir = "query_encoder"
    passage_encoder_dir = "passage_encoder"
    model.save(Path(save_dir), lm1_name=query_encoder_dir, lm2_name=passage_encoder_dir)
    query_tokenizer.save_pretrained(save_dir + f"/{query_encoder_dir}")
    passage_tokenizer.save_pretrained(save_dir + f"/{passage_encoder_dir}")

    # load model from disk
    loaded_query_tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=Path(save_dir) / query_encoder_dir, use_fast=True)  # tokenizer class is inferred automatically
    loaded_query_encoder = LanguageModel.load(pretrained_model_name_or_path=Path(save_dir) / query_encoder_dir,
                                       language_model_class="DPRQuestionEncoder")
    loaded_passage_tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=Path(save_dir) / passage_encoder_dir, use_fast=True)
    loaded_passage_encoder = LanguageModel.load(pretrained_model_name_or_path=Path(save_dir) / passage_encoder_dir,
                                         language_model_class="DPRContextEncoder")

    loaded_processor = TextSimilarityProcessor(query_tokenizer=loaded_query_tokenizer,
                                        passage_tokenizer=loaded_passage_tokenizer,
                                        max_seq_len_passage=256,
                                        max_seq_len_query=256,
                                        label_list=["hard_negative", "positive"],
                                        metric="text_similarity_metric",
                                        embed_title=True,
                                        num_hard_negatives=0,
                                        num_positives=1)
    loaded_prediction_head = TextSimilarityHead(similarity_function="dot_product")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    loaded_model = BiAdaptiveModel(
        language_model1=loaded_query_encoder,
        language_model2=loaded_passage_encoder,
        prediction_heads=[loaded_prediction_head],
        embeds_dropout_prob=0.1,
        lm1_output_types=["per_sequence"],
        lm2_output_types=["per_sequence"],
        device=device,
    )
    loaded_model.connect_heads_with_processor(loaded_processor.tasks, require_labels=False)

    # compare model loaded from model hub with model loaded from disk
    dataset, tensor_names, _ = processor.dataset_from_dicts(dicts=[d], return_baskets=False)
    dataset2, tensor_names2, _ = loaded_processor.dataset_from_dicts(dicts=[d], return_baskets=False)
    assert np.array_equal(dataset.tensors[0], dataset2.tensors[0])

    # generate embeddings with model loaded from model hub
    dataset, tensor_names, _, baskets = processor.dataset_from_dicts(
        dicts=[d], indices=[i for i in range(len([d]))], return_baskets=True
    )

    data_loader = NamedDataLoader(
        dataset=dataset, sampler=SequentialSampler(dataset), batch_size=16, tensor_names=tensor_names
    )
    all_embeddings = {"query": [], "passages": []}
    model.eval()

    for i, batch in enumerate(tqdm(data_loader, desc=f"Creating Embeddings", unit=" Batches", disable=True)):
        batch = {key: batch[key].to(device) for key in batch}

        # get logits
        with torch.no_grad():
            query_embeddings, passage_embeddings = model.forward(**batch)[0]
            if query_embeddings is not None:
                all_embeddings["query"].append(query_embeddings.cpu().numpy())
            if passage_embeddings is not None:
                all_embeddings["passages"].append(passage_embeddings.cpu().numpy())

    if all_embeddings["passages"]:
        all_embeddings["passages"] = np.concatenate(all_embeddings["passages"])
    if all_embeddings["query"]:
        all_embeddings["query"] = np.concatenate(all_embeddings["query"])

    # generate embeddings with model loaded from disk
    dataset2, tensor_names2, _, baskets2 = loaded_processor.dataset_from_dicts(
        dicts=[d], indices=[i for i in range(len([d]))], return_baskets=True
    )

    data_loader = NamedDataLoader(
        dataset=dataset2, sampler=SequentialSampler(dataset2), batch_size=16, tensor_names=tensor_names2
    )
    all_embeddings2 = {"query": [], "passages": []}
    loaded_model.eval()

    for i, batch in enumerate(tqdm(data_loader, desc=f"Creating Embeddings", unit=" Batches", disable=True)):
        batch = {key: batch[key].to(device) for key in batch}

        # get logits
        with torch.no_grad():
            query_embeddings, passage_embeddings = loaded_model.forward(**batch)[0]
            if query_embeddings is not None:
                all_embeddings2["query"].append(query_embeddings.cpu().numpy())
            if passage_embeddings is not None:
                all_embeddings2["passages"].append(passage_embeddings.cpu().numpy())

    if all_embeddings2["passages"]:
        all_embeddings2["passages"] = np.concatenate(all_embeddings2["passages"])
    if all_embeddings2["query"]:
        all_embeddings2["query"] = np.concatenate(all_embeddings2["query"])

    # compare embeddings of model loaded from model hub and model loaded from disk
    assert np.array_equal(all_embeddings["query"][0], all_embeddings2["query"][0])

    # save the model that was loaded from disk to disk
    save_dir = "testsave/dpr_model"
    query_encoder_dir = "query_encoder"
    passage_encoder_dir = "passage_encoder"
    loaded_model.save(Path(save_dir), lm1_name=query_encoder_dir, lm2_name=passage_encoder_dir)
    loaded_query_tokenizer.save_pretrained(save_dir + f"/{query_encoder_dir}")
    loaded_passage_tokenizer.save_pretrained(save_dir + f"/{passage_encoder_dir}")

    # load model from disk
    query_tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=Path(save_dir) / query_encoder_dir)  # tokenizer class is inferred automatically
    query_encoder = LanguageModel.load(pretrained_model_name_or_path=Path(save_dir) / query_encoder_dir,
                                       language_model_class="DPRQuestionEncoder")
    passage_tokenizer = Tokenizer.load(pretrained_model_name_or_path=Path(save_dir) / passage_encoder_dir)
    passage_encoder = LanguageModel.load(pretrained_model_name_or_path=Path(save_dir) / passage_encoder_dir,
                                         language_model_class="DPRContextEncoder")

    processor = TextSimilarityProcessor(query_tokenizer=query_tokenizer,
                                        passage_tokenizer=passage_tokenizer,
                                        max_seq_len_passage=256,
                                        max_seq_len_query=256,
                                        label_list=["hard_negative", "positive"],
                                        metric="text_similarity_metric",
                                        embed_title=True,
                                        num_hard_negatives=0,
                                        num_positives=1)
    prediction_head = TextSimilarityHead(similarity_function="dot_product")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = BiAdaptiveModel(
        language_model1=query_encoder,
        language_model2=passage_encoder,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm1_output_types=["per_sequence"],
        lm2_output_types=["per_sequence"],
        device=device,
    )
    model.connect_heads_with_processor(processor.tasks, require_labels=False)

    # compare a model loaded from disk that originated from the model hub and was then saved disk with
    # a model loaded from disk that also originated from a FARM style model that was saved to disk
    dataset3, tensor_names3, _ = processor.dataset_from_dicts(dicts=[d], return_baskets=False)
    dataset2, tensor_names2, _ = loaded_processor.dataset_from_dicts(dicts=[d], return_baskets=False)
    assert np.array_equal(dataset3.tensors[0], dataset2.tensors[0])

    # generate embeddings with model loaded from disk that originated from a FARM style model that was saved to disk earlier
    dataset3, tensor_names3, _, baskets3 = loaded_processor.dataset_from_dicts(
        dicts=[d], indices=[i for i in range(len([d]))], return_baskets=True
    )

    data_loader = NamedDataLoader(
        dataset=dataset3, sampler=SequentialSampler(dataset3), batch_size=16, tensor_names=tensor_names3
    )
    all_embeddings3 = {"query": [], "passages": []}
    loaded_model.eval()

    for i, batch in enumerate(tqdm(data_loader, desc=f"Creating Embeddings", unit=" Batches", disable=True)):
        batch = {key: batch[key].to(device) for key in batch}

        # get logits
        with torch.no_grad():
            query_embeddings, passage_embeddings = loaded_model.forward(**batch)[0]
            if query_embeddings is not None:
                all_embeddings3["query"].append(query_embeddings.cpu().numpy())
            if passage_embeddings is not None:
                all_embeddings3["passages"].append(passage_embeddings.cpu().numpy())

    if all_embeddings3["passages"]:
        all_embeddings3["passages"] = np.concatenate(all_embeddings3["passages"])
    if all_embeddings3["query"]:
        all_embeddings3["query"] = np.concatenate(all_embeddings3["query"])

    # compare embeddings of model loaded from model hub and model loaded from disk that originated from a FARM style
    # model that was saved to disk earlier
    assert np.array_equal(all_embeddings["query"][0], all_embeddings3["query"][0])


#TODO fix CI errors (test pass locally or on AWS, next steps: isolate PyTorch versions once FARM dependency is removed)
# def test_dpr_training():
#     batch_size = 1
#     n_epochs = 1
#     distributed = False  # enable for multi GPU training via DDP
#     evaluate_every = 1
#     question_lang_model = "microsoft/MiniLM-L12-H384-uncased"
#     passage_lang_model = "microsoft/MiniLM-L12-H384-uncased"
#     do_lower_case = True
#     use_fast = True
#     similarity_function = "dot_product"
#
#
#
#     device, n_gpu = initialize_device_settings(use_cuda=False)
#
#     query_tokenizer = Tokenizer.load(pretrained_model_name_or_path=question_lang_model,
#                                      do_lower_case=do_lower_case, use_fast=use_fast)
#     passage_tokenizer = Tokenizer.load(pretrained_model_name_or_path=passage_lang_model,
#                                        do_lower_case=do_lower_case, use_fast=use_fast)
#     label_list = ["hard_negative", "positive"]
#
#     processor = TextSimilarityProcessor(query_tokenizer=query_tokenizer,
#                                         passage_tokenizer=passage_tokenizer,
#                                         max_seq_len_query=10,
#                                         max_seq_len_passage=10,
#                                         label_list=label_list,
#                                         metric="text_similarity_metric",
#                                         data_dir="samples/dpr/",
#                                         train_filename="sample.json",
#                                         dev_filename="sample.json",
#                                         test_filename=None,
#                                         embed_title=True,
#                                         num_hard_negatives=1,
#                                         dev_split=0,
#                                         max_samples=2)
#
#     data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False)
#
#     question_language_model = LanguageModel.load(pretrained_model_name_or_path=question_lang_model,
#                                                  language_model_class="DPRQuestionEncoder")
#     passage_language_model = LanguageModel.load(pretrained_model_name_or_path=passage_lang_model,
#                                                 language_model_class="DPRContextEncoder")
#
#     prediction_head = TextSimilarityHead(similarity_function=similarity_function)
#
#     model = BiAdaptiveModel(
#         language_model1=question_language_model,
#         language_model2=passage_language_model,
#         prediction_heads=[prediction_head],
#         embeds_dropout_prob=0.1,
#         lm1_output_types=["per_sequence"],
#         lm2_output_types=["per_sequence"],
#         device=device,
#     )
#
#     model, optimizer, lr_schedule = initialize_optimizer(
#         model=model,
#         learning_rate=1e-5,
#         optimizer_opts={"name": "TransformersAdamW", "correct_bias": True, "weight_decay": 0.0, \
#                         "eps": 1e-08},
#         schedule_opts={"name": "LinearWarmup", "num_warmup_steps": 100},
#         n_batches=len(data_silo.loaders["train"]),
#         n_epochs=n_epochs,
#         grad_acc_steps=1,
#         device=device,
#         distributed=distributed
#     )
#
#     trainer = Trainer(
#         model=model,
#         optimizer=optimizer,
#         data_silo=data_silo,
#         epochs=n_epochs,
#         n_gpu=n_gpu,
#         lr_schedule=lr_schedule,
#         evaluate_every=evaluate_every,
#         device=device,
#     )
#
#     trainer.train()
#
#     ######## save and load model again
#     save_dir = Path("testsave/dpr-model")
#     model.save(save_dir)
#     del model
#
#     model2 = BiAdaptiveModel.load(save_dir, device=device)
#     model2, optimizer2, lr_schedule = initialize_optimizer(
#         model=model2,
#         learning_rate=1e-5,
#         optimizer_opts={"name": "TransformersAdamW", "correct_bias": True, "weight_decay": 0.0, \
#                         "eps": 1e-08},
#         schedule_opts={"name": "LinearWarmup", "num_warmup_steps": 100},
#         n_batches=len(data_silo.loaders["train"]),
#         n_epochs=n_epochs,
#         grad_acc_steps=1,
#         device=device,
#         distributed=distributed
#     )
#     trainer2 = Trainer(
#         model=model2,
#         optimizer=optimizer,
#         data_silo=data_silo,
#         epochs=n_epochs,
#         n_gpu=n_gpu,
#         lr_schedule=lr_schedule,
#         evaluate_every=evaluate_every,
#         device=device,
#     )
#
#     trainer2.train()


if __name__=="__main__":
    # test_dpr_training()
    test_dpr_context_only()
    # test_dpr_modules()

# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                      |    Stmts |     Miss |   Cover |   Missing |
|-------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| haystack/\_\_init\_\_.py                                                  |       11 |        0 |    100% |           |
| haystack/components/\_\_init\_\_.py                                       |        0 |        0 |    100% |           |
| haystack/components/agents/\_\_init\_\_.py                                |        5 |        0 |    100% |           |
| haystack/components/agents/agent.py                                       |      397 |        2 |     99% |  208, 809 |
| haystack/components/agents/state/\_\_init\_\_.py                          |        5 |        0 |    100% |           |
| haystack/components/agents/state/state.py                                 |       77 |        2 |     97% |    76, 80 |
| haystack/components/agents/state/state\_utils.py                          |       18 |        0 |    100% |           |
| haystack/components/agents/tool\_calling.py                               |      250 |        5 |     98% |30, 257-259, 635 |
| haystack/components/agents/utils.py                                       |      120 |        3 |     98% |80, 126, 297 |
| haystack/components/builders/\_\_init\_\_.py                              |        5 |        0 |    100% |           |
| haystack/components/builders/answer\_builder.py                           |       98 |        3 |     97% |270, 291, 303 |
| haystack/components/builders/chat\_prompt\_builder.py                     |      114 |        2 |     98% |  170, 258 |
| haystack/components/builders/prompt\_builder.py                           |       50 |        0 |    100% |           |
| haystack/components/caching/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| haystack/components/caching/cache\_checker.py                             |       43 |        0 |    100% |           |
| haystack/components/converters/\_\_init\_\_.py                            |        5 |        0 |    100% |           |
| haystack/components/converters/csv.py                                     |       87 |        4 |     95% |162-163, 190-191 |
| haystack/components/converters/docx.py                                    |      172 |        1 |     99% |       254 |
| haystack/components/converters/file\_to\_file\_content.py                 |       32 |        0 |    100% |           |
| haystack/components/converters/html.py                                    |       54 |        0 |    100% |           |
| haystack/components/converters/image/\_\_init\_\_.py                      |        5 |        0 |    100% |           |
| haystack/components/converters/image/document\_to\_image.py               |       47 |        0 |    100% |           |
| haystack/components/converters/image/file\_to\_document.py                |       27 |        1 |     96% |        94 |
| haystack/components/converters/image/file\_to\_image.py                   |       53 |        0 |    100% |           |
| haystack/components/converters/image/image\_utils.py                      |      129 |        5 |     96% |78-82, 104, 109-110 |
| haystack/components/converters/image/pdf\_to\_image.py                    |       46 |        3 |     93% |   137-141 |
| haystack/components/converters/json.py                                    |       93 |       11 |     88% |228-229, 231-234, 238-239, 254-255, 286-288 |
| haystack/components/converters/markdown.py                                |       71 |       10 |     86% |107, 120-122, 128-134, 152, 173-178 |
| haystack/components/converters/msg.py                                     |       77 |        5 |     94% |91, 165-167, 179 |
| haystack/components/converters/multi\_file\_converter.py                  |       50 |        0 |    100% |           |
| haystack/components/converters/output\_adapter.py                         |       71 |        1 |     99% |       133 |
| haystack/components/converters/pdfminer.py                                |      105 |        0 |    100% |           |
| haystack/components/converters/pptx.py                                    |       69 |        2 |     97% |     89-90 |
| haystack/components/converters/pypdf.py                                   |      103 |        6 |     94% |192, 209-211, 267-271 |
| haystack/components/converters/txt.py                                     |       34 |        3 |     91% |     88-92 |
| haystack/components/converters/utils.py                                   |       37 |        0 |    100% |           |
| haystack/components/converters/xlsx.py                                    |      119 |        3 |     97% |83, 183, 185 |
| haystack/components/embedders/\_\_init\_\_.py                             |        5 |        0 |    100% |           |
| haystack/components/embedders/azure\_document\_embedder.py                |       69 |        2 |     97% |  121, 124 |
| haystack/components/embedders/azure\_text\_embedder.py                    |       63 |        2 |     97% |  107, 110 |
| haystack/components/embedders/mock\_document\_embedder.py                 |       58 |        0 |    100% |           |
| haystack/components/embedders/mock\_text\_embedder.py                     |       50 |        0 |    100% |           |
| haystack/components/embedders/mock\_utils.py                              |       25 |        0 |    100% |           |
| haystack/components/embedders/openai\_document\_embedder.py               |      145 |       26 |     82% |187, 225, 258, 280-281, 303, 309-315, 325-326, 377-398 |
| haystack/components/embedders/openai\_text\_embedder.py                   |       73 |        9 |     88% |162, 229-231, 249-253 |
| haystack/components/embedders/types/\_\_init\_\_.py                       |        2 |        0 |    100% |           |
| haystack/components/embedders/types/protocol.py                           |        6 |        0 |    100% |           |
| haystack/components/evaluators/\_\_init\_\_.py                            |        5 |        0 |    100% |           |
| haystack/components/evaluators/answer\_exact\_match.py                    |       15 |        0 |    100% |           |
| haystack/components/evaluators/context\_relevance.py                      |       50 |        0 |    100% |           |
| haystack/components/evaluators/document\_map.py                           |       52 |        4 |     92% |73, 76-80, 134 |
| haystack/components/evaluators/document\_mrr.py                           |       42 |        4 |     90% |71, 74-78, 122 |
| haystack/components/evaluators/document\_ndcg.py                          |       70 |        0 |    100% |           |
| haystack/components/evaluators/document\_recall.py                        |       73 |        2 |     97% |   109-113 |
| haystack/components/evaluators/faithfulness.py                            |       50 |        0 |    100% |           |
| haystack/components/evaluators/llm\_evaluator.py                          |      160 |        1 |     99% |       241 |
| haystack/components/evaluators/sas\_evaluator.py                          |       57 |       25 |     56% |111-125, 153-188 |
| haystack/components/extractors/\_\_init\_\_.py                            |        5 |        0 |    100% |           |
| haystack/components/extractors/image/\_\_init\_\_.py                      |        5 |        0 |    100% |           |
| haystack/components/extractors/image/llm\_document\_content\_extractor.py |      152 |        0 |    100% |           |
| haystack/components/extractors/llm\_metadata\_extractor.py                |      166 |       12 |     93% |315-323, 338-346, 407, 469 |
| haystack/components/extractors/regex\_text\_extractor.py                  |       49 |        0 |    100% |           |
| haystack/components/fetchers/\_\_init\_\_.py                              |        5 |        0 |    100% |           |
| haystack/components/fetchers/link\_content.py                             |      202 |       18 |     91% |212-220, 288, 337-341, 426-430, 471, 504 |
| haystack/components/generators/\_\_init\_\_.py                            |        5 |        0 |    100% |           |
| haystack/components/generators/chat/\_\_init\_\_.py                       |        5 |        0 |    100% |           |
| haystack/components/generators/chat/azure.py                              |       85 |        3 |     96% |212, 362, 365 |
| haystack/components/generators/chat/azure\_responses.py                   |       46 |        1 |     98% |       274 |
| haystack/components/generators/chat/fallback.py                           |      102 |        3 |     97% |   258-263 |
| haystack/components/generators/chat/llm.py                                |       40 |        1 |     98% |       124 |
| haystack/components/generators/chat/mock.py                               |      154 |        0 |    100% |           |
| haystack/components/generators/chat/openai.py                             |      227 |        4 |     98% |460, 511, 683, 783 |
| haystack/components/generators/chat/openai\_responses.py                  |      365 |       28 |     92% |297, 421, 502, 518, 555, 580, 633-642, 658-659, 664, 684-693, 703, 850, 892-893, 916, 968, 993, 1012 |
| haystack/components/generators/chat/types/\_\_init\_\_.py                 |        2 |        0 |    100% |           |
| haystack/components/generators/chat/types/protocol.py                     |        4 |        0 |    100% |           |
| haystack/components/generators/openai\_image\_generator.py                |       80 |        0 |    100% |           |
| haystack/components/generators/utils.py                                   |       95 |        9 |     91% |69, 83, 129, 162-163, 198, 214, 216, 218 |
| haystack/components/joiners/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| haystack/components/joiners/answer\_joiner.py                             |       57 |        1 |     98% |       164 |
| haystack/components/joiners/branch.py                                     |       21 |        0 |    100% |           |
| haystack/components/joiners/document\_joiner.py                           |      104 |        0 |    100% |           |
| haystack/components/joiners/list\_joiner.py                               |       23 |        0 |    100% |           |
| haystack/components/joiners/string\_joiner.py                             |        8 |        0 |    100% |           |
| haystack/components/preprocessors/\_\_init\_\_.py                         |        5 |        0 |    100% |           |
| haystack/components/preprocessors/\_page\_numbers.py                      |        9 |        1 |     89% |        29 |
| haystack/components/preprocessors/csv\_document\_cleaner.py               |       64 |        0 |    100% |           |
| haystack/components/preprocessors/csv\_document\_splitter.py              |      108 |        4 |     96% |128, 131, 142-146 |
| haystack/components/preprocessors/document\_cleaner.py                    |      125 |        1 |     99% |       107 |
| haystack/components/preprocessors/document\_preprocessor.py               |       47 |        2 |     96% |  168, 204 |
| haystack/components/preprocessors/document\_splitter.py                   |      280 |        2 |     99% |  213, 239 |
| haystack/components/preprocessors/embedding\_based\_document\_splitter.py |      230 |       12 |     95% |300-301, 432, 484-496, 514, 531, 600-601 |
| haystack/components/preprocessors/hierarchical\_document\_splitter.py     |       60 |        0 |    100% |           |
| haystack/components/preprocessors/markdown\_header\_splitter.py           |      173 |        6 |     97% |279-280, 295-296, 355, 418 |
| haystack/components/preprocessors/python\_code\_splitter.py               |      291 |       15 |     95% |151, 163, 191, 198, 232, 281, 309-311, 412-414, 416, 529, 597 |
| haystack/components/preprocessors/recursive\_splitter.py                  |      266 |       20 |     92% |164-167, 208-210, 249-251, 264-266, 299, 338, 472, 478-481 |
| haystack/components/preprocessors/sentence\_tokenizer.py                  |       87 |        4 |     95% |65-66, 73-78 |
| haystack/components/preprocessors/text\_cleaner.py                        |       33 |        0 |    100% |           |
| haystack/components/query/\_\_init\_\_.py                                 |        5 |        0 |    100% |           |
| haystack/components/query/query\_expander.py                              |      142 |       10 |     93% |282-283, 289, 299-304, 311-317, 328-336 |
| haystack/components/rankers/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| haystack/components/rankers/llm\_ranker.py                                |      154 |        5 |     97% |342, 376, 404, 413, 417 |
| haystack/components/rankers/lost\_in\_the\_middle.py                      |       43 |        2 |     95% |   83, 117 |
| haystack/components/rankers/meta\_field.py                                |      119 |        0 |    100% |           |
| haystack/components/rankers/meta\_field\_grouping\_ranker.py              |       39 |        0 |    100% |           |
| haystack/components/retrievers/\_\_init\_\_.py                            |        5 |        0 |    100% |           |
| haystack/components/retrievers/auto\_merging\_retriever.py                |       86 |        0 |    100% |           |
| haystack/components/retrievers/filter\_retriever.py                       |       30 |        1 |     97% |        55 |
| haystack/components/retrievers/in\_memory/\_\_init\_\_.py                 |        5 |        0 |    100% |           |
| haystack/components/retrievers/in\_memory/bm25\_retriever.py              |       48 |        0 |    100% |           |
| haystack/components/retrievers/in\_memory/embedding\_retriever.py         |       53 |        1 |     98% |       102 |
| haystack/components/retrievers/multi\_query\_embedding\_retriever.py      |       82 |        2 |     98% |  202, 221 |
| haystack/components/retrievers/multi\_query\_text\_retriever.py           |       72 |        3 |     96% |124, 178, 193 |
| haystack/components/retrievers/multi\_retriever.py                        |      131 |        1 |     99% |       160 |
| haystack/components/retrievers/sentence\_window\_retriever.py             |      112 |        1 |     99% |       138 |
| haystack/components/retrievers/text\_embedding\_retriever.py              |       52 |        0 |    100% |           |
| haystack/components/retrievers/types/\_\_init\_\_.py                      |        2 |        0 |    100% |           |
| haystack/components/retrievers/types/protocol.py                          |        5 |        0 |    100% |           |
| haystack/components/routers/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| haystack/components/routers/conditional\_router.py                        |      179 |        8 |     96% |484-485, 521, 539, 581, 599, 609, 621 |
| haystack/components/routers/document\_length\_router.py                   |       15 |        0 |    100% |           |
| haystack/components/routers/document\_type\_router.py                     |       48 |        0 |    100% |           |
| haystack/components/routers/file\_type\_router.py                         |       78 |        6 |     92% |   197-202 |
| haystack/components/routers/llm\_messages\_router.py                      |       92 |        1 |     99% |       204 |
| haystack/components/routers/metadata\_router.py                           |       38 |        0 |    100% |           |
| haystack/components/samplers/\_\_init\_\_.py                              |        5 |        0 |    100% |           |
| haystack/components/samplers/top\_p.py                                    |       67 |        0 |    100% |           |
| haystack/components/validators/\_\_init\_\_.py                            |        5 |        0 |    100% |           |
| haystack/components/validators/json\_schema.py                            |       73 |        7 |     90% |23-24, 142, 144, 159, 247, 253 |
| haystack/components/writers/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| haystack/components/writers/document\_writer.py                           |       38 |        0 |    100% |           |
| haystack/core/\_\_init\_\_.py                                             |        2 |        0 |    100% |           |
| haystack/core/component/\_\_init\_\_.py                                   |        3 |        0 |    100% |           |
| haystack/core/component/component.py                                      |      179 |        1 |     99% |       346 |
| haystack/core/component/sockets.py                                        |       41 |        6 |     85% |82, 117-124, 129-130 |
| haystack/core/component/types.py                                          |       42 |        3 |     93% |     88-90 |
| haystack/core/errors.py                                                   |       72 |       10 |     86% |60-67, 127, 142-144, 151-153, 166 |
| haystack/core/pipeline/\_\_init\_\_.py                                    |        2 |        0 |    100% |           |
| haystack/core/pipeline/base.py                                            |      640 |       51 |     92% |270-271, 275, 490, 710, 747-757, 947, 1015, 1598, 1765, 1769, 1779-1785, 1800-1862 |
| haystack/core/pipeline/breakpoint.py                                      |      109 |       11 |     90% |52, 71, 90, 98, 125-130, 197 |
| haystack/core/pipeline/component\_checks.py                               |       66 |        0 |    100% |           |
| haystack/core/pipeline/descriptions.py                                    |        6 |        0 |    100% |           |
| haystack/core/pipeline/draw.py                                            |      199 |       41 |     79% |35-58, 142, 145, 148, 150, 158, 162-169, 296, 306, 378-380, 385, 393-394, 422-427 |
| haystack/core/pipeline/pipeline.py                                        |      331 |       19 |     94% |174, 181, 426, 569, 604, 772, 967, 972, 989-993, 997-1009, 1050-1059, 1084 |
| haystack/core/pipeline/utils.py                                           |       71 |        1 |     99% |       206 |
| haystack/core/serialization.py                                            |      129 |        6 |     95% |76, 100, 247, 334, 350-351 |
| haystack/core/serialization\_security.py                                  |      135 |        0 |    100% |           |
| haystack/core/super\_component/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| haystack/core/super\_component/super\_component.py                        |      195 |        7 |     96% |67, 199, 212, 250, 304, 346, 599 |
| haystack/core/super\_component/utils.py                                   |       95 |        4 |     96% |77, 112, 124, 181 |
| haystack/core/type\_utils.py                                              |      173 |        6 |     97% |95, 97, 102, 167, 211, 306 |
| haystack/dataclasses/\_\_init\_\_.py                                      |        5 |        0 |    100% |           |
| haystack/dataclasses/answer.py                                            |       57 |        0 |    100% |           |
| haystack/dataclasses/breakpoints.py                                       |       39 |        0 |    100% |           |
| haystack/dataclasses/byte\_stream.py                                      |       39 |        0 |    100% |           |
| haystack/dataclasses/chat\_message.py                                     |      420 |        4 |     99% |160, 401, 417, 639 |
| haystack/dataclasses/document.py                                          |      104 |        4 |     96% |79, 81, 83, 85 |
| haystack/dataclasses/file\_content.py                                     |       68 |        0 |    100% |           |
| haystack/dataclasses/image\_content.py                                    |       85 |        2 |     98% |   100-104 |
| haystack/dataclasses/skill\_info.py                                       |        3 |        0 |    100% |           |
| haystack/dataclasses/sparse\_embedding.py                                 |       14 |        0 |    100% |           |
| haystack/dataclasses/streaming\_chunk.py                                  |       81 |        1 |     99% |       182 |
| haystack/document\_stores/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| haystack/document\_stores/errors/\_\_init\_\_.py                          |        2 |        0 |    100% |           |
| haystack/document\_stores/errors/errors.py                                |        6 |        0 |    100% |           |
| haystack/document\_stores/in\_memory/\_\_init\_\_.py                      |        5 |        0 |    100% |           |
| haystack/document\_stores/in\_memory/document\_store.py                   |      446 |       16 |     96% |66, 468-469, 478, 584, 665, 716, 718, 745-746, 802, 944, 946, 964, 969-970 |
| haystack/document\_stores/types/\_\_init\_\_.py                           |        4 |        0 |    100% |           |
| haystack/document\_stores/types/filter\_policy.py                         |       65 |       10 |     85% |25, 38-39, 190, 198-204, 250-255, 259-265 |
| haystack/document\_stores/types/policy.py                                 |        6 |        0 |    100% |           |
| haystack/document\_stores/types/protocol.py                               |       11 |        0 |    100% |           |
| haystack/errors.py                                                        |        2 |        0 |    100% |           |
| haystack/evaluation/\_\_init\_\_.py                                       |        5 |        0 |    100% |           |
| haystack/evaluation/eval\_run\_result.py                                  |       93 |       17 |     82% |75, 92-97, 112-113, 117, 120, 189, 192, 195, 200, 211, 215 |
| haystack/hooks/\_\_init\_\_.py                                            |        5 |        0 |    100% |           |
| haystack/hooks/budget/\_\_init\_\_.py                                     |        5 |        0 |    100% |           |
| haystack/hooks/budget/hooks.py                                            |       32 |        0 |    100% |           |
| haystack/hooks/compaction/\_\_init\_\_.py                                 |        5 |        0 |    100% |           |
| haystack/hooks/compaction/hooks.py                                        |      111 |        1 |     99% |       296 |
| haystack/hooks/compaction/sliding\_window.py                              |       67 |        0 |    100% |           |
| haystack/hooks/compaction/summarization.py                                |      171 |       16 |     91% |89, 314-317, 475-479, 483-484, 488-492 |
| haystack/hooks/compaction/tool\_result\_pruning.py                        |       55 |        0 |    100% |           |
| haystack/hooks/compaction/types/\_\_init\_\_.py                           |        2 |        0 |    100% |           |
| haystack/hooks/compaction/types/protocol.py                               |       12 |        0 |    100% |           |
| haystack/hooks/compaction/utils.py                                        |       49 |        0 |    100% |           |
| haystack/hooks/from\_function.py                                          |       45 |        0 |    100% |           |
| haystack/hooks/human\_in\_the\_loop/\_\_init\_\_.py                       |        5 |        0 |    100% |           |
| haystack/hooks/human\_in\_the\_loop/dataclasses.py                        |       16 |        0 |    100% |           |
| haystack/hooks/human\_in\_the\_loop/hooks.py                              |       30 |        1 |     97% |       117 |
| haystack/hooks/human\_in\_the\_loop/policies.py                           |       17 |        0 |    100% |           |
| haystack/hooks/human\_in\_the\_loop/strategies.py                         |      188 |        5 |     97% |338, 386, 430, 499, 693 |
| haystack/hooks/human\_in\_the\_loop/types/\_\_init\_\_.py                 |        2 |        0 |    100% |           |
| haystack/hooks/human\_in\_the\_loop/types/protocol.py                     |       25 |        0 |    100% |           |
| haystack/hooks/human\_in\_the\_loop/user\_interfaces.py                   |       96 |        2 |     98% |   65, 103 |
| haystack/hooks/invocation.py                                              |       40 |        0 |    100% |           |
| haystack/hooks/protocol.py                                                |       15 |        0 |    100% |           |
| haystack/hooks/tool\_result\_offloading/\_\_init\_\_.py                   |        5 |        0 |    100% |           |
| haystack/hooks/tool\_result\_offloading/hooks.py                          |      121 |        1 |     99% |       205 |
| haystack/hooks/tool\_result\_offloading/policies.py                       |       17 |        0 |    100% |           |
| haystack/hooks/tool\_result\_offloading/stores.py                         |       34 |        0 |    100% |           |
| haystack/hooks/tool\_result\_offloading/types/\_\_init\_\_.py             |        2 |        0 |    100% |           |
| haystack/hooks/tool\_result\_offloading/types/protocol.py                 |       19 |        2 |     89% |    56, 61 |
| haystack/hooks/utils.py                                                   |       47 |        0 |    100% |           |
| haystack/lazy\_imports.py                                                 |       13 |        0 |    100% |           |
| haystack/logging.py                                                       |      142 |        4 |     97% |251-252, 383-385 |
| haystack/marshal/\_\_init\_\_.py                                          |        5 |        0 |    100% |           |
| haystack/marshal/protocol.py                                              |        4 |        0 |    100% |           |
| haystack/marshal/yaml.py                                                  |       21 |        1 |     95% |        42 |
| haystack/skill\_stores/\_\_init\_\_.py                                    |        0 |        0 |    100% |           |
| haystack/skill\_stores/file\_system/\_\_init\_\_.py                       |        5 |        0 |    100% |           |
| haystack/skill\_stores/file\_system/skill\_store.py                       |      105 |        2 |     98% |  102, 136 |
| haystack/skill\_stores/types/\_\_init\_\_.py                              |        2 |        0 |    100% |           |
| haystack/skill\_stores/types/protocol.py                                  |       11 |        0 |    100% |           |
| haystack/telemetry/\_\_init\_\_.py                                        |        1 |        0 |    100% |           |
| haystack/telemetry/\_environment.py                                       |       25 |        0 |    100% |           |
| haystack/telemetry/\_telemetry.py                                         |       83 |       12 |     86% |70-76, 94-95, 114-115, 188 |
| haystack/token\_counters/\_\_init\_\_.py                                  |        5 |        0 |    100% |           |
| haystack/token\_counters/approximate\_counter.py                          |       21 |        0 |    100% |           |
| haystack/token\_counters/openai\_counter.py                               |       48 |        1 |     98% |       107 |
| haystack/token\_counters/tiktoken\_counter.py                             |       28 |        0 |    100% |           |
| haystack/token\_counters/types/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| haystack/token\_counters/types/protocol.py                                |       10 |        0 |    100% |           |
| haystack/token\_counters/utils.py                                         |       49 |        0 |    100% |           |
| haystack/tools/\_\_init\_\_.py                                            |        7 |        0 |    100% |           |
| haystack/tools/agent\_tool.py                                             |       51 |        0 |    100% |           |
| haystack/tools/component\_tool.py                                         |      110 |        5 |     95% |381-382, 425-427 |
| haystack/tools/errors.py                                                  |        6 |        0 |    100% |           |
| haystack/tools/from\_function.py                                          |       64 |        0 |    100% |           |
| haystack/tools/parameters\_schema\_utils.py                               |       97 |        3 |     97% |95, 135-136 |
| haystack/tools/pipeline\_tool.py                                          |       30 |        2 |     93% |  240, 243 |
| haystack/tools/searchable\_toolset.py                                     |      124 |        0 |    100% |           |
| haystack/tools/serde\_utils.py                                            |       43 |        3 |     93% |33, 35, 57 |
| haystack/tools/skills/\_\_init\_\_.py                                     |        2 |        0 |    100% |           |
| haystack/tools/skills/skill\_toolset.py                                   |       64 |        0 |    100% |           |
| haystack/tools/tool.py                                                    |      154 |        5 |     97% |175, 198, 242-243, 374 |
| haystack/tools/tool\_types.py                                             |        5 |        0 |    100% |           |
| haystack/tools/toolset.py                                                 |       46 |        3 |     93% |101, 157, 211 |
| haystack/tools/utils.py                                                   |       53 |        0 |    100% |           |
| haystack/tracing/\_\_init\_\_.py                                          |        1 |        0 |    100% |           |
| haystack/tracing/logging\_tracer.py                                       |       36 |        0 |    100% |           |
| haystack/tracing/tracer.py                                                |       54 |        4 |     93% |28, 74, 94, 103 |
| haystack/tracing/utils.py                                                 |       26 |        0 |    100% |           |
| haystack/utils/\_\_init\_\_.py                                            |        5 |        0 |    100% |           |
| haystack/utils/async\_utils.py                                            |       19 |        0 |    100% |           |
| haystack/utils/auth.py                                                    |      105 |        6 |     94% |22, 116, 124, 128, 133, 161 |
| haystack/utils/azure.py                                                   |        6 |        2 |     67% |     15-16 |
| haystack/utils/base\_serialization.py                                     |      136 |        9 |     93% |186-189, 210, 221, 331-335 |
| haystack/utils/callable\_serialization.py                                 |       66 |        8 |     88% |56, 60, 117-119, 129, 140, 171 |
| haystack/utils/dataclasses.py                                             |       22 |        0 |    100% |           |
| haystack/utils/deserialization.py                                         |       18 |        1 |     94% |        54 |
| haystack/utils/device.py                                                  |      216 |       17 |     92% |237-239, 325-327, 364, 403, 456, 477, 493, 521-524, 548-549 |
| haystack/utils/experimental.py                                            |       14 |        0 |    100% |           |
| haystack/utils/filters.py                                                 |      158 |        0 |    100% |           |
| haystack/utils/hf.py                                                      |       62 |       15 |     76% |23-31, 40-51 |
| haystack/utils/http\_client.py                                            |       14 |        0 |    100% |           |
| haystack/utils/jinja2\_chat\_extension.py                                 |      146 |        1 |     99% |       405 |
| haystack/utils/jinja2\_extensions.py                                      |       47 |        0 |    100% |           |
| haystack/utils/jinja2\_sandbox.py                                         |       16 |        0 |    100% |           |
| haystack/utils/jupyter.py                                                 |        9 |        3 |     67% |     15-17 |
| haystack/utils/misc.py                                                    |       99 |        5 |     95% |52-53, 132-135 |
| haystack/utils/requests\_utils.py                                         |       33 |        0 |    100% |           |
| haystack/utils/type\_serialization.py                                     |      144 |        6 |     96% |106, 108, 260-261, 269, 285 |
| haystack/utils/url\_validation.py                                         |        4 |        0 |    100% |           |
| haystack/version.py                                                       |        5 |        2 |     60% |      9-10 |
| **TOTAL**                                                                 | **17569** |  **710** | **96%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.
# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                      |    Stmts |     Miss |   Cover |   Missing |
|-------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| haystack/\_\_init\_\_.py                                                  |       11 |        0 |    100% |           |
| haystack/components/\_\_init\_\_.py                                       |        0 |        0 |    100% |           |
| haystack/components/agents/\_\_init\_\_.py                                |        5 |        0 |    100% |           |
| haystack/components/agents/agent.py                                       |      448 |        6 |     99% |141, 209, 225, 349, 925, 1258 |
| haystack/components/agents/state/\_\_init\_\_.py                          |        5 |        0 |    100% |           |
| haystack/components/agents/state/state.py                                 |       73 |        2 |     97% |    75, 79 |
| haystack/components/agents/state/state\_utils.py                          |       18 |        0 |    100% |           |
| haystack/components/agents/tool\_calling.py                               |      248 |        5 |     98% |30, 247-249, 618 |
| haystack/components/builders/\_\_init\_\_.py                              |        5 |        0 |    100% |           |
| haystack/components/builders/answer\_builder.py                           |       99 |        4 |     96% |269, 277, 290, 302 |
| haystack/components/builders/chat\_prompt\_builder.py                     |      110 |        2 |     98% |  179, 266 |
| haystack/components/builders/prompt\_builder.py                           |       50 |        0 |    100% |           |
| haystack/components/caching/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| haystack/components/caching/cache\_checker.py                             |       37 |        0 |    100% |           |
| haystack/components/converters/\_\_init\_\_.py                            |        5 |        0 |    100% |           |
| haystack/components/converters/csv.py                                     |       88 |        4 |     95% |161-162, 189-190 |
| haystack/components/converters/docx.py                                    |      159 |        1 |     99% |       262 |
| haystack/components/converters/file\_to\_file\_content.py                 |       32 |        0 |    100% |           |
| haystack/components/converters/html.py                                    |       50 |        0 |    100% |           |
| haystack/components/converters/image/\_\_init\_\_.py                      |        5 |        0 |    100% |           |
| haystack/components/converters/image/document\_to\_image.py               |       43 |        0 |    100% |           |
| haystack/components/converters/image/file\_to\_document.py                |       27 |        1 |     96% |        94 |
| haystack/components/converters/image/file\_to\_image.py                   |       53 |        0 |    100% |           |
| haystack/components/converters/image/image\_utils.py                      |      128 |        5 |     96% |78-82, 104, 109-110 |
| haystack/components/converters/image/pdf\_to\_image.py                    |       46 |        3 |     93% |   137-141 |
| haystack/components/converters/json.py                                    |       87 |       11 |     87% |220-221, 223-226, 230-231, 243-244, 275-277 |
| haystack/components/converters/markdown.py                                |       71 |       10 |     86% |107, 120-122, 128-134, 152, 173-178 |
| haystack/components/converters/msg.py                                     |       75 |        5 |     93% |91, 165-167, 179 |
| haystack/components/converters/multi\_file\_converter.py                  |       50 |        0 |    100% |           |
| haystack/components/converters/output\_adapter.py                         |       65 |        1 |     98% |       129 |
| haystack/components/converters/pdfminer.py                                |       68 |        0 |    100% |           |
| haystack/components/converters/pptx.py                                    |       69 |        2 |     97% |     89-90 |
| haystack/components/converters/pypdf.py                                   |       74 |        3 |     96% |   209-213 |
| haystack/components/converters/txt.py                                     |       34 |        3 |     91% |     87-91 |
| haystack/components/converters/utils.py                                   |       21 |        0 |    100% |           |
| haystack/components/converters/xlsx.py                                    |      113 |        3 |     97% |83, 183, 185 |
| haystack/components/embedders/\_\_init\_\_.py                             |        5 |        0 |    100% |           |
| haystack/components/embedders/azure\_document\_embedder.py                |       67 |        2 |     97% |  121, 124 |
| haystack/components/embedders/azure\_text\_embedder.py                    |       61 |        2 |     97% |  107, 110 |
| haystack/components/embedders/mock\_document\_embedder.py                 |       58 |        0 |    100% |           |
| haystack/components/embedders/mock\_text\_embedder.py                     |       50 |        0 |    100% |           |
| haystack/components/embedders/mock\_utils.py                              |       25 |        0 |    100% |           |
| haystack/components/embedders/openai\_document\_embedder.py               |      143 |       44 |     69% |179, 217, 250, 272-273, 284-320, 369-390 |
| haystack/components/embedders/openai\_text\_embedder.py                   |       71 |        9 |     87% |154, 221-223, 241-245 |
| haystack/components/embedders/types/\_\_init\_\_.py                       |        2 |        0 |    100% |           |
| haystack/components/embedders/types/protocol.py                           |        6 |        0 |    100% |           |
| haystack/components/evaluators/\_\_init\_\_.py                            |        5 |        0 |    100% |           |
| haystack/components/evaluators/answer\_exact\_match.py                    |       15 |        0 |    100% |           |
| haystack/components/evaluators/context\_relevance.py                      |       49 |        0 |    100% |           |
| haystack/components/evaluators/document\_map.py                           |       46 |        4 |     91% |73, 76-80, 126 |
| haystack/components/evaluators/document\_mrr.py                           |       42 |        4 |     90% |71, 74-78, 122 |
| haystack/components/evaluators/document\_ndcg.py                          |       66 |        0 |    100% |           |
| haystack/components/evaluators/document\_recall.py                        |       73 |        2 |     97% |   109-113 |
| haystack/components/evaluators/faithfulness.py                            |       49 |        0 |    100% |           |
| haystack/components/evaluators/llm\_evaluator.py                          |      155 |        5 |     97% |238, 254, 325-326, 331 |
| haystack/components/evaluators/sas\_evaluator.py                          |       57 |       25 |     56% |111-125, 153-188 |
| haystack/components/extractors/\_\_init\_\_.py                            |        5 |        0 |    100% |           |
| haystack/components/extractors/image/\_\_init\_\_.py                      |        5 |        0 |    100% |           |
| haystack/components/extractors/image/llm\_document\_content\_extractor.py |      147 |        1 |     99% |       307 |
| haystack/components/extractors/llm\_metadata\_extractor.py                |      165 |       19 |     88% |317-328, 337-345, 363-366, 409, 458 |
| haystack/components/extractors/regex\_text\_extractor.py                  |       49 |        0 |    100% |           |
| haystack/components/fetchers/\_\_init\_\_.py                              |        5 |        0 |    100% |           |
| haystack/components/fetchers/link\_content.py                             |      202 |       27 |     87% |192-200, 267, 276-283, 316-320, 409-416, 449, 482 |
| haystack/components/generators/\_\_init\_\_.py                            |        5 |        0 |    100% |           |
| haystack/components/generators/chat/\_\_init\_\_.py                       |        5 |        0 |    100% |           |
| haystack/components/generators/chat/azure.py                              |       90 |        3 |     97% |213, 363, 366 |
| haystack/components/generators/chat/azure\_responses.py                   |       46 |        2 |     96% |  224, 271 |
| haystack/components/generators/chat/fallback.py                           |       98 |        3 |     97% |   252-257 |
| haystack/components/generators/chat/llm.py                                |       40 |        1 |     98% |       124 |
| haystack/components/generators/chat/mock.py                               |      144 |        0 |    100% |           |
| haystack/components/generators/chat/openai.py                             |      230 |        4 |     98% |458, 509, 680, 780 |
| haystack/components/generators/chat/openai\_responses.py                  |      358 |       41 |     89% |278, 309, 398, 478, 494, 531, 538-545, 553, 606-615, 631-632, 637, 647, 657-673, 682, 739, 836, 878-879, 892, 944, 966, 985 |
| haystack/components/generators/chat/types/\_\_init\_\_.py                 |        2 |        0 |    100% |           |
| haystack/components/generators/chat/types/protocol.py                     |        4 |        0 |    100% |           |
| haystack/components/generators/openai\_image\_generator.py                |       78 |        0 |    100% |           |
| haystack/components/generators/utils.py                                   |       85 |        9 |     89% |31, 45, 91, 124-125, 157, 173, 175, 177 |
| haystack/components/joiners/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| haystack/components/joiners/answer\_joiner.py                             |       49 |        1 |     98% |       140 |
| haystack/components/joiners/branch.py                                     |       20 |        3 |     85% |105, 116-117 |
| haystack/components/joiners/document\_joiner.py                           |       97 |        0 |    100% |           |
| haystack/components/joiners/list\_joiner.py                               |       23 |        0 |    100% |           |
| haystack/components/joiners/string\_joiner.py                             |        8 |        0 |    100% |           |
| haystack/components/preprocessors/\_\_init\_\_.py                         |        5 |        0 |    100% |           |
| haystack/components/preprocessors/csv\_document\_cleaner.py               |       64 |        0 |    100% |           |
| haystack/components/preprocessors/csv\_document\_splitter.py              |      107 |        4 |     96% |127, 130, 141-145 |
| haystack/components/preprocessors/document\_cleaner.py                    |      117 |        1 |     99% |       101 |
| haystack/components/preprocessors/document\_preprocessor.py               |       46 |        2 |     96% |  162, 197 |
| haystack/components/preprocessors/document\_splitter.py                   |      206 |        0 |    100% |           |
| haystack/components/preprocessors/embedding\_based\_document\_splitter.py |      212 |       40 |     81% |204-205, 244, 254-263, 271-280, 287-299, 305-308, 314-317, 427, 458-481, 544-545 |
| haystack/components/preprocessors/hierarchical\_document\_splitter.py     |       59 |        0 |    100% |           |
| haystack/components/preprocessors/markdown\_header\_splitter.py           |      162 |        6 |     96% |219-220, 228, 276, 299, 354 |
| haystack/components/preprocessors/python\_code\_splitter.py               |      289 |       15 |     95% |150, 162, 190, 197, 231, 280, 308-310, 411-413, 415, 528, 594 |
| haystack/components/preprocessors/recursive\_splitter.py                  |      237 |       18 |     92% |147-150, 192-194, 235-237, 253-255, 275, 400-403 |
| haystack/components/preprocessors/sentence\_tokenizer.py                  |       83 |        5 |     94% |62-63, 70-75, 215 |
| haystack/components/preprocessors/text\_cleaner.py                        |       29 |        0 |    100% |           |
| haystack/components/query/\_\_init\_\_.py                                 |        5 |        0 |    100% |           |
| haystack/components/query/query\_expander.py                              |      126 |       10 |     92% |270-271, 275, 284-285, 292-298, 309-312 |
| haystack/components/rankers/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| haystack/components/rankers/llm\_ranker.py                                |      147 |       12 |     92% |310, 313, 320-321, 342-349, 371, 380, 384 |
| haystack/components/rankers/lost\_in\_the\_middle.py                      |       43 |        4 |     91% |57, 83, 87, 117 |
| haystack/components/rankers/meta\_field.py                                |      118 |        0 |    100% |           |
| haystack/components/rankers/meta\_field\_grouping\_ranker.py              |       38 |        0 |    100% |           |
| haystack/components/retrievers/\_\_init\_\_.py                            |        5 |        0 |    100% |           |
| haystack/components/retrievers/auto\_merging\_retriever.py                |       80 |        0 |    100% |           |
| haystack/components/retrievers/filter\_retriever.py                       |       22 |        1 |     95% |        55 |
| haystack/components/retrievers/in\_memory/\_\_init\_\_.py                 |        5 |        0 |    100% |           |
| haystack/components/retrievers/in\_memory/bm25\_retriever.py              |       48 |        2 |     96% |  147, 185 |
| haystack/components/retrievers/in\_memory/embedding\_retriever.py         |       53 |        3 |     94% |102, 167, 218 |
| haystack/components/retrievers/multi\_query\_embedding\_retriever.py      |       77 |        2 |     97% |  193, 212 |
| haystack/components/retrievers/multi\_query\_text\_retriever.py           |       67 |        3 |     96% |123, 169, 184 |
| haystack/components/retrievers/multi\_retriever.py                        |      114 |        1 |     99% |       135 |
| haystack/components/retrievers/sentence\_window\_retriever.py             |       98 |        6 |     94% |139, 249, 269-276, 291-298 |
| haystack/components/retrievers/text\_embedding\_retriever.py              |       52 |        0 |    100% |           |
| haystack/components/retrievers/types/\_\_init\_\_.py                      |        2 |        0 |    100% |           |
| haystack/components/retrievers/types/protocol.py                          |        5 |        0 |    100% |           |
| haystack/components/routers/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| haystack/components/routers/conditional\_router.py                        |      172 |        8 |     95% |458-459, 495, 513, 555, 573, 583, 595 |
| haystack/components/routers/document\_length\_router.py                   |       15 |        0 |    100% |           |
| haystack/components/routers/document\_type\_router.py                     |       46 |        0 |    100% |           |
| haystack/components/routers/file\_type\_router.py                         |       72 |        6 |     92% |   182-187 |
| haystack/components/routers/llm\_messages\_router.py                      |       81 |        1 |     99% |       188 |
| haystack/components/routers/metadata\_router.py                           |       35 |        0 |    100% |           |
| haystack/components/samplers/\_\_init\_\_.py                              |        5 |        0 |    100% |           |
| haystack/components/samplers/top\_p.py                                    |       65 |        0 |    100% |           |
| haystack/components/validators/\_\_init\_\_.py                            |        5 |        0 |    100% |           |
| haystack/components/validators/json\_schema.py                            |       71 |        8 |     89% |23-24, 137, 139, 154, 242, 248, 252 |
| haystack/components/writers/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| haystack/components/writers/document\_writer.py                           |       32 |        0 |    100% |           |
| haystack/core/\_\_init\_\_.py                                             |        2 |        0 |    100% |           |
| haystack/core/component/\_\_init\_\_.py                                   |        3 |        0 |    100% |           |
| haystack/core/component/component.py                                      |      185 |        1 |     99% |       355 |
| haystack/core/component/sockets.py                                        |       41 |        6 |     85% |82, 117-124, 129-130 |
| haystack/core/component/types.py                                          |       42 |        3 |     93% |     87-89 |
| haystack/core/errors.py                                                   |       72 |       10 |     86% |60-67, 127, 142-144, 151-153, 166 |
| haystack/core/pipeline/\_\_init\_\_.py                                    |        2 |        0 |    100% |           |
| haystack/core/pipeline/base.py                                            |      590 |       53 |     91% |123, 265-266, 270, 409, 421, 611, 648-658, 823, 891, 1461, 1627, 1631, 1641-1647, 1664-1726 |
| haystack/core/pipeline/breakpoint.py                                      |      125 |       11 |     91% |53, 72, 81, 89, 116-121, 188 |
| haystack/core/pipeline/component\_checks.py                               |       57 |        0 |    100% |           |
| haystack/core/pipeline/descriptions.py                                    |        6 |        0 |    100% |           |
| haystack/core/pipeline/draw.py                                            |      199 |       41 |     79% |35-58, 142, 145, 148, 150, 158, 162-169, 293, 303, 375-377, 382, 390-391, 419-424 |
| haystack/core/pipeline/pipeline.py                                        |      317 |       23 |     93% |171, 178, 408, 452-453, 550, 573, 741, 932-936, 940, 957-961, 965-977, 1018-1027, 1052 |
| haystack/core/pipeline/utils.py                                           |       71 |        1 |     99% |       206 |
| haystack/core/serialization.py                                            |      127 |        6 |     95% |75, 99, 246, 333, 349-350 |
| haystack/core/serialization\_security.py                                  |       70 |        1 |     99% |       216 |
| haystack/core/super\_component/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| haystack/core/super\_component/super\_component.py                        |      194 |        7 |     96% |67, 196, 209, 247, 301, 343, 596 |
| haystack/core/super\_component/utils.py                                   |       95 |        4 |     96% |77, 112, 124, 181 |
| haystack/core/type\_utils.py                                              |      153 |        5 |     97% |51, 58, 121, 157, 249 |
| haystack/dataclasses/\_\_init\_\_.py                                      |        5 |        0 |    100% |           |
| haystack/dataclasses/answer.py                                            |       57 |        0 |    100% |           |
| haystack/dataclasses/breakpoints.py                                       |       37 |        0 |    100% |           |
| haystack/dataclasses/byte\_stream.py                                      |       39 |        0 |    100% |           |
| haystack/dataclasses/chat\_message.py                                     |      341 |        4 |     99% |140, 381, 397, 619 |
| haystack/dataclasses/document.py                                          |       91 |        4 |     96% |79, 81, 83, 85 |
| haystack/dataclasses/file\_content.py                                     |       68 |        0 |    100% |           |
| haystack/dataclasses/image\_content.py                                    |       85 |        2 |     98% |   100-104 |
| haystack/dataclasses/skill\_info.py                                       |        3 |        0 |    100% |           |
| haystack/dataclasses/sparse\_embedding.py                                 |       14 |        0 |    100% |           |
| haystack/dataclasses/streaming\_chunk.py                                  |       81 |        1 |     99% |       182 |
| haystack/document\_stores/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| haystack/document\_stores/errors/\_\_init\_\_.py                          |        2 |        0 |    100% |           |
| haystack/document\_stores/errors/errors.py                                |        6 |        0 |    100% |           |
| haystack/document\_stores/in\_memory/\_\_init\_\_.py                      |        5 |        0 |    100% |           |
| haystack/document\_stores/in\_memory/document\_store.py                   |      424 |       18 |     96% |411-412, 421, 521, 584, 623, 625, 652-653, 668, 692, 766-767, 828, 830, 848, 853-854 |
| haystack/document\_stores/types/\_\_init\_\_.py                           |        4 |        0 |    100% |           |
| haystack/document\_stores/types/filter\_policy.py                         |       65 |       12 |     82% |25, 38-39, 166, 174-181, 224-229, 233-239, 319 |
| haystack/document\_stores/types/policy.py                                 |        6 |        0 |    100% |           |
| haystack/document\_stores/types/protocol.py                               |       11 |        0 |    100% |           |
| haystack/errors.py                                                        |        2 |        0 |    100% |           |
| haystack/evaluation/\_\_init\_\_.py                                       |        5 |        0 |    100% |           |
| haystack/evaluation/eval\_run\_result.py                                  |       93 |       35 |     62% |72-97, 111-120, 189, 192, 195, 200, 211, 215 |
| haystack/hooks/\_\_init\_\_.py                                            |        5 |        0 |    100% |           |
| haystack/hooks/from\_function.py                                          |       50 |        2 |     96% |     29-30 |
| haystack/hooks/human\_in\_the\_loop/\_\_init\_\_.py                       |        5 |        0 |    100% |           |
| haystack/hooks/human\_in\_the\_loop/dataclasses.py                        |       16 |        0 |    100% |           |
| haystack/hooks/human\_in\_the\_loop/hooks.py                              |       30 |        1 |     97% |       117 |
| haystack/hooks/human\_in\_the\_loop/policies.py                           |       17 |        0 |    100% |           |
| haystack/hooks/human\_in\_the\_loop/strategies.py                         |      158 |        9 |     94% |120-121, 265, 314, 358, 419, 453, 510, 620 |
| haystack/hooks/human\_in\_the\_loop/types/\_\_init\_\_.py                 |        2 |        0 |    100% |           |
| haystack/hooks/human\_in\_the\_loop/types/protocol.py                     |       25 |        0 |    100% |           |
| haystack/hooks/human\_in\_the\_loop/user\_interfaces.py                   |       96 |        2 |     98% |   65, 103 |
| haystack/hooks/invocation.py                                              |        9 |        0 |    100% |           |
| haystack/hooks/protocol.py                                                |       15 |        0 |    100% |           |
| haystack/hooks/tool\_result\_offloading/\_\_init\_\_.py                   |        5 |        0 |    100% |           |
| haystack/hooks/tool\_result\_offloading/hooks.py                          |       95 |        1 |     99% |       222 |
| haystack/hooks/tool\_result\_offloading/policies.py                       |       17 |        0 |    100% |           |
| haystack/hooks/tool\_result\_offloading/stores.py                         |       22 |        0 |    100% |           |
| haystack/hooks/tool\_result\_offloading/types/\_\_init\_\_.py             |        2 |        0 |    100% |           |
| haystack/hooks/tool\_result\_offloading/types/protocol.py                 |       18 |        2 |     89% |    39, 44 |
| haystack/hooks/utils.py                                                   |       47 |        0 |    100% |           |
| haystack/lazy\_imports.py                                                 |       13 |        0 |    100% |           |
| haystack/logging.py                                                       |      129 |        4 |     97% |247-248, 358-360 |
| haystack/marshal/\_\_init\_\_.py                                          |        5 |        0 |    100% |           |
| haystack/marshal/protocol.py                                              |        4 |        0 |    100% |           |
| haystack/marshal/yaml.py                                                  |       21 |        1 |     95% |        42 |
| haystack/skill\_stores/\_\_init\_\_.py                                    |        0 |        0 |    100% |           |
| haystack/skill\_stores/file\_system/\_\_init\_\_.py                       |        5 |        0 |    100% |           |
| haystack/skill\_stores/file\_system/skill\_store.py                       |       95 |        1 |     99% |       119 |
| haystack/skill\_stores/types/\_\_init\_\_.py                              |        2 |        0 |    100% |           |
| haystack/skill\_stores/types/protocol.py                                  |       11 |        0 |    100% |           |
| haystack/telemetry/\_\_init\_\_.py                                        |        1 |        0 |    100% |           |
| haystack/telemetry/\_environment.py                                       |       25 |        0 |    100% |           |
| haystack/telemetry/\_telemetry.py                                         |       83 |       12 |     86% |70-76, 94-95, 114-115, 188 |
| haystack/tools/\_\_init\_\_.py                                            |        7 |        0 |    100% |           |
| haystack/tools/component\_tool.py                                         |      100 |        5 |     95% |372-373, 416-418 |
| haystack/tools/errors.py                                                  |        6 |        0 |    100% |           |
| haystack/tools/from\_function.py                                          |       56 |        0 |    100% |           |
| haystack/tools/parameters\_schema\_utils.py                               |       97 |        3 |     97% |95, 135-136 |
| haystack/tools/pipeline\_tool.py                                          |       30 |        2 |     93% |  240, 243 |
| haystack/tools/searchable\_toolset.py                                     |      120 |        2 |     98% |  191, 214 |
| haystack/tools/serde\_utils.py                                            |       43 |        3 |     93% |33, 35, 57 |
| haystack/tools/skills/\_\_init\_\_.py                                     |        2 |        0 |    100% |           |
| haystack/tools/skills/skill\_toolset.py                                   |       61 |        0 |    100% |           |
| haystack/tools/tool.py                                                    |      150 |        5 |     97% |174, 197, 241-242, 376 |
| haystack/tools/tool\_types.py                                             |        5 |        0 |    100% |           |
| haystack/tools/toolset.py                                                 |      126 |       13 |     90% |119, 193, 249, 301, 431-434, 440-444 |
| haystack/tools/utils.py                                                   |       29 |        0 |    100% |           |
| haystack/tracing/\_\_init\_\_.py                                          |        1 |        0 |    100% |           |
| haystack/tracing/logging\_tracer.py                                       |       36 |        0 |    100% |           |
| haystack/tracing/tracer.py                                                |       54 |        4 |     93% |28, 74, 94, 103 |
| haystack/tracing/utils.py                                                 |       26 |        0 |    100% |           |
| haystack/utils/\_\_init\_\_.py                                            |        5 |        0 |    100% |           |
| haystack/utils/async\_utils.py                                            |       10 |        0 |    100% |           |
| haystack/utils/auth.py                                                    |      105 |       11 |     90% |22, 116, 124, 128, 133, 161, 234-238 |
| haystack/utils/azure.py                                                   |        6 |        2 |     67% |     15-16 |
| haystack/utils/base\_serialization.py                                     |      115 |       14 |     88% |106-112, 149, 160, 257-261 |
| haystack/utils/callable\_serialization.py                                 |       59 |        8 |     86% |51, 55, 104-106, 110, 121, 130 |
| haystack/utils/dataclasses.py                                             |       22 |        0 |    100% |           |
| haystack/utils/deserialization.py                                         |       18 |        1 |     94% |        54 |
| haystack/utils/device.py                                                  |      214 |       17 |     92% |237-239, 325-327, 364, 403, 447, 468, 484, 512-515, 539-540 |
| haystack/utils/experimental.py                                            |       14 |        0 |    100% |           |
| haystack/utils/filters.py                                                 |      119 |        0 |    100% |           |
| haystack/utils/hf.py                                                      |       61 |       15 |     75% |23-31, 40-51 |
| haystack/utils/http\_client.py                                            |       14 |        0 |    100% |           |
| haystack/utils/jinja2\_chat\_extension.py                                 |      146 |        1 |     99% |       405 |
| haystack/utils/jinja2\_extensions.py                                      |       47 |        0 |    100% |           |
| haystack/utils/jupyter.py                                                 |        9 |        3 |     67% |     15-17 |
| haystack/utils/misc.py                                                    |       96 |        5 |     95% |52-53, 129-132 |
| haystack/utils/requests\_utils.py                                         |       33 |        0 |    100% |           |
| haystack/utils/type\_serialization.py                                     |      121 |        6 |     95% |65, 67, 187-188, 196, 218 |
| haystack/utils/url\_validation.py                                         |        4 |        0 |    100% |           |
| haystack/version.py                                                       |        5 |        2 |     60% |      9-10 |
| **TOTAL**                                                                 | **15819** |  **845** | **95%** |           |


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
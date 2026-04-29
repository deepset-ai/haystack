# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                                |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| haystack/\_\_init\_\_.py                                                            |       12 |        0 |    100% |           |
| haystack/components/\_\_init\_\_.py                                                 |        0 |        0 |    100% |           |
| haystack/components/agents/\_\_init\_\_.py                                          |        5 |        0 |    100% |           |
| haystack/components/agents/agent.py                                                 |      380 |       12 |     97% |477, 651, 704-705, 707, 853, 927, 1187-1188, 1194, 1229-1231 |
| haystack/components/agents/state/\_\_init\_\_.py                                    |        5 |        0 |    100% |           |
| haystack/components/agents/state/state.py                                           |       73 |        2 |     97% |    75, 79 |
| haystack/components/agents/state/state\_utils.py                                    |       18 |        0 |    100% |           |
| haystack/components/audio/\_\_init\_\_.py                                           |        5 |        0 |    100% |           |
| haystack/components/audio/whisper\_local.py                                         |       66 |        6 |     91% |128, 164-167, 183 |
| haystack/components/audio/whisper\_remote.py                                        |       43 |       13 |     70% |96, 149-164 |
| haystack/components/builders/\_\_init\_\_.py                                        |        5 |        0 |    100% |           |
| haystack/components/builders/answer\_builder.py                                     |       70 |        1 |     99% |       242 |
| haystack/components/builders/chat\_prompt\_builder.py                               |      110 |        2 |     98% |  176, 263 |
| haystack/components/builders/prompt\_builder.py                                     |       50 |        0 |    100% |           |
| haystack/components/caching/\_\_init\_\_.py                                         |        5 |        0 |    100% |           |
| haystack/components/caching/cache\_checker.py                                       |       24 |        0 |    100% |           |
| haystack/components/classifiers/\_\_init\_\_.py                                     |        5 |        0 |    100% |           |
| haystack/components/classifiers/document\_language\_classifier.py                   |       35 |        0 |    100% |           |
| haystack/components/classifiers/zero\_shot\_document\_classifier.py                 |       58 |        5 |     91% |140-142, 217, 220 |
| haystack/components/connectors/\_\_init\_\_.py                                      |        5 |        0 |    100% |           |
| haystack/components/connectors/openapi.py                                           |       24 |        0 |    100% |           |
| haystack/components/connectors/openapi\_service.py                                  |      146 |       55 |     62% |51-140, 338, 365, 373 |
| haystack/components/converters/\_\_init\_\_.py                                      |        5 |        0 |    100% |           |
| haystack/components/converters/azure.py                                             |      211 |       22 |     90% |117, 146-148, 195, 259-260, 266, 274, 281, 288-293, 301, 310, 346-347, 361, 402-408, 468 |
| haystack/components/converters/csv.py                                               |       88 |        4 |     95% |161-162, 184-185 |
| haystack/components/converters/docx.py                                              |      159 |        1 |     99% |       262 |
| haystack/components/converters/file\_to\_file\_content.py                           |       32 |        0 |    100% |           |
| haystack/components/converters/html.py                                              |       45 |        0 |    100% |           |
| haystack/components/converters/image/\_\_init\_\_.py                                |        5 |        0 |    100% |           |
| haystack/components/converters/image/document\_to\_image.py                         |       43 |        0 |    100% |           |
| haystack/components/converters/image/file\_to\_document.py                          |       27 |        1 |     96% |        93 |
| haystack/components/converters/image/file\_to\_image.py                             |       53 |        0 |    100% |           |
| haystack/components/converters/image/image\_utils.py                                |      123 |        5 |     96% |78-82, 104, 109-110 |
| haystack/components/converters/image/pdf\_to\_image.py                              |       46 |        3 |     93% |   137-141 |
| haystack/components/converters/json.py                                              |       87 |       11 |     87% |220-221, 223-226, 230-231, 243-244, 275-277 |
| haystack/components/converters/markdown.py                                          |       44 |        7 |     84% |88, 101-103, 107-113 |
| haystack/components/converters/msg.py                                               |       75 |        5 |     93% |91, 165-167, 179 |
| haystack/components/converters/multi\_file\_converter.py                            |       50 |        0 |    100% |           |
| haystack/components/converters/openapi\_functions.py                                |      117 |       25 |     79% |86-87, 100-103, 111-112, 139, 144, 159-160, 180-187, 192-195, 231-232, 250, 254-258 |
| haystack/components/converters/output\_adapter.py                                   |       65 |        1 |     98% |       129 |
| haystack/components/converters/pdfminer.py                                          |       68 |        0 |    100% |           |
| haystack/components/converters/pptx.py                                              |       69 |        2 |     97% |     89-90 |
| haystack/components/converters/pypdf.py                                             |       74 |        3 |     96% |   209-213 |
| haystack/components/converters/tika.py                                              |       59 |        4 |     93% |36, 133-139 |
| haystack/components/converters/txt.py                                               |       34 |        3 |     91% |     87-91 |
| haystack/components/converters/utils.py                                             |       21 |        0 |    100% |           |
| haystack/components/converters/xlsx.py                                              |      113 |        3 |     97% |83, 183, 185 |
| haystack/components/embedders/\_\_init\_\_.py                                       |        5 |        0 |    100% |           |
| haystack/components/embedders/azure\_document\_embedder.py                          |       50 |        2 |     96% |  121, 124 |
| haystack/components/embedders/azure\_text\_embedder.py                              |       44 |        2 |     95% |  107, 110 |
| haystack/components/embedders/backends/\_\_init\_\_.py                              |        0 |        0 |    100% |           |
| haystack/components/embedders/backends/sentence\_transformers\_backend.py           |       25 |        0 |    100% |           |
| haystack/components/embedders/backends/sentence\_transformers\_sparse\_backend.py   |       31 |        0 |    100% |           |
| haystack/components/embedders/hugging\_face\_api\_document\_embedder.py             |      122 |        4 |     97% |174-175, 344, 372 |
| haystack/components/embedders/hugging\_face\_api\_text\_embedder.py                 |       83 |        4 |     95% |137-138, 252, 254 |
| haystack/components/embedders/image/\_\_init\_\_.py                                 |        5 |        0 |    100% |           |
| haystack/components/embedders/image/sentence\_transformers\_doc\_image\_embedder.py |       78 |        1 |     99% |       237 |
| haystack/components/embedders/openai\_document\_embedder.py                         |      126 |       42 |     67% |147, 185, 218, 238-239, 250-284, 331-350 |
| haystack/components/embedders/openai\_text\_embedder.py                             |       54 |        6 |     89% |122, 188-189, 207-209 |
| haystack/components/embedders/sentence\_transformers\_document\_embedder.py         |       64 |        2 |     97% |  153, 243 |
| haystack/components/embedders/sentence\_transformers\_sparse\_document\_embedder.py |       60 |        2 |     97% |  130, 215 |
| haystack/components/embedders/sentence\_transformers\_sparse\_text\_embedder.py     |       49 |        2 |     96% |  109, 191 |
| haystack/components/embedders/sentence\_transformers\_text\_embedder.py             |       54 |        2 |     96% |  141, 229 |
| haystack/components/embedders/types/\_\_init\_\_.py                                 |        2 |        0 |    100% |           |
| haystack/components/embedders/types/protocol.py                                     |        6 |        0 |    100% |           |
| haystack/components/evaluators/\_\_init\_\_.py                                      |        5 |        0 |    100% |           |
| haystack/components/evaluators/answer\_exact\_match.py                              |       15 |        0 |    100% |           |
| haystack/components/evaluators/context\_relevance.py                                |       36 |        0 |    100% |           |
| haystack/components/evaluators/document\_map.py                                     |       46 |        4 |     91% |73, 76-80, 126 |
| haystack/components/evaluators/document\_mrr.py                                     |       42 |        4 |     90% |71, 74-78, 122 |
| haystack/components/evaluators/document\_ndcg.py                                    |       43 |        0 |    100% |           |
| haystack/components/evaluators/document\_recall.py                                  |       69 |        3 |     96% |106, 109-113 |
| haystack/components/evaluators/faithfulness.py                                      |       36 |        0 |    100% |           |
| haystack/components/evaluators/llm\_evaluator.py                                    |      109 |        2 |     98% |  216, 232 |
| haystack/components/evaluators/sas\_evaluator.py                                    |       57 |       25 |     56% |111-125, 153-188 |
| haystack/components/extractors/\_\_init\_\_.py                                      |        5 |        0 |    100% |           |
| haystack/components/extractors/image/\_\_init\_\_.py                                |        5 |        0 |    100% |           |
| haystack/components/extractors/image/llm\_document\_content\_extractor.py           |      105 |        0 |    100% |           |
| haystack/components/extractors/llm\_metadata\_extractor.py                          |      153 |       31 |     80% |287-298, 305-316, 331-343, 380, 422-426, 437 |
| haystack/components/extractors/named\_entity\_extractor.py                          |      182 |       59 |     68% |24, 39, 158-161, 170-177, 196, 202, 261, 274, 377-389, 392-397, 412, 440-448, 452-464, 467-474, 484, 488, 501-508 |
| haystack/components/extractors/regex\_text\_extractor.py                            |       49 |        0 |    100% |           |
| haystack/components/fetchers/\_\_init\_\_.py                                        |        5 |        0 |    100% |           |
| haystack/components/fetchers/link\_content.py                                       |      188 |       26 |     86% |160-165, 222-224, 246, 255-262, 292-296, 385-392, 425, 458 |
| haystack/components/generators/\_\_init\_\_.py                                      |        5 |        0 |    100% |           |
| haystack/components/generators/azure.py                                             |       48 |        4 |     92% |130, 133, 209, 212 |
| haystack/components/generators/chat/\_\_init\_\_.py                                 |        5 |        0 |    100% |           |
| haystack/components/generators/chat/azure.py                                        |       68 |        4 |     94% |198, 201, 312, 315 |
| haystack/components/generators/chat/azure\_responses.py                             |       46 |        2 |     96% |  224, 271 |
| haystack/components/generators/chat/fallback.py                                     |       86 |        4 |     95% |92, 240-245 |
| haystack/components/generators/chat/hugging\_face\_api.py                           |      229 |        7 |     97% |189, 420-421, 487, 579, 632, 703 |
| haystack/components/generators/chat/hugging\_face\_local.py                         |      208 |       27 |     87% |70-72, 75, 83-85, 235, 278-280, 340, 370, 426, 455-460, 487, 538-541, 576-582, 598, 654-655 |
| haystack/components/generators/chat/llm.py                                          |       30 |        2 |     93% |  117, 185 |
| haystack/components/generators/chat/openai.py                                       |      196 |        4 |     98% |421, 471, 609, 709 |
| haystack/components/generators/chat/openai\_responses.py                            |      316 |       39 |     88% |237, 268, 356, 434, 449, 486, 493-500, 508, 549-558, 574-575, 580, 593-609, 618, 748, 790-791, 804, 847, 869, 888 |
| haystack/components/generators/chat/types/\_\_init\_\_.py                           |        2 |        0 |    100% |           |
| haystack/components/generators/chat/types/protocol.py                               |        4 |        0 |    100% |           |
| haystack/components/generators/hugging\_face\_api.py                                |       86 |        3 |     97% |164-165, 263 |
| haystack/components/generators/hugging\_face\_local.py                              |       87 |        7 |     92% |128-130, 193, 242-248 |
| haystack/components/generators/openai.py                                            |       68 |        4 |     94% |149, 215, 217, 243 |
| haystack/components/generators/openai\_dalle.py                                     |       44 |        2 |     95% |   132-133 |
| haystack/components/generators/utils.py                                             |       79 |        9 |     89% |31, 45, 91, 124-125, 157, 167, 169, 171 |
| haystack/components/joiners/\_\_init\_\_.py                                         |        5 |        0 |    100% |           |
| haystack/components/joiners/answer\_joiner.py                                       |       49 |        2 |     96% |  132, 138 |
| haystack/components/joiners/branch.py                                               |       20 |        3 |     85% |105, 116-117 |
| haystack/components/joiners/document\_joiner.py                                     |      103 |        0 |    100% |           |
| haystack/components/joiners/list\_joiner.py                                         |       23 |        0 |    100% |           |
| haystack/components/joiners/string\_joiner.py                                       |        8 |        0 |    100% |           |
| haystack/components/preprocessors/\_\_init\_\_.py                                   |        5 |        0 |    100% |           |
| haystack/components/preprocessors/csv\_document\_cleaner.py                         |       64 |        0 |    100% |           |
| haystack/components/preprocessors/csv\_document\_splitter.py                        |      107 |        4 |     96% |127, 130, 141-145 |
| haystack/components/preprocessors/document\_cleaner.py                              |      117 |        2 |     98% |  101, 345 |
| haystack/components/preprocessors/document\_preprocessor.py                         |       46 |        2 |     96% |  162, 197 |
| haystack/components/preprocessors/document\_splitter.py                             |      204 |        0 |    100% |           |
| haystack/components/preprocessors/embedding\_based\_document\_splitter.py           |      202 |       42 |     79% |170-171, 198-202, 218, 228-237, 245-254, 261-273, 279-282, 288-291, 401, 432-455, 518-519 |
| haystack/components/preprocessors/hierarchical\_document\_splitter.py               |       54 |        0 |    100% |           |
| haystack/components/preprocessors/markdown\_header\_splitter.py                     |      162 |        6 |     96% |219-220, 228, 276, 299, 354 |
| haystack/components/preprocessors/recursive\_splitter.py                            |      227 |       21 |     91% |105-106, 145-148, 190-192, 233-235, 251-253, 273, 356, 395-398 |
| haystack/components/preprocessors/sentence\_tokenizer.py                            |       83 |        5 |     94% |62-63, 70-75, 215 |
| haystack/components/preprocessors/text\_cleaner.py                                  |       29 |        0 |    100% |           |
| haystack/components/query/\_\_init\_\_.py                                           |        5 |        0 |    100% |           |
| haystack/components/query/query\_expander.py                                        |       83 |        0 |    100% |           |
| haystack/components/rankers/\_\_init\_\_.py                                         |        5 |        0 |    100% |           |
| haystack/components/rankers/hugging\_face\_tei.py                                   |       78 |        5 |     94% |226-227, 271, 288-289 |
| haystack/components/rankers/llm\_ranker.py                                          |      107 |        3 |     97% |291, 300, 304 |
| haystack/components/rankers/lost\_in\_the\_middle.py                                |       43 |        5 |     88% |57, 83, 87, 104, 117 |
| haystack/components/rankers/meta\_field.py                                          |      118 |        0 |    100% |           |
| haystack/components/rankers/meta\_field\_grouping\_ranker.py                        |       33 |        0 |    100% |           |
| haystack/components/rankers/sentence\_transformers\_diversity.py                    |      151 |        8 |     95% |243, 258, 389, 410, 422-425 |
| haystack/components/rankers/sentence\_transformers\_similarity.py                   |       79 |        2 |     97% |  147, 250 |
| haystack/components/rankers/transformers\_similarity.py                             |      109 |       10 |     91% |143, 148, 154, 175, 261, 272, 275, 296, 308-309 |
| haystack/components/readers/\_\_init\_\_.py                                         |        5 |        0 |    100% |           |
| haystack/components/readers/extractive.py                                           |      229 |       10 |     96% |133, 195, 220-225, 326, 410, 467, 580, 625-626 |
| haystack/components/retrievers/\_\_init\_\_.py                                      |        5 |        0 |    100% |           |
| haystack/components/retrievers/auto\_merging\_retriever.py                          |       80 |        0 |    100% |           |
| haystack/components/retrievers/filter\_retriever.py                                 |       22 |        1 |     95% |        55 |
| haystack/components/retrievers/in\_memory/\_\_init\_\_.py                           |        5 |        0 |    100% |           |
| haystack/components/retrievers/in\_memory/bm25\_retriever.py                        |       48 |        2 |     96% |  147, 185 |
| haystack/components/retrievers/in\_memory/embedding\_retriever.py                   |       53 |       13 |     75% |102, 167, 217-236 |
| haystack/components/retrievers/multi\_query\_embedding\_retriever.py                |       48 |        3 |     94% |94, 96, 141 |
| haystack/components/retrievers/multi\_query\_text\_retriever.py                     |       42 |        3 |     93% |76, 100, 120 |
| haystack/components/retrievers/multi\_retriever.py                                  |       85 |        2 |     98% |  132, 135 |
| haystack/components/retrievers/sentence\_window\_retriever.py                       |       98 |        6 |     94% |139, 249, 269-276, 291-298 |
| haystack/components/retrievers/text\_embedding\_retriever.py                        |       32 |        2 |     94% |    77, 79 |
| haystack/components/retrievers/types/\_\_init\_\_.py                                |        2 |        0 |    100% |           |
| haystack/components/retrievers/types/protocol.py                                    |        5 |        0 |    100% |           |
| haystack/components/routers/\_\_init\_\_.py                                         |        5 |        0 |    100% |           |
| haystack/components/routers/conditional\_router.py                                  |      159 |        8 |     95% |372-373, 409, 424, 466, 484, 494, 506 |
| haystack/components/routers/document\_length\_router.py                             |       15 |        0 |    100% |           |
| haystack/components/routers/document\_type\_router.py                               |       46 |        0 |    100% |           |
| haystack/components/routers/file\_type\_router.py                                   |       72 |        6 |     92% |   181-186 |
| haystack/components/routers/llm\_messages\_router.py                                |       51 |        0 |    100% |           |
| haystack/components/routers/metadata\_router.py                                     |       35 |        0 |    100% |           |
| haystack/components/routers/text\_language\_router.py                               |       31 |        0 |    100% |           |
| haystack/components/routers/transformers\_text\_router.py                           |       51 |        5 |     90% |111, 120-122, 136 |
| haystack/components/routers/zero\_shot\_text\_router.py                             |       47 |        3 |     94% |   141-143 |
| haystack/components/samplers/\_\_init\_\_.py                                        |        5 |        0 |    100% |           |
| haystack/components/samplers/top\_p.py                                              |       65 |        0 |    100% |           |
| haystack/components/tools/\_\_init\_\_.py                                           |        5 |        0 |    100% |           |
| haystack/components/tools/tool\_invoker.py                                          |      281 |        8 |     97% |274, 296-304, 768, 782, 799-800 |
| haystack/components/validators/\_\_init\_\_.py                                      |        5 |        0 |    100% |           |
| haystack/components/validators/json\_schema.py                                      |       71 |        8 |     89% |23-24, 137, 139, 154, 242, 248, 252 |
| haystack/components/websearch/\_\_init\_\_.py                                       |        5 |        0 |    100% |           |
| haystack/components/websearch/searchapi.py                                          |       78 |        1 |     99% |        96 |
| haystack/components/websearch/serper\_dev.py                                        |      104 |       21 |     80% |139-141, 223, 241-255, 265-268 |
| haystack/components/writers/\_\_init\_\_.py                                         |        5 |        0 |    100% |           |
| haystack/components/writers/document\_writer.py                                     |       32 |        0 |    100% |           |
| haystack/core/\_\_init\_\_.py                                                       |        2 |        0 |    100% |           |
| haystack/core/component/\_\_init\_\_.py                                             |        3 |        0 |    100% |           |
| haystack/core/component/component.py                                                |      185 |        1 |     99% |       355 |
| haystack/core/component/sockets.py                                                  |       41 |        6 |     85% |82, 117-124, 129-130 |
| haystack/core/component/types.py                                                    |       36 |        3 |     92% |     85-87 |
| haystack/core/errors.py                                                             |       76 |        9 |     88% |60-67, 127, 146, 150, 161, 165, 178, 187 |
| haystack/core/pipeline/\_\_init\_\_.py                                              |        3 |        0 |    100% |           |
| haystack/core/pipeline/async\_pipeline.py                                           |      171 |       48 |     72% |91-92, 97, 258-301, 313, 370-376, 389, 394, 411-413, 417-420, 440-453, 463 |
| haystack/core/pipeline/base.py                                                      |      562 |       53 |     91% |124, 240-241, 245, 362, 374, 564, 601-611, 776, 844, 1373, 1539, 1543, 1553-1559, 1576-1638 |
| haystack/core/pipeline/breakpoint.py                                                |      175 |       14 |     92% |68, 73, 81, 102, 111, 119, 127, 150-155, 226 |
| haystack/core/pipeline/component\_checks.py                                         |       64 |        0 |    100% |           |
| haystack/core/pipeline/descriptions.py                                              |        6 |        0 |    100% |           |
| haystack/core/pipeline/draw.py                                                      |      161 |       41 |     75% |35-58, 142, 145, 148, 150, 158, 162-169, 218, 228, 296-298, 303, 311-312, 340-345 |
| haystack/core/pipeline/pipeline.py                                                  |      119 |        7 |     94% |238-242, 313, 344, 362-363, 447 |
| haystack/core/pipeline/utils.py                                                     |       69 |        1 |     99% |       201 |
| haystack/core/serialization.py                                                      |      116 |        5 |     96% |74, 98, 109, 240, 307 |
| haystack/core/super\_component/\_\_init\_\_.py                                      |        2 |        0 |    100% |           |
| haystack/core/super\_component/super\_component.py                                  |      196 |        8 |     96% |68, 148, 185, 198, 236, 290, 332, 587 |
| haystack/core/super\_component/utils.py                                             |       95 |        4 |     96% |77, 112, 124, 181 |
| haystack/core/type\_utils.py                                                        |      151 |        5 |     97% |51, 58, 121, 157, 245 |
| haystack/dataclasses/\_\_init\_\_.py                                                |        5 |        0 |    100% |           |
| haystack/dataclasses/answer.py                                                      |       59 |        0 |    100% |           |
| haystack/dataclasses/breakpoints.py                                                 |       72 |        3 |     96% | 61-62, 91 |
| haystack/dataclasses/byte\_stream.py                                                |       39 |        0 |    100% |           |
| haystack/dataclasses/chat\_message.py                                               |      330 |        4 |     99% |140, 367, 383, 605 |
| haystack/dataclasses/document.py                                                    |       90 |        6 |     93% |33, 37, 81, 83, 85, 87 |
| haystack/dataclasses/file\_content.py                                               |       68 |        0 |    100% |           |
| haystack/dataclasses/image\_content.py                                              |       85 |        2 |     98% |   100-104 |
| haystack/dataclasses/sparse\_embedding.py                                           |       14 |        0 |    100% |           |
| haystack/dataclasses/streaming\_chunk.py                                            |       73 |        3 |     96% |179, 239, 241 |
| haystack/document\_stores/\_\_init\_\_.py                                           |        0 |        0 |    100% |           |
| haystack/document\_stores/errors/\_\_init\_\_.py                                    |        2 |        0 |    100% |           |
| haystack/document\_stores/errors/errors.py                                          |        6 |        0 |    100% |           |
| haystack/document\_stores/in\_memory/\_\_init\_\_.py                                |        5 |        0 |    100% |           |
| haystack/document\_stores/in\_memory/document\_store.py                             |      408 |       21 |     95% |400-401, 410, 507, 564, 603, 605, 632-633, 648, 671-675, 737-738, 799, 801, 816, 821-822, 953 |
| haystack/document\_stores/types/\_\_init\_\_.py                                     |        4 |        0 |    100% |           |
| haystack/document\_stores/types/filter\_policy.py                                   |       65 |       12 |     82% |25, 38-39, 166, 174-181, 224-229, 233-239, 319 |
| haystack/document\_stores/types/policy.py                                           |        6 |        0 |    100% |           |
| haystack/document\_stores/types/protocol.py                                         |       11 |        0 |    100% |           |
| haystack/errors.py                                                                  |        2 |        0 |    100% |           |
| haystack/evaluation/\_\_init\_\_.py                                                 |        5 |        0 |    100% |           |
| haystack/evaluation/eval\_run\_result.py                                            |       93 |       35 |     62% |72-97, 111-120, 189, 192, 195, 200, 211, 215 |
| haystack/human\_in\_the\_loop/\_\_init\_\_.py                                       |        5 |        0 |    100% |           |
| haystack/human\_in\_the\_loop/dataclasses.py                                        |       16 |        0 |    100% |           |
| haystack/human\_in\_the\_loop/policies.py                                           |       17 |        0 |    100% |           |
| haystack/human\_in\_the\_loop/strategies.py                                         |      157 |       15 |     90% |124-125, 261, 285-303, 328-346, 372, 443, 486, 543 |
| haystack/human\_in\_the\_loop/types/\_\_init\_\_.py                                 |        2 |        0 |    100% |           |
| haystack/human\_in\_the\_loop/types/protocol.py                                     |       25 |        0 |    100% |           |
| haystack/human\_in\_the\_loop/user\_interfaces.py                                   |       96 |        2 |     98% |   65, 103 |
| haystack/lazy\_imports.py                                                           |       13 |        0 |    100% |           |
| haystack/logging.py                                                                 |      118 |        7 |     94% |230-231, 236-237, 272, 317-319 |
| haystack/marshal/\_\_init\_\_.py                                                    |        5 |        0 |    100% |           |
| haystack/marshal/protocol.py                                                        |        4 |        0 |    100% |           |
| haystack/marshal/yaml.py                                                            |       21 |        1 |     95% |        42 |
| haystack/telemetry/\_\_init\_\_.py                                                  |        1 |        0 |    100% |           |
| haystack/telemetry/\_environment.py                                                 |       25 |        0 |    100% |           |
| haystack/telemetry/\_telemetry.py                                                   |       81 |       12 |     85% |69-75, 93-94, 113-114, 187 |
| haystack/tools/\_\_init\_\_.py                                                      |       10 |        0 |    100% |           |
| haystack/tools/component\_tool.py                                                   |       90 |        5 |     94% |348-349, 381-383 |
| haystack/tools/errors.py                                                            |        6 |        0 |    100% |           |
| haystack/tools/from\_function.py                                                    |       55 |        0 |    100% |           |
| haystack/tools/parameters\_schema\_utils.py                                         |      105 |        3 |     97% |95, 148-149 |
| haystack/tools/pipeline\_tool.py                                                    |       32 |        2 |     94% |  249, 252 |
| haystack/tools/searchable\_toolset.py                                               |      110 |        0 |    100% |           |
| haystack/tools/serde\_utils.py                                                      |       43 |        3 |     93% |33, 35, 57 |
| haystack/tools/tool.py                                                              |      152 |        5 |     97% |155, 178, 219-220, 320 |
| haystack/tools/toolset.py                                                           |       86 |       13 |     85% |157, 187, 230-233, 278, 351-354, 362-364 |
| haystack/tools/utils.py                                                             |       29 |        0 |    100% |           |
| haystack/tracing/\_\_init\_\_.py                                                    |        2 |        0 |    100% |           |
| haystack/tracing/datadog.py                                                         |       47 |        1 |     98% |        95 |
| haystack/tracing/logging\_tracer.py                                                 |       33 |        0 |    100% |           |
| haystack/tracing/opentelemetry.py                                                   |       36 |        1 |     97% |        72 |
| haystack/tracing/tracer.py                                                          |       88 |        7 |     92% |33, 79, 99, 108, 223-226 |
| haystack/tracing/utils.py                                                           |       26 |        0 |    100% |           |
| haystack/utils/\_\_init\_\_.py                                                      |        5 |        0 |    100% |           |
| haystack/utils/asynchronous.py                                                      |        4 |        0 |    100% |           |
| haystack/utils/auth.py                                                              |      103 |       11 |     89% |22, 116, 124, 128, 133, 161, 230-234 |
| haystack/utils/azure.py                                                             |        6 |        2 |     67% |     15-16 |
| haystack/utils/base\_serialization.py                                               |      129 |       14 |     89% |148-154, 191, 202, 295-299 |
| haystack/utils/callable\_serialization.py                                           |       47 |        2 |     96% |    42, 76 |
| haystack/utils/dataclasses.py                                                       |       22 |        0 |    100% |           |
| haystack/utils/deserialization.py                                                   |       18 |        0 |    100% |           |
| haystack/utils/device.py                                                            |      214 |       13 |     94% |237-239, 364, 447, 468, 484, 512-515, 539-540 |
| haystack/utils/experimental.py                                                      |       14 |        0 |    100% |           |
| haystack/utils/filters.py                                                           |      108 |        5 |     95% |19-21, 98, 102 |
| haystack/utils/hf.py                                                                |      198 |       25 |     87% |99, 233-254, 358, 363-366, 416-419 |
| haystack/utils/http\_client.py                                                      |       14 |        0 |    100% |           |
| haystack/utils/jinja2\_chat\_extension.py                                           |      119 |        1 |     99% |       294 |
| haystack/utils/jinja2\_extensions.py                                                |       47 |        0 |    100% |           |
| haystack/utils/jupyter.py                                                           |        9 |        3 |     67% |     15-17 |
| haystack/utils/misc.py                                                              |       76 |       11 |     86% |47-50, 53, 60-61, 64, 121-124 |
| haystack/utils/requests\_utils.py                                                   |       33 |        0 |    100% |           |
| haystack/utils/type\_serialization.py                                               |      111 |       12 |     89% |62, 64, 174-175, 186-189, 195, 211-214 |
| haystack/utils/url\_validation.py                                                   |        4 |        0 |    100% |           |
| haystack/version.py                                                                 |        5 |        2 |     60% |      9-10 |
| **TOTAL**                                                                           | **17215** | **1228** | **93%** |           |


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
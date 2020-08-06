
Scaling to Open Domain
======================

Variable Length Documents
-------------------------

Transformers are limited in the amount of text they can process in one pass
Practically usually around 512 (see max seq len arg in ...)

To deal with bigger docs, sliding window approach
Passages are created from full doc by taking spans of that max seq len size
Each one starts doc_stride tokens apart from each other

Code snippet of Reader with params set (max seq len, window size, doc stride)

Reader will generate candidate answers in each paragraph
All candidates across all passages are compared and best chosen
How are no answers treated?
Point to parts of code in FARM and HF

Code snippet of number of candidate answers

See blog article for more on sliding window approach?

Variable Numbers of Documents
-----------------------------

How do we do this actually? Direct comparison of diff doc candidates? Top k?

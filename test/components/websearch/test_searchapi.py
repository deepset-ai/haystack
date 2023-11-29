import os
from unittest.mock import Mock, patch

import pytest
from requests import Timeout, RequestException, HTTPError

from haystack import Document
from haystack.components.websearch.searchapi import SearchApiError, SearchApiWebSearch


EXAMPLE_SEARCHAPI_RESPONSE = {
    "search_metadata": {
        "id": "search_Y16dWXw4JOrIwNjjvqoKNGlE",
        "status": "Success",
        "created_at": "2023-11-22T16:10:56Z",
        "request_time_taken": 1.98,
        "parsing_time_taken": 0.16,
        "total_time_taken": 2.15,
        "request_url": "https://www.google.com/search?q=Who+is+CEO+of+Microsoft%3F&oq=Who+is+CEO+of+Microsoft%3F&gl=us&hl=en&ie=UTF-8",
        "html_url": "https://www.searchapi.io/api/v1/searches/search_Y16dWXw4JOrIwNjjvqoKNGlE.html",
        "json_url": "https://www.searchapi.io/api/v1/searches/search_Y16dWXw4JOrIwNjjvqoKNGlE",
    },
    "search_parameters": {
        "engine": "google",
        "q": "Who is CEO of Microsoft?",
        "device": "desktop",
        "google_domain": "google.com",
        "hl": "en",
        "gl": "us",
    },
    "search_information": {
        "query_displayed": "Who is CEO of Microsoft?",
        "total_results": 429000000,
        "time_taken_displayed": 0.48,
    },
    "answer_box": {
        "type": "organic_result",
        "title": "Microsoft Corporation/CEO",
        "answer": "Satya Nadella",
        "answer_date": "Feb 4, 2014–",
        "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Satya+Nadella&stick=H4sIAAAAAAAAAONgVuLSz9U3KDQxqMjKesRoyi3w8sc9YSmdSWtOXmNU4-IKzsgvd80rySypFJLgYoOy-KR4uJC08Sxi5Q1OLKlMVPBLTEnNyUkEALvb1RBWAAAA&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQzIcDKAB6BAgyEAE",
        "snippet": "Microsoft CEO Satya Nadella speaks during the OpenAI DevDay event on November 06, 2023 in San Francisco, California.",
        "date": "1 day ago",
        "organic_result": {
            "title": "Microsoft CEO Satya Nadella's response to the OpenAI board ...",
            "link": "https://fortune.com/2023/11/21/microsoft-ceo-satya-nadella-openai-ceo-sam-altman-move-fast-fix-things/#:~:text=Microsoft%20CEO%20Satya%20Nadella%20speaks,2023%20in%20San%20Francisco%2C%20California.",
            "source": "Fortune",
            "domain": "fortune.com",
            "displayed_link": "https://fortune.com › 2023/11/21 › microsoft-ceo-satya-...",
        },
        "people_also_search_for": [
            {
                "title": "Sundar Pichai",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Sundar+Pichai&stick=H4sIAAAAAAAAAONgFuLQz9U3MCkuM1HiArEs01OKzU20-AJSi4rz84IzU1LLEyuLF7HyBpfmpSQWKQRkJmckZgIAJfaYezgAAAA&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQxA16BAgnEAQ",
            },
            {
                "title": "Steve Ballmer",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Steve+Ballmer&stick=H4sIAAAAAAAAAONgFuLQz9U3MCkuM1ECs8yTssu0-AJSi4rz84IzU1LLEyuLF7HyBpeklqUqOCXm5OSmFgEA31ogfDYAAAA&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQxA16BAgnEAY",
            },
            {
                "title": "Anupama Nadella",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Anupama+Nadella&stick=H4sIAAAAAAAAAONgFuLQz9U3MCkuM1Hi1U_XNzRMMjPMzTHMMtHiC0gtKs7PC85MSS1PrCxexMrvmFdakJibqOCXmJKak5MIAEx0yhM9AAAA&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQxA16BAgnEAg",
            },
            {
                "title": "Zain Nadella",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Zain+Nadella&stick=H4sIAAAAAAAAAONgFuLQz9U3MCkuM1Hi1U_XNzRMMjMyKCgsj9fiC0gtKs7PC85MSS1PrCxexMoTlZiZp-CXmJKak5MIANDRqOs6AAAA&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQxA16BAgnEAo",
            },
            {
                "title": "Bill Gates",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Bill+Gates&stick=H4sIAAAAAAAAAONgFuLQz9U3MCkuM1ECswzN80q0-AJSi4rz84IzU1LLEyuLF7FyOWXm5Ci4J5akFgMAF5_u-TMAAAA&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQxA16BAgnEAw",
            },
            {
                "title": "Shantanu Narayen",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Shantanu+Narayen&stick=H4sIAAAAAAAAAONgFuLQz9U3MCkuM1HiArGMzC0ts5O0-AJSi4rz84IzU1LLEyuLF7EKBGck5pUk5pUq-CUWJVam5gEA2xdRszsAAAA&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQxA16BAgnEA4",
            },
            {
                "title": "Paul Allen",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Paul+Allen&stick=H4sIAAAAAAAAAONgFuLQz9U3MCkuM1ECs0xLsnO1-AJSi4rz84IzU1LLEyuLF7FyBSSW5ig45uSk5gEA_4-yKDMAAAA&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQxA16BAgnEBA",
            },
        ],
    },
    "knowledge_graph": {
        "kgmid": "/m/0q40xjj",
        "knowledge_graph_type": "People",
        "title": "Satya Nadella",
        "type": "CEO of Microsoft",
        "description": "Satya Narayana Nadella is an Indian-American business executive. He is the executive chairman and CEO of Microsoft, succeeding Steve Ballmer in 2014 as CEO and John W. Thompson in 2021 as chairman.",
        "source": {"name": "Wikipedia", "link": "https://en.wikipedia.org/wiki/Satya_Nadella"},
        "born": "August 19, 1967 (age 56 years), Hyderabad, India",
        "born_links": [
            {
                "text": "Hyderabad, India",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Hyderabad&si=ALGXSlZS0YT-iRe81F2cKC9lM9KWTK4y0m5Atx8g9YliNNw2meVELJr66A46Jmr2L7YaEMWXarsN12T-Vg9bXBeu7mCHCG-SpT-gWQmluIDs5SvdST1r6rBUhcAOclNosjy4RgkGlWnecyHsBen2Ttz-NbCqTmTwwPK9ro0lfOFPb0CUDvLAkTbBXx4xNX7WWUJ19n0EWeuA&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQmxMoAHoECGUQAg",
            }
        ],
        "awards": "Padma Bhushan, CNN-IBN Indian of the Year Global Indian",
        "awards_links": [
            {
                "text": "Padma Bhushan",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Padma+Bhushan&si=ALGXSlYh1-GEPndq7qMo--O-TPixQtNN4JMroSxgItz5kq0stCyOa5BGWGIYt20KbMd-zdQdvwREsU7qSkWcyv0yzHS195H46le5meMq90to5z-nIHo4evgG3koKwps5uC-gu8Huemxmq6P1usjVEj5YR9okGopoUaOxuuyZP-isnQAmC6otzjnjf1O9jMuQObZmAnl2HH7coBXCHbIx1QvAHw1KZOYyJKPnYhWaYgqfQo7yF5BOVVLXvtr_8FhnFIxxl7f_V2B6&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQmxMoAHoECF8QAg",
            },
            {
                "text": "CNN-IBN Indian of the Year Global Indian",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=CNN-IBN+Indian+of+the+Year+Global+Indian&si=ALGXSlZZLz93Q5j8HVkpXyxpTaoqXw8cocmoi-DFAGsSj5diF8YzT48GvLer52UWWyGCjf3yeWD9YQzPqUV-LEVPLmirdkrJ_7HPexciHWOKnyaMVi0vXdKPSwvc8pE4fD3qmgVyw7qAFoNmy-T-U6OlosYKKVbf9CZnaOonmPhLRRFHGEEmKVtb_0FdKkXeUE2RIDgUJ1n1LWZoTeporPHOj4JfKSJADc-hymzzDEb5-uW3KxQtTdv_GJNMOoleFxqH9cvObQvW0_NvpfHZcThW9b_9g1BXjLfozVqh6hjRTbb40p5vu5e9Oi4sNqxtACf4Xoys_QX5&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQmxMoAXoECF8QAw",
            },
        ],
        "nominations": "CNN-IBN Indian of the Year Global Indian",
        "nominations_links": [
            {
                "text": "CNN-IBN Indian of the Year Global Indian",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=CNN-IBN+Indian+of+the+Year+Global+Indian&si=ALGXSlZZLz93Q5j8HVkpXyxpTaoqXw8cocmoi-DFAGsSj5diF8YzT48GvLer52UWWyGCjf3yeWD9YQzPqUV-LEVPLmirdkrJ_7HPexciHWOKnyaMVlh5LgokSYRM8a-Dib-kzfIaD6Uw_x_3lxo6j3NNKQbuBs4v4kkSCjL68joimLMo16eCX83PFrnvSsVKsgu6aFRoBYQt5p5NRofNfBXtVt2jzFVAWh23VsBHyyAxOuC2aQmgvKp-FGYymourIbHCdJ3rcx-Z&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQmxMoAHoECGIQAg",
            }
        ],
        "books": "Hit Refresh: The Quest to Rediscover Microsoft's Soul and Imagine a Better Future for Everyone",
        "books_links": [
            {
                "text": "Hit Refresh: The Quest to Rediscover Microsoft's Soul and Imagine a Better Future for Everyone",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Hit+Refresh&si=ALGXSlZZLz93Q5j8HVkpXyxpTaoqXw8cocmoi-DFAGsSj5diFzM3kSV8cu0gYZuy4n6At7XJ8qKh8mnRaXfDbxUaZoS_kPW87tGFHpw6B9zAS2a52vwJDx-fkzytheyPXaMQENZSl3bwqC9Nz3bqn7-Pglqh0Bik5Ow9AdVr2XI8mdVktN4SkCIaPE4qQfjAurt8rjUVyQzu3OFQx04nfPH3Gv7vP8aKqg%3D%3D&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQmxMoAHoECGEQAg",
            }
        ],
        "children": "Zain Nadella, Divya Nadella, Tara Nadella",
        "children_links": [
            {
                "text": "Zain Nadella",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Zain+Nadella&si=ALGXSlZZLz93Q5j8HVkpXyxpTaoqXw8cocmoi-DFAGsSj5diFxtguEvrMR1GmF2uy_-DLcVXwF5gosIuQudQPkad9bBUZxVKOG9PFdHXdEGQHrfXekG0E0x_raEKuDnHD6kk8_HfD3LZ57PWZ3Zyz0uhKPE15DfvA42IpAByWbms0fsgRw5IFCWwB5XMd3WM5U8KKsgeb_DmdoooQ_k3RrxO57jTcm5ZwgDlpBpGq0wj2Ksc2A65RQvA8NPJtpEqDcvEpJ4xWQ_tM_rHduCXRfsv9XFr84DzwA%3D%3D&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQmxMoAHoECGQQAg",
            },
            {
                "text": "Divya Nadella",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Divya+Nadella&si=ALGXSlZZLz93Q5j8HVkpXyxpTaoqXw8cocmoi-DFAGsSj5diFwYr_pFPi4_6apkHPz96V-E6wDawAGH_i6kZL7ZB-ETzV3LLESN1a8BgFguu3LOpz1qAQypmcVosQxCFWSJVexciDel34yrgWJmUu5bY2zzEmu1h95LQ35yUDkf6Mqcn-TiwyLu7OzGYkw6D9P4kNkS2D3gNPnRZb6vQJbqdayQg-wgn-LG2BmwR-RntneXFgSSZgotziGaY96UzeZ0zgRWYp6LAKlRqlTbeDeCbDDY2_VIWjQ%3D%3D&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQmxMoAXoECGQQAw",
            },
            {
                "text": "Tara Nadella",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Tara+Nadella&si=ALGXSlZZLz93Q5j8HVkpXyxpTaoqXw8cocmoi-DFAGsSj5diF465A_RPTnaELE1D-l5XgaKmBEpoAyayrOAdoXqBSLZ8Qu5UB1hBz6xLN4I1DdUSzqN0G0e9_8lfDbD_Qnx2uLJL_3XUNJ3gPrjCNvCyYeR9a9wkCnMBLchfUhVji9EHiobO4WgdWkxKd44YXHxfMBIYEek8OfbdUx9tplETPYtu7X1HRtGzqp8lXsQ6Vacj-aT7K6Xw0psbP4NXwHRQ71MYjLS-A5_VpSnitGScPsP-1m41Kg%3D%3D&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQmxMoAnoECGQQBA",
            },
        ],
        "education": "University of Wisconsin-Milwaukee (1990), MORE",
        "education_links": [
            {
                "text": "University of Wisconsin-Milwaukee",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=University+of+Wisconsin+Milwaukee&si=ALGXSlYh1-GEPndq7qMo--O-TPixQtNN4JMroSxgItz5kq0stDerMl6ZWDVLeoc_9LMxC6poOvWKyPlaxlQHHC7l9sV2e_sYZ2w92bas10emnFKqvF8PcMhCIIHCiTbdtg6nHIA-ihu0l0dNJtl3ZXuRejodvwikfjAsz-cGgFCLkxoi_eMM95SSZ77VXB0gP7fPTA6q__pIRK7T6ZfiSyM2xTbDt3YUvrWFmx5LBSJwRd2K1f0DK6sGaIa3ozdQOGvGXZkTOTLEG_a2ssbGBTX4MyU4cHmLsvW-Gfpq-makl3esSS7fQTc%3D&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQmxMoAHoECGAQAg",
            },
            {
                "text": "MORE",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=satya+nadella+education&stick=H4sIAAAAAAAAAOPgE-LSz9U3KDQxqMjK0pLOTrbSL0jNL8hJBVJFxfl5VqkppcmJJZn5eYtYxYsTSyoTFfISU1JzchIV4DIAcrWm-UUAAAA&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQ44YBKAF6BAhgEAM",
            },
        ],
        "full_name": "Satya Narayana Nadella",
        "profiles": [
            {"name": "LinkedIn", "link": "https://www.linkedin.com/in/satyanadella"},
            {"name": "Twitter", "link": "https://twitter.com/satyanadella"},
        ],
    },
    "organic_results": [
        {
            "position": 1,
            "title": "Satya Nadella - Stories",
            "link": "https://news.microsoft.com/exec/satya-nadella/",
            "source": "Microsoft",
            "domain": "news.microsoft.com",
            "displayed_link": "https://news.microsoft.com › exec › satya-nadella",
            "snippet": "Satya Nadella is Chairman and Chief Executive Officer of Microsoft. Before being named CEO in February 2014, Nadella held leadership roles in both ...",
            "snippet_highlighted_words": ["Satya Nadella"],
            "cached_page_link": "https://webcache.googleusercontent.com/search?q=cache:jTiZ69Cck7EJ:https://news.microsoft.com/exec/satya-nadella/&hl=en&gl=us",
        },
        {
            "position": 2,
            "title": "Satya Nadella",
            "link": "https://en.wikipedia.org/wiki/Satya_Nadella",
            "source": "Wikipedia",
            "domain": "en.wikipedia.org",
            "displayed_link": "https://en.wikipedia.org › wiki › Satya_Nadella",
            "snippet": "Satya Narayana Nadella is an Indian-American business executive. He is the executive chairman and CEO of Microsoft, succeeding Steve Ballmer in 2014 as CEO ...",
            "snippet_highlighted_words": ["Satya Narayana Nadella"],
            "sitelinks": {
                "inline": [
                    {
                        "title": "Manipal Institute of Technology",
                        "link": "https://en.wikipedia.org/wiki/Manipal_Institute_of_Technology",
                    },
                    {
                        "title": "University of Wisconsin",
                        "link": "https://en.wikipedia.org/wiki/University_of_Wisconsin%E2%80%93Milwaukee",
                    },
                    {"title": "S. Somasegar", "link": "https://en.wikipedia.org/wiki/S._Somasegar"},
                ]
            },
            "cached_page_link": "https://webcache.googleusercontent.com/search?q=cache:Tgw93hG0PnoJ:https://en.wikipedia.org/wiki/Satya_Nadella&hl=en&gl=us",
        },
        {
            "position": 3,
            "title": "Satya Nadella",
            "link": "https://www.linkedin.com/in/satyanadella",
            "source": "LinkedIn · Satya Nadella",
            "domain": "www.linkedin.com",
            "displayed_link": "10.5M+ followers",
            "snippet": "As chairman and CEO of Microsoft, I define my mission and that of my company as empowering… | Learn more about Satya Nadella's work experience, education, ...",
            "snippet_highlighted_words": ["Satya Nadella's"],
        },
        {
            "position": 4,
            "title": "Who is Satya Nadella, Family, Salary, Education, Net Worth ...",
            "link": "https://www.business-standard.com/about/who-is-satya-nadella",
            "source": "Business Standard",
            "domain": "www.business-standard.com",
            "displayed_link": "https://www.business-standard.com › about › who-is-s...",
            "snippet": "Satya Narayana Nadella is the chief executive officer (CEO) of Microsoft. Under him, Microsoft has more cloud computing revenue than Google, more subscribers ...",
            "snippet_highlighted_words": ["Satya Narayana Nadella"],
            "cached_page_link": "https://webcache.googleusercontent.com/search?q=cache:yQ0bmLSmP8gJ:https://www.business-standard.com/about/who-is-satya-nadella&hl=en&gl=us",
        },
        {
            "position": 5,
            "title": "Satya Nadella (@satyanadella) / X",
            "link": "https://twitter.com/satyanadella",
            "source": "Twitter · satyanadella",
            "domain": "twitter.com",
            "displayed_link": "3.1M+ followers",
            "snippet": "Chairman and CEO of Microsoft Corporation.",
            "snippet_highlighted_words": ["CEO of Microsoft"],
            "cached_page_link": "https://webcache.googleusercontent.com/search?q=cache:dEJiGKzwLfkJ:https://twitter.com/satyanadella&hl=en&gl=us",
        },
        {
            "position": 6,
            "title": "Satya Nadella | Biography & Facts",
            "link": "https://www.britannica.com/biography/Satya-Nadella",
            "source": "Britannica",
            "domain": "www.britannica.com",
            "displayed_link": "https://www.britannica.com › biography › Satya-Nadella",
            "snippet": "Satya Nadella (born August 19, 1967, Hyderabad, India) Indian-born business executive who was CEO of the computer software company Microsoft (2014– ).",
            "snippet_highlighted_words": ["Satya Nadella"],
            "cached_page_link": "https://webcache.googleusercontent.com/search?q=cache:a0S8ke4I9qgJ:https://www.britannica.com/biography/Satya-Nadella&hl=en&gl=us",
        },
        {
            "position": 7,
            "title": "Satya Nadella",
            "link": "https://www.forbes.com/profile/satya-nadella/",
            "source": "Forbes",
            "domain": "www.forbes.com",
            "displayed_link": "https://www.forbes.com › profile › satya-nadella",
            "snippet": "Satya Nadella replaced billionaire Steve Ballmer as Microsoft CEO in 2014. Prior to that, Nadella was Microsoft EVP of the cloud and enterprise group.",
            "snippet_highlighted_words": ["Satya Nadella"],
            "cached_page_link": "https://webcache.googleusercontent.com/search?q=cache:q_CXTYNnHSMJ:https://www.forbes.com/profile/satya-nadella/&hl=en&gl=us",
        },
        {
            "position": 8,
            "title": "5 Facts You Didn't Know About Microsoft CEO Satya Nadella",
            "link": "https://in.benzinga.com/content/35911756/5-facts-you-didnt-know-about-microsoft-ceo-satya-nadella",
            "source": "Benzinga",
            "domain": "in.benzinga.com",
            "displayed_link": "https://in.benzinga.com › content › 5-facts-you-didnt-...",
            "snippet": "Satya Nadella's journey at Microsoft underscores the importance of diverse experiences in shaping effective and empathetic leadership in the ...",
            "snippet_highlighted_words": ["Satya Nadella's"],
            "date": "8 hours ago",
            "cached_page_link": "https://webcache.googleusercontent.com/search?q=cache:hCbtJUTgvEQJ:https://in.benzinga.com/content/35911756/5-facts-you-didnt-know-about-microsoft-ceo-satya-nadella&hl=en&gl=us",
        },
        {
            "position": 9,
            "title": "Microsoft CEO Satya Nadella: Q&A - The Wall Street Journal",
            "link": "https://www.wsj.com/video/microsoft-ceo-satya-nadella-qa/41D02815-935C-421D-8021-5E1BFD3DDE84",
            "source": "Wall Street Journal",
            "domain": "www.wsj.com",
            "displayed_link": "https://www.wsj.com › video › microsoft-ceo-satya-nadel...",
            "snippet": "Microsoft CEO Satya Nadella talks about his biggest accomplishment, how to make successful acquisitions and how the tech industry could improve its image ...",
            "snippet_highlighted_words": ["Microsoft CEO"],
            "video": {"source": "The Wall Street Journal", "channel": "The Wall Street Journal", "date": "Feb 1, 2019"},
        },
    ],
    "related_questions": [
        {
            "question": "Who is the real CEO of Microsoft?",
            "answer": "Satya Nadella is Chairman and Chief Executive Officer of Microsoft.",
            "answer_highlight": "Satya Nadella",
            "source": {
                "title": "Satya Nadella - Stories - Microsoft News",
                "link": "https://news.microsoft.com/exec/satya-nadella/#:~:text=Satya%20Nadella%20is%20Chairman%20and%20Chief%20Executive%20Officer%20of%20Microsoft.",
                "source": "Microsoft",
                "domain": "news.microsoft.com",
                "displayed_link": "https://news.microsoft.com › exec › satya-nadella",
            },
            "search": {
                "title": "Search for: Who is the real CEO of Microsoft?",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Who+is+the+real+CEO+of+Microsoft%3F&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQzmd6BAgeEAY",
            },
        },
        {
            "question": "Who is the CEO of Microsoft 2023?",
            "answer": "Microsoft Corp. chief executive officer Satya Nadella signaled that he'd be open to Sam Altman going back to OpenAI, rather than joining his company as part of a surprise move announced over the weekend.",
            "date": "2 days ago",
            "source": {
                "title": "Microsoft CEO Satya Nadella signals willingness to have Sam Altman ...",
                "link": "https://economictimes.indiatimes.com/tech/technology/microsoft-ceo-satya-nadella-signals-willingness-to-have-sam-altman-rejoin-openai/articleshow/105370026.cms#:~:text=Microsoft%20Corp.%20chief%20executive%20officer,move%20announced%20over%20the%20weekend.",
                "source": "indiatimes.com",
                "domain": "economictimes.indiatimes.com",
                "displayed_link": "https://economictimes.indiatimes.com › tech › articleshow",
            },
            "search": {
                "title": "Search for: Who is the CEO of Microsoft 2023?",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Who+is+the+CEO+of+Microsoft+2023%3F&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQzmd6BAgcEAY",
            },
        },
        {
            "question": "How many degrees does Satya Nadella have?",
            "answer": "He earned a bachelor's degree in electrical engineering from Mangalore University, a master's degree in computer science from the University of Wisconsin – Milwaukee and a master's degree in business administration from the University of Chicago.",
            "source": {
                "title": "Satya Nadella - Institutional - BlackRock",
                "link": "https://www.blackrock.com/institutions/en-zz/biographies/satya-nadella#:~:text=He%20earned%20a%20bachelor's%20degree,from%20the%20University%20of%20Chicago.",
                "source": "blackrock.com",
                "domain": "www.blackrock.com",
                "displayed_link": "https://www.blackrock.com › en-zz › biographies › satya...",
            },
            "search": {
                "title": "Search for: How many degrees does Satya Nadella have?",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=How+many+degrees+does+Satya+Nadella+have%3F&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQzmd6BAgdEAY",
            },
        },
        {
            "question": "How old is Satya Nadella?",
            "answer_highlight": "56 years (August 19, 1967)",
            "entity": {"subject": "Satya Nadella", "attribute": "Age", "value": "56 years (August 19, 1967)"},
            "search": {
                "title": "Search for: How old is Satya Nadella?",
                "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=How+old+is+Satya+Nadella%3F&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQzmd6BAgREAY",
            },
        },
    ],
    "related_searches": [
        {
            "query": "Who is ceo of microsoft wife",
            "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Who+is+ceo+of+microsoft+wife&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQ1QJ6BAhWEAE",
        },
        {
            "query": "Who is ceo of microsoft and microsoft",
            "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Who+is+ceo+of+microsoft+and+microsoft&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQ1QJ6BAhVEAE",
        },
        {
            "query": "Who is ceo of microsoft wikipedia",
            "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Who+is+ceo+of+microsoft+wikipedia&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQ1QJ6BAhUEAE",
        },
        {
            "query": "microsoft founder",
            "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Microsoft+founder&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQ1QJ6BAhSEAE",
        },
        {
            "query": "Who is ceo of microsoft 2020",
            "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Who+is+ceo+of+microsoft+2020&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQ1QJ6BAhTEAE",
        },
        {
            "query": "satya nadella net worth",
            "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=Satya+Nadella+net+worth&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQ1QJ6BAhREAE",
        },
        {
            "query": "ceo of microsoft salary",
            "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=CEO+of+Microsoft+salary&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQ1QJ6BAhQEAE",
        },
        {
            "query": "ceo of apple",
            "link": "https://www.google.com/search?sca_esv=584620230&gl=us&hl=en&q=CEO+of+Apple&sa=X&ved=2ahUKEwi89re3_9eCAxU4IUQIHfHeB6MQ1QJ6BAhXEAE",
        },
    ],
}


@pytest.fixture
def mock_searchapi_search_result():
    with patch("haystack.components.websearch.searchapi.requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=200, json=lambda: EXAMPLE_SEARCHAPI_RESPONSE)
        yield mock_get


class TestSearchApiSearchAPI:
    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("SEARCHAPI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="SearchApiWebSearch expects an API key"):
            SearchApiWebSearch()

    def test_to_dict(self):
        component = SearchApiWebSearch(
            api_key="api_key", top_k=10, allowed_domains=["testdomain.com"], search_params={"param": "test params"}
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.websearch.searchapi.SearchApiWebSearch",
            "init_parameters": {
                "top_k": 10,
                "allowed_domains": ["testdomain.com"],
                "search_params": {"param": "test params"},
            },
        }

    @pytest.mark.parametrize("top_k", [1, 5, 7])
    def test_web_search_top_k(self, mock_searchapi_search_result, top_k: int):
        ws = SearchApiWebSearch(api_key="api_key", top_k=top_k)
        results = ws.run(query="Who is CEO of Microsoft?")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == top_k
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)

    @patch("requests.get")
    def test_timeout_error(self, mock_get):
        mock_get.side_effect = Timeout
        ws = SearchApiWebSearch(api_key="api_key")

        with pytest.raises(TimeoutError):
            ws.run(query="Who is CEO of Microsoft?")

    @patch("requests.get")
    def test_request_exception(self, mock_get):
        mock_get.side_effect = RequestException
        ws = SearchApiWebSearch(api_key="api_key")

        with pytest.raises(SearchApiError):
            ws.run(query="Who is CEO of Microsoft?")

    @patch("requests.get")
    def test_bad_response_code(self, mock_get):
        mock_response = mock_get.return_value
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPError
        ws = SearchApiWebSearch(api_key="api_key")

        with pytest.raises(SearchApiError):
            ws.run(query="Who is CEO of Microsoft?")

    @pytest.mark.skipif(
        not os.environ.get("SEARCHAPI_API_KEY", None),
        reason="Export an env var called SEARCHAPI_API_KEY containing the SearchApi API key to run this test.",
    )
    @pytest.mark.integration
    def test_web_search(self):
        ws = SearchApiWebSearch(api_key=os.environ.get("SEARCHAPI_API_KEY", None), top_k=10)
        results = ws.run(query="Who is CEO of Microsoft?")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == 10
        assert all(isinstance(doc, Document) for doc in results)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)

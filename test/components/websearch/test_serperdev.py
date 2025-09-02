# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import Mock, patch

import pytest
from requests import HTTPError, RequestException, Timeout

from haystack import Document
from haystack.components.websearch.serper_dev import SerperDevError, SerperDevWebSearch
from haystack.utils.auth import Secret

EXAMPLE_SERPERDEV_RESPONSE = {
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
            "snippet": "Looks like Olivia Wilde and Jason Sudeikis are starting 2023 on good terms. Amid their highly "
            "publicized custody battle – and the actress' ...",
            "date": "Jan 29, 2023",
            "position": 1,
        },
        {
            "title": "Olivia Wilde Is 'Quietly Dating' Again Following Harry Styles Split: 'He Makes Her Happy'",
            "link": "https://www.yahoo.com/now/olivia-wilde-quietly-dating-again-183844364.html",
            "snippet": "Olivia Wilde is “quietly dating again” following her November 2022 split from Harry Styles, "
            "a source exclusively tells Life & Style.",
            "date": "Feb 10, 2023",
            "position": 2,
        },
        {
            "title": "Olivia Wilde and Harry Styles' Relationship Timeline: The Way They Were - Us Weekly",
            "link": "https://www.usmagazine.com/celebrity-news/pictures/olivia-wilde-and-harry-styles-relationship-timeline/",
            "snippet": "Olivia Wilde started dating Harry Styles after ending her years-long engagement to Jason "
            "Sudeikis — see their relationship timeline.",
            "date": "Mar 10, 2023",
            "imageUrl": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSgTcalNFvptTbYBiDXX55s8yCGfn6F1qbed9DAN16LvynTr9GayK5SPmY&s",
            "position": 3,
        },
        {
            "title": "Olivia Wilde Is 'Ready to Date Again' After Harry Styles Split - Us Weekly",
            "link": "https://www.usmagazine.com/celebrity-news/news/olivia-wilde-is-ready-to-date-again-after-harry-styles-split/",
            "snippet": "Ready for love! Olivia Wilde is officially back on the dating scene following her split from "
            "her ex-boyfriend, Harry Styles.",
            "date": "Mar 1, 2023",
            "imageUrl": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRCRAeRy5sVE631ZctzbzuOF70xkIOHaTvh2K7dYvdiVBwALiKrIjpscok&s",
            "position": 4,
        },
        {
            "title": "Harry Styles and Olivia Wilde's Definitive Relationship Timeline - Harper's Bazaar",
            "link": "https://www.harpersbazaar.com/celebrity/latest/a35172115/harry-styles-olivia-wilde-relationship-timeline/",
            "snippet": "November 2020: News breaks about Olivia splitting from fiancé Jason Sudeikis. ... "
            "In mid-November, news breaks of Olivia Wilde's split from Jason ...",
            "date": "Feb 23, 2023",
            "imageUrl": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRRqw3fvZOIGHEepxCc7yFAWYsS_v_1H6X-4nxyFJxdfRuFQw_BrI6JVzI&s",
            "position": 5,
        },
        {
            "title": "Harry Styles and Olivia Wilde's Relationship Timeline - People",
            "link": "https://people.com/music/harry-styles-olivia-wilde-relationship-timeline/",
            "snippet": "Harry Styles and Olivia Wilde first met on the set of Don't Worry Darling and stepped out as "
            "a couple in January 2021. Relive all their biggest relationship ...",
            "position": 6,
        },
        {
            "title": "Jason Sudeikis and Olivia Wilde's Relationship Timeline - People",
            "link": "https://people.com/movies/jason-sudeikis-olivia-wilde-relationship-timeline/",
            "snippet": "Jason Sudeikis and Olivia Wilde ended their engagement of seven years in 2020. Here's a "
            "complete timeline of their relationship.",
            "date": "Mar 24, 2023",
            "imageUrl": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSleZoXusQyJJe2WMgIuck_cVaJ8AE0_hU2QxsXzYvKANi55UQlv82yAVI&s",
            "position": 7,
        },
        {
            "title": "Olivia Wilde's anger at ex-boyfriend Harry Styles: She resents him and thinks he was using her "
            "| Marca",
            "link": "https://www.marca.com/en/lifestyle/celebrities/2023/02/23/63f779a4e2704e8d988b4624.html",
            "snippet": "The two started dating after Wilde split up with actor Jason Sudeikisin 2020. However, their "
            "relationship came to an end last November.",
            "date": "Feb 23, 2023",
            "imageUrl": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQBgJF2mSnIWCvPrqUqM4WTI9xPNWPyLvHuune85swpB1yE_G8cy_7KRh0&s",
            "position": 8,
        },
        {
            "title": "Olivia Wilde's dating history: Who has the actress dated? | The US Sun",
            "link": "https://www.the-sun.com/entertainment/5221040/olivia-wildes-dating-history/",
            "snippet": "AMERICAN actress Olivia Wilde started dating Harry Styles in January 2021 after breaking off "
            "her engagement the year prior.",
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
def mock_serper_dev_search_result():
    with patch("haystack.components.websearch.serper_dev.requests") as mock_run:
        mock_run.post.return_value = Mock(status_code=200, json=lambda: EXAMPLE_SERPERDEV_RESPONSE)
        yield mock_run


@pytest.fixture
def mock_serper_dev_search_result_no_snippet():
    resp = {**EXAMPLE_SERPERDEV_RESPONSE}
    del resp["organic"][0]["snippet"]
    with patch("haystack.components.websearch.serper_dev.requests") as mock_run:
        mock_run.post.return_value = Mock(status_code=200, json=lambda: resp)
        yield mock_run


class TestSerperDevSearchAPI:
    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("SERPERDEV_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            SerperDevWebSearch()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("SERPERDEV_API_KEY", "test-api-key")
        component = SerperDevWebSearch(top_k=10, allowed_domains=["test.com"], search_params={"param": "test"})
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.websearch.serper_dev.SerperDevWebSearch",
            "init_parameters": {
                "api_key": {"env_vars": ["SERPERDEV_API_KEY"], "strict": True, "type": "env_var"},
                "top_k": 10,
                "allowed_domains": ["test.com"],
                "exclude_subdomains": False,
                "search_params": {"param": "test"},
            },
        }

    @pytest.mark.parametrize("top_k", [1, 5, 7])
    def test_web_search_top_k(self, mock_serper_dev_search_result, top_k: int):
        ws = SerperDevWebSearch(api_key=Secret.from_token("test-api-key"), top_k=top_k)
        results = ws.run(query="Who is the boyfriend of Olivia Wilde?")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == top_k
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)

    def test_no_snippet(self, mock_serper_dev_search_result_no_snippet):
        ws = SerperDevWebSearch(api_key=Secret.from_token("test-api-key"), top_k=1)
        ws.run(query="Who is the boyfriend of Olivia Wilde?")

    @patch("requests.post")
    def test_timeout_error(self, mock_post):
        mock_post.side_effect = Timeout
        ws = SerperDevWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(TimeoutError):
            ws.run(query="Who is the boyfriend of Olivia Wilde?")

    @patch("requests.post")
    def test_request_exception(self, mock_post):
        mock_post.side_effect = RequestException
        ws = SerperDevWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(SerperDevError):
            ws.run(query="Who is the boyfriend of Olivia Wilde?")

    @patch("requests.post")
    def test_bad_response_code(self, mock_post):
        mock_response = mock_post.return_value
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPError
        ws = SerperDevWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(SerperDevError):
            ws.run(query="Who is the boyfriend of Olivia Wilde?")

    @pytest.mark.skipif(
        not os.environ.get("SERPERDEV_API_KEY", None),
        reason="Export an env var called SERPERDEV_API_KEY containing the SerperDev API key to run this test.",
    )
    @pytest.mark.integration
    def test_web_search(self):
        ws = SerperDevWebSearch(top_k=10)
        results = ws.run(query="Who is the boyfriend of Olivia Wilde?")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == 10
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)

    def test_exclude_subdomains_filtering(self, monkeypatch):
        """Test that exclude_subdomains parameter properly filters results."""
        monkeypatch.setenv("SERPERDEV_API_KEY", "test-api-key")

        # Mock response with mixed domains and subdomains
        mock_response = {
            "organic": [
                {
                    "title": "Main domain result",
                    "link": "https://example.com/page1",
                    "snippet": "Content from main domain",
                },
                {
                    "title": "Subdomain result 1",
                    "link": "https://blog.example.com/post1",
                    "snippet": "Content from blog subdomain",
                },
                {
                    "title": "Subdomain result 2",
                    "link": "https://shop.example.com/product1",
                    "snippet": "Content from shop subdomain",
                },
                {
                    "title": "Different domain result",
                    "link": "https://other.com/page1",
                    "snippet": "Content from different domain",
                },
            ]
        }

        with patch("haystack.components.websearch.serper_dev.requests") as mock_requests:
            mock_requests.post.return_value = Mock(status_code=200, json=lambda: mock_response)

            # Test with exclude_subdomains=False (default behavior)
            ws_include = SerperDevWebSearch(
                api_key=Secret.from_token("test-api-key"), allowed_domains=["example.com"], exclude_subdomains=False
            )
            results_include = ws_include.run(query="test query")

            # Should include main domain and subdomains but exclude other domains
            assert len(results_include["documents"]) == 3  # example.com + 2 subdomains
            assert len(results_include["links"]) == 3

            included_links = results_include["links"]
            assert "https://example.com/page1" in included_links
            assert "https://blog.example.com/post1" in included_links
            assert "https://shop.example.com/product1" in included_links
            assert "https://other.com/page1" not in included_links

            # Test with exclude_subdomains=True
            ws_exclude = SerperDevWebSearch(
                api_key=Secret.from_token("test-api-key"), allowed_domains=["example.com"], exclude_subdomains=True
            )
            results_exclude = ws_exclude.run(query="test query")

            # Should only include main domain, exclude subdomains and other domains
            assert len(results_exclude["documents"]) == 1  # only example.com
            assert len(results_exclude["links"]) == 1

            excluded_links = results_exclude["links"]
            assert "https://example.com/page1" in excluded_links
            assert "https://blog.example.com/post1" not in excluded_links
            assert "https://shop.example.com/product1" not in excluded_links
            assert "https://other.com/page1" not in excluded_links

    def test_is_domain_allowed_helper_method(self, monkeypatch):
        """Test the _is_domain_allowed helper method directly."""
        monkeypatch.setenv("SERPERDEV_API_KEY", "test-api-key")

        # Test with exclude_subdomains=False
        ws_include = SerperDevWebSearch(
            api_key=Secret.from_token("test-api-key"),
            allowed_domains=["example.com", "test.org"],
            exclude_subdomains=False,
        )

        # Should allow main domains and subdomains
        assert ws_include._is_domain_allowed("https://example.com/page") == True
        assert ws_include._is_domain_allowed("https://blog.example.com/post") == True
        assert ws_include._is_domain_allowed("https://shop.example.com/product") == True
        assert ws_include._is_domain_allowed("https://test.org/page") == True
        assert ws_include._is_domain_allowed("https://sub.test.org/page") == True
        assert ws_include._is_domain_allowed("https://other.com/page") == False

        # Test with exclude_subdomains=True
        ws_exclude = SerperDevWebSearch(
            api_key=Secret.from_token("test-api-key"),
            allowed_domains=["example.com", "test.org"],
            exclude_subdomains=True,
        )

        # Should only allow exact domain matches
        assert ws_exclude._is_domain_allowed("https://example.com/page") == True
        assert ws_exclude._is_domain_allowed("https://blog.example.com/post") == False
        assert ws_exclude._is_domain_allowed("https://shop.example.com/product") == False
        assert ws_exclude._is_domain_allowed("https://test.org/page") == True
        assert ws_exclude._is_domain_allowed("https://sub.test.org/page") == False
        assert ws_exclude._is_domain_allowed("https://other.com/page") == False

        # Test with no allowed_domains (should allow all)
        ws_no_filter = SerperDevWebSearch(api_key=Secret.from_token("test-api-key"), allowed_domains=None)
        assert ws_no_filter._is_domain_allowed("https://any.domain.com/page") == True

import pytest

from haystack.preview.components.file_converters.markdown import MarkdownToTextDocument


class TestMarkdownToTextDocument:
    @pytest.mark.unit
    def test_markdown_converter(self, preview_samples_path):
        converter = MarkdownToTextDocument()
        results = converter.run(
            paths=[preview_samples_path / "markdown" / "sample.md", preview_samples_path / "markdown" / "sample.md"]
        )
        assert results["documents"][0].text.startswith("\nWhat to build with Haystack")
        assert "# git clone https://github.com/deepset-ai/haystack.git" not in results["documents"][0].text

    @pytest.mark.unit
    def test_markdown_converter_headline_extraction(self, preview_samples_path):
        expected_headlines = [
            ("What to build with Haystack", 1),
            ("Core Features", 1),
            ("Quick Demo", 1),
            ("2nd level headline for testing purposes", 2),
            ("3rd level headline for testing purposes", 3),
        ]

        converter = MarkdownToTextDocument(extract_headlines=True, remove_code_snippets=False)
        results = converter.run(paths=[preview_samples_path / "markdown" / "sample.md"])

        # Check if correct number of headlines are extracted
        assert len(results["documents"][0].metadata["headlines"]) == 5
        for extracted_headline, (expected_headline, expected_level) in zip(
            results["documents"][0].metadata["headlines"], expected_headlines
        ):
            # Check if correct headline and level is extracted
            assert extracted_headline["headline"] == expected_headline
            assert extracted_headline["level"] == expected_level

            # Check if correct start_idx is extracted
            start_idx = extracted_headline["start_idx"]
            hl_len = len(extracted_headline["headline"])
            assert extracted_headline["headline"] == results["documents"][0].text[start_idx : start_idx + hl_len]

    @pytest.mark.unit
    def test_markdown_converter_frontmatter_to_meta(self, preview_samples_path):
        converter = MarkdownToTextDocument(add_frontmatter_to_meta=True)
        results = converter.run(paths=[preview_samples_path / "markdown" / "sample.md"])
        assert results["documents"][0].metadata["type"] == "intro"
        assert results["documents"][0].metadata["date"] == "1.1.2023"

    @pytest.mark.unit
    def test_markdown_converter_remove_code_snippets(self, preview_samples_path):
        converter = MarkdownToTextDocument(remove_code_snippets=False)
        results = converter.run(paths=[preview_samples_path / "markdown" / "sample.md"])
        assert results["documents"][0].text.startswith("pip install farm-haystack")

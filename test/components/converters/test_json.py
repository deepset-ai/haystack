# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from unittest.mock import patch
from pathlib import Path
import logging

import pytest

from haystack.components.converters import JSONConverter
from haystack.dataclasses import ByteStream


test_data = [
    {
        "year": "1997",
        "category": "literature",
        "laureates": [
            {
                "id": "674",
                "firstname": "Dario",
                "surname": "Fokin",
                "motivation": "who emulates the jesters of the Middle Ages in scourging authority and upholding the dignity of the downtrodden",
                "share": "1",
            }
        ],
    },
    {
        "year": "1986",
        "category": "medicine",
        "laureates": [
            {
                "id": "434",
                "firstname": "Stanley",
                "surname": "Cohen",
                "motivation": "for their discoveries of growth factors",
                "share": "2",
            },
            {
                "id": "435",
                "firstname": "Rita",
                "surname": "Levi-Montalcini",
                "motivation": "for their discoveries of growth factors",
                "share": "2",
            },
        ],
    },
    {
        "year": "1938",
        "category": "physics",
        "laureates": [
            {
                "id": "46",
                "firstname": "Enrico",
                "surname": "Fermi",
                "motivation": "for his demonstrations of the existence of new radioactive elements produced by neutron irradiation, and for his related discovery of nuclear reactions brought about by slow neutrons",
                "share": "1",
            }
        ],
    },
]


def test_init_without_jq_schema_and_content_key():
    with pytest.raises(
        ValueError, match="No `jq_schema` nor `content_key` specified. Set either or both to extract data."
    ):
        JSONConverter()


@patch("haystack.components.converters.json.jq_import")
def test_init_without_jq_schema_and_missing_dependency(jq_import):
    converter = JSONConverter(content_key="foo")
    jq_import.check.assert_not_called()
    assert converter._jq_schema is None
    assert converter._content_key == "foo"
    assert converter._meta_fields is None


@patch("haystack.components.converters.json.jq_import")
def test_init_with_jq_schema_and_missing_dependency(jq_import):
    jq_import.check.side_effect = ImportError
    with pytest.raises(ImportError):
        JSONConverter(jq_schema=".laureates[].motivation")


def test_init_with_jq_schema():
    converter = JSONConverter(jq_schema=".")
    assert converter._jq_schema == "."
    assert converter._content_key is None
    assert converter._meta_fields is None


def test_to_dict():
    converter = JSONConverter(
        jq_schema=".laureates[]", content_key="motivation", extra_meta_fields={"firstname", "surname"}
    )

    assert converter.to_dict() == {
        "type": "haystack.components.converters.json.JSONConverter",
        "init_parameters": {
            "content_key": "motivation",
            "jq_schema": ".laureates[]",
            "extra_meta_fields": {"firstname", "surname"},
            "store_full_path": False,
        },
    }


def test_from_dict():
    data = {
        "type": "haystack.components.converters.json.JSONConverter",
        "init_parameters": {
            "content_key": "motivation",
            "jq_schema": ".laureates[]",
            "extra_meta_fields": ["firstname", "surname"],
            "store_full_path": True,
        },
    }
    converter = JSONConverter.from_dict(data)

    assert converter._jq_schema == ".laureates[]"
    assert converter._content_key == "motivation"
    assert converter._meta_fields == ["firstname", "surname"]


def test_run(tmpdir):
    first_test_file = Path(tmpdir / "first_test_file.json")
    second_test_file = Path(tmpdir / "second_test_file.json")

    first_test_file.write_text(json.dumps(test_data[0]), "utf-8")
    second_test_file.write_text(json.dumps(test_data[1]), "utf-8")
    byte_stream = ByteStream.from_string(json.dumps(test_data[2]))

    sources = [str(first_test_file), second_test_file, byte_stream]

    converter = JSONConverter(jq_schema='.laureates[] | .firstname + " " + .surname + " " + .motivation')
    result = converter.run(sources=sources)
    assert len(result) == 1
    assert len(result["documents"]) == 4
    assert (
        result["documents"][0].content
        == "Dario Fokin who emulates the jesters of the Middle Ages in scourging authority and "
        "upholding the dignity of the downtrodden"
    )
    assert result["documents"][0].meta == {"file_path": os.path.basename(first_test_file)}
    assert result["documents"][1].content == "Stanley Cohen for their discoveries of growth factors"
    assert result["documents"][1].meta == {"file_path": os.path.basename(second_test_file)}
    assert result["documents"][2].content == "Rita Levi-Montalcini for their discoveries of growth factors"
    assert result["documents"][2].meta == {"file_path": os.path.basename(second_test_file)}
    assert (
        result["documents"][3].content == "Enrico Fermi for his demonstrations of the existence of new "
        "radioactive elements produced by neutron irradiation, and for his related discovery of nuclear "
        "reactions brought about by slow neutrons"
    )
    assert result["documents"][3].meta == {}


def test_run_with_store_full_path_false(tmpdir):
    """
    Test if the component runs correctly with store_full_path=False
    """
    first_test_file = Path(tmpdir / "first_test_file.json")
    second_test_file = Path(tmpdir / "second_test_file.json")

    first_test_file.write_text(json.dumps(test_data[0]), "utf-8")
    second_test_file.write_text(json.dumps(test_data[1]), "utf-8")
    byte_stream = ByteStream.from_string(json.dumps(test_data[2]))

    sources = [str(first_test_file), second_test_file, byte_stream]

    converter = JSONConverter(
        jq_schema='.laureates[] | .firstname + " " + .surname + " " + .motivation', store_full_path=False
    )
    result = converter.run(sources=sources)
    assert len(result) == 1
    assert len(result["documents"]) == 4
    assert (
        result["documents"][0].content
        == "Dario Fokin who emulates the jesters of the Middle Ages in scourging authority and "
        "upholding the dignity of the downtrodden"
    )
    assert result["documents"][0].meta == {"file_path": "first_test_file.json"}
    assert result["documents"][1].content == "Stanley Cohen for their discoveries of growth factors"
    assert result["documents"][1].meta == {"file_path": "second_test_file.json"}
    assert result["documents"][2].content == "Rita Levi-Montalcini for their discoveries of growth factors"
    assert result["documents"][2].meta == {"file_path": "second_test_file.json"}
    assert (
        result["documents"][3].content == "Enrico Fermi for his demonstrations of the existence of new "
        "radioactive elements produced by neutron irradiation, and for his related discovery of nuclear "
        "reactions brought about by slow neutrons"
    )
    assert result["documents"][3].meta == {}


def test_run_with_non_json_file(tmpdir, caplog):
    test_file = Path(tmpdir / "test_file.md")
    test_file.write_text("This is not a JSON file.", "utf-8")

    sources = [test_file]
    converter = JSONConverter(".laureates | .motivation")

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        result = converter.run(sources=sources)

    records = caplog.records
    assert len(records) == 1
    assert (
        records[0].msg
        == f"Failed to extract text from {test_file}. Skipping it. Error: parse error: Invalid numeric literal at line 1, column 5"
    )
    assert result == {"documents": []}


def test_run_with_bad_filter(tmpdir, caplog):
    test_file = Path(tmpdir / "test_file.json")
    test_file.write_text(json.dumps(test_data[0]), "utf-8")

    sources = [test_file]
    converter = JSONConverter(".laureates | .motivation")

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        result = converter.run(sources=sources)

    records = caplog.records
    assert len(records) == 1
    assert (
        records[0].msg
        == f'Failed to extract text from {test_file}. Skipping it. Error: Cannot index array with string "motivation"'
    )
    assert result == {"documents": []}


def test_run_with_single_meta(tmpdir):
    first_test_file = Path(tmpdir / "first_test_file.json")
    second_test_file = Path(tmpdir / "second_test_file.json")

    first_test_file.write_text(json.dumps(test_data[0]), "utf-8")
    second_test_file.write_text(json.dumps(test_data[1]), "utf-8")
    byte_stream = ByteStream.from_string(json.dumps(test_data[2]))

    sources = [str(first_test_file), second_test_file, byte_stream]
    meta = {"creation_date": "1945-05-25T00:00:00"}
    converter = JSONConverter(jq_schema='.laureates[] | .firstname + " " + .surname + " " + .motivation')
    result = converter.run(sources=sources, meta=meta)
    assert len(result) == 1
    assert len(result["documents"]) == 4
    assert (
        result["documents"][0].content
        == "Dario Fokin who emulates the jesters of the Middle Ages in scourging authority and "
        "upholding the dignity of the downtrodden"
    )
    assert result["documents"][0].meta == {
        "file_path": os.path.basename(first_test_file),
        "creation_date": "1945-05-25T00:00:00",
    }
    assert result["documents"][1].content == "Stanley Cohen for their discoveries of growth factors"
    assert result["documents"][1].meta == {
        "file_path": os.path.basename(second_test_file),
        "creation_date": "1945-05-25T00:00:00",
    }
    assert result["documents"][2].content == "Rita Levi-Montalcini for their discoveries of growth factors"
    assert result["documents"][2].meta == {
        "file_path": os.path.basename(second_test_file),
        "creation_date": "1945-05-25T00:00:00",
    }
    assert (
        result["documents"][3].content == "Enrico Fermi for his demonstrations of the existence of new "
        "radioactive elements produced by neutron irradiation, and for his related discovery of nuclear "
        "reactions brought about by slow neutrons"
    )
    assert result["documents"][3].meta == {"creation_date": "1945-05-25T00:00:00"}


def test_run_with_meta_list(tmpdir):
    first_test_file = Path(tmpdir / "first_test_file.json")
    second_test_file = Path(tmpdir / "second_test_file.json")

    first_test_file.write_text(json.dumps(test_data[0]), "utf-8")
    second_test_file.write_text(json.dumps(test_data[1]), "utf-8")
    byte_stream = ByteStream.from_string(json.dumps(test_data[2]))

    sources = [str(first_test_file), second_test_file, byte_stream]
    meta = [
        {"creation_date": "1945-05-25T00:00:00"},
        {"creation_date": "1943-09-03T00:00:00"},
        {"creation_date": "1989-11-09T00:00:00"},
    ]
    converter = JSONConverter(jq_schema='.laureates[] | .firstname + " " + .surname + " " + .motivation')
    result = converter.run(sources=sources, meta=meta)
    assert len(result) == 1
    assert len(result["documents"]) == 4
    assert (
        result["documents"][0].content
        == "Dario Fokin who emulates the jesters of the Middle Ages in scourging authority and "
        "upholding the dignity of the downtrodden"
    )
    assert result["documents"][0].meta == {
        "file_path": os.path.basename(first_test_file),
        "creation_date": "1945-05-25T00:00:00",
    }
    assert result["documents"][1].content == "Stanley Cohen for their discoveries of growth factors"
    assert result["documents"][1].meta == {
        "file_path": os.path.basename(second_test_file),
        "creation_date": "1943-09-03T00:00:00",
    }
    assert result["documents"][2].content == "Rita Levi-Montalcini for their discoveries of growth factors"
    assert result["documents"][2].meta == {
        "file_path": os.path.basename(second_test_file),
        "creation_date": "1943-09-03T00:00:00",
    }
    assert (
        result["documents"][3].content == "Enrico Fermi for his demonstrations of the existence of new "
        "radioactive elements produced by neutron irradiation, and for his related discovery of nuclear "
        "reactions brought about by slow neutrons"
    )
    assert result["documents"][3].meta == {"creation_date": "1989-11-09T00:00:00"}


def test_run_with_meta_list_of_differing_length(tmpdir):
    sources = ["random_file.json"]

    meta = [{}, {}]
    converter = JSONConverter(jq_schema=".")
    with pytest.raises(ValueError, match="The length of the metadata list must match the number of sources."):
        converter.run(sources=sources, meta=meta)


def test_run_with_jq_schema_and_content_key(tmpdir):
    first_test_file = Path(tmpdir / "first_test_file.json")
    second_test_file = Path(tmpdir / "second_test_file.json")

    first_test_file.write_text(json.dumps(test_data[0]), "utf-8")
    second_test_file.write_text(json.dumps(test_data[1]), "utf-8")
    byte_stream = ByteStream.from_string(json.dumps(test_data[2]))

    sources = [str(first_test_file), second_test_file, byte_stream]
    converter = JSONConverter(jq_schema=".laureates[]", content_key="motivation")
    result = converter.run(sources=sources)
    assert len(result) == 1
    assert len(result["documents"]) == 4
    assert (
        result["documents"][0].content == "who emulates the jesters of the Middle Ages in scourging authority and "
        "upholding the dignity of the downtrodden"
    )
    assert result["documents"][0].meta == {"file_path": os.path.basename(first_test_file)}
    assert result["documents"][1].content == "for their discoveries of growth factors"
    assert result["documents"][1].meta == {"file_path": os.path.basename(second_test_file)}
    assert result["documents"][2].content == "for their discoveries of growth factors"
    assert result["documents"][2].meta == {"file_path": os.path.basename(second_test_file)}
    assert (
        result["documents"][3].content == "for his demonstrations of the existence of new "
        "radioactive elements produced by neutron irradiation, and for his related discovery of nuclear "
        "reactions brought about by slow neutrons"
    )
    assert result["documents"][3].meta == {}


def test_run_with_jq_schema_content_key_and_extra_meta_fields(tmpdir):
    first_test_file = Path(tmpdir / "first_test_file.json")
    second_test_file = Path(tmpdir / "second_test_file.json")

    first_test_file.write_text(json.dumps(test_data[0]), "utf-8")
    second_test_file.write_text(json.dumps(test_data[1]), "utf-8")
    byte_stream = ByteStream.from_string(json.dumps(test_data[2]))

    sources = [str(first_test_file), second_test_file, byte_stream]
    converter = JSONConverter(
        jq_schema=".laureates[]", content_key="motivation", extra_meta_fields={"firstname", "surname"}
    )
    result = converter.run(sources=sources)
    assert len(result) == 1
    assert len(result["documents"]) == 4
    assert (
        result["documents"][0].content == "who emulates the jesters of the Middle Ages in scourging authority and "
        "upholding the dignity of the downtrodden"
    )
    assert result["documents"][0].meta == {
        "file_path": os.path.basename(first_test_file),
        "firstname": "Dario",
        "surname": "Fokin",
    }
    assert result["documents"][1].content == "for their discoveries of growth factors"
    assert result["documents"][1].meta == {
        "file_path": os.path.basename(second_test_file),
        "firstname": "Stanley",
        "surname": "Cohen",
    }
    assert result["documents"][2].content == "for their discoveries of growth factors"
    assert result["documents"][2].meta == {
        "file_path": os.path.basename(second_test_file),
        "firstname": "Rita",
        "surname": "Levi-Montalcini",
    }
    assert (
        result["documents"][3].content == "for his demonstrations of the existence of new "
        "radioactive elements produced by neutron irradiation, and for his related discovery of nuclear "
        "reactions brought about by slow neutrons"
    )
    assert result["documents"][3].meta == {"firstname": "Enrico", "surname": "Fermi"}


def test_run_with_content_key(tmpdir):
    first_test_file = Path(tmpdir / "first_test_file.json")
    second_test_file = Path(tmpdir / "second_test_file.json")

    first_test_file.write_text(json.dumps(test_data[0]), "utf-8")
    second_test_file.write_text(json.dumps(test_data[1]), "utf-8")
    byte_stream = ByteStream.from_string(json.dumps(test_data[2]))

    sources = [str(first_test_file), second_test_file, byte_stream]
    converter = JSONConverter(content_key="category")
    result = converter.run(sources=sources)
    assert len(result) == 1
    assert len(result["documents"]) == 3
    assert result["documents"][0].content == "literature"
    assert result["documents"][0].meta == {"file_path": os.path.basename(first_test_file)}
    assert result["documents"][1].content == "medicine"
    assert result["documents"][1].meta == {"file_path": os.path.basename(second_test_file)}
    assert result["documents"][2].content == "physics"
    assert result["documents"][2].meta == {}


def test_run_with_content_key_and_extra_meta_fields(tmpdir):
    first_test_file = Path(tmpdir / "first_test_file.json")
    second_test_file = Path(tmpdir / "second_test_file.json")

    first_test_file.write_text(json.dumps(test_data[0]), "utf-8")
    second_test_file.write_text(json.dumps(test_data[1]), "utf-8")
    byte_stream = ByteStream.from_string(json.dumps(test_data[2]))

    sources = [str(first_test_file), second_test_file, byte_stream]
    converter = JSONConverter(content_key="category", extra_meta_fields={"year"})
    result = converter.run(sources=sources)
    assert len(result) == 1
    assert len(result["documents"]) == 3
    assert result["documents"][0].content == "literature"
    assert result["documents"][0].meta == {"file_path": os.path.basename(first_test_file), "year": "1997"}
    assert result["documents"][1].content == "medicine"
    assert result["documents"][1].meta == {"file_path": os.path.basename(second_test_file), "year": "1986"}
    assert result["documents"][2].content == "physics"
    assert result["documents"][2].meta == {"year": "1938"}


def test_run_with_jq_schema_content_key_and_extra_meta_fields_literal(tmpdir):
    first_test_file = Path(tmpdir / "first_test_file.json")
    second_test_file = Path(tmpdir / "second_test_file.json")

    first_test_file.write_text(json.dumps(test_data[0]), "utf-8")
    second_test_file.write_text(json.dumps(test_data[1]), "utf-8")
    byte_stream = ByteStream.from_string(json.dumps(test_data[2]))

    sources = [str(first_test_file), second_test_file, byte_stream]
    converter = JSONConverter(jq_schema=".laureates[]", content_key="motivation", extra_meta_fields="*")
    result = converter.run(sources=sources)
    assert len(result) == 1
    assert len(result["documents"]) == 4
    assert (
        result["documents"][0].content
        == "who emulates the jesters of the Middle Ages in scourging authority and upholding the dignity of the downtrodden"
    )
    assert result["documents"][0].meta == {
        "file_path": os.path.basename(first_test_file),
        "id": "674",
        "firstname": "Dario",
        "surname": "Fokin",
        "share": "1",
    }
    assert result["documents"][1].content == "for their discoveries of growth factors"
    assert result["documents"][1].meta == {
        "file_path": os.path.basename(second_test_file),
        "id": "434",
        "firstname": "Stanley",
        "surname": "Cohen",
        "share": "2",
    }
    assert result["documents"][2].content == "for their discoveries of growth factors"
    assert result["documents"][2].meta == {
        "file_path": os.path.basename(second_test_file),
        "id": "435",
        "firstname": "Rita",
        "surname": "Levi-Montalcini",
        "share": "2",
    }
    assert (
        result["documents"][3].content
        == "for his demonstrations of the existence of new radioactive elements produced by neutron irradiation, "
        "and for his related discovery of nuclear reactions brought about by slow neutrons"
    )
    assert result["documents"][3].meta == {"id": "46", "firstname": "Enrico", "surname": "Fermi", "share": "1"}

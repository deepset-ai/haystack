from ctypes import c_int32
from typing import List


class AI21J2CompleteInput:
    params = {
        "numResults": c_int32,
        "prompt": str,
        "numResults": c_int32,
        "maxTokens": c_int32,
        "minTokens": c_int32,
        "temperature": float,
        "topP": float,
        "stopSequences": List[str],
        "topKReturn": c_int32,
        "frequencyPenalty": {
            "scale": c_int32,
            "applyToWhitespaces": bool,
            "applyToPunctuation": bool,
            "applyToNumbers": bool,
            "applyToStopwords": bool,
            "applyToEmojis": bool,
        },
        "presencePenalty": {
            "scale": c_int32,
            "applyToWhitespaces": bool,
            "applyToPunctuation": bool,
            "applyToNumbers": bool,
            "applyToStopwords": bool,
            "applyToEmojis": bool,
        },
        "countPenalty": {
            "scale": c_int32,
            "applyToWhitespaces": bool,
            "applyToPunctuation": bool,
            "applyToNumbers": bool,
            "applyToStopwords": bool,
            "applyToEmojis": bool,
        },
    }
    default_params = {
        "prompt": None,
        "numResults": 1,
        "maxTokens": 16,
        "minTokens": 0,
        "temperature": 0.7,
        "topP": 1,
        "stopSequences": None,
        "topKReturn": 0,
        "frequencyPenalty": {
            "scale": 1,
            "applyToWhitespaces": True,
            "applyToPunctuation": True,
            "applyToNumbers": True,
            "applyToStopwords": True,
            "applyToEmojis": True,
        },
        "presencePenalty": {
            "scale": 0,
            "applyToWhitespaces": True,
            "applyToPunctuation": True,
            "applyToNumbers": True,
            "applyToStopwords": True,
            "applyToEmojis": True,
        },
        "countPenalty": {
            "scale": 0,
            "applyToWhitespaces": True,
            "applyToPunctuation": True,
            "applyToNumbers": True,
            "applyToStopwords": True,
            "applyToEmojis": True,
        },
    }

    test_payload = {"prompt": "Hello world!"}


class AI21TaskSpecificContextualAnswers(AI21J2CompleteInput):
    params = {"context": str, "question": str}

    default_params = {"context": None, "question": None}

    test_payload = {
        "context": "The tower is 330 metres (1,083 ft) tall,[6] about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest human-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure in the world to surpass both the 200-metre and 300-metre mark in height. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
        "question": "What is the height of the Eiffel tower?",
    }

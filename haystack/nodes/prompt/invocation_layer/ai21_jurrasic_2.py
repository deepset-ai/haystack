class J2CompleteInput:  
    params = [
      "num_results"
    ]

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
            "applyToEmojis": True
        },
        "presencePenalty": {
            "scale":0,
            "applyToWhitespaces": True,
            "applyToPunctuation": True,
            "applyToNumbers": True,
            "applyToStopwords": True,
            "applyToEmojis": True
        },
        "countPenalty": {
            "scale":0,
            "applyToWhitespaces": True,
            "applyToPunctuation": True,
            "applyToNumbers": True,
            "applyToStopwords": True,
            "applyToEmojis": True
        }
    }



class TaskSpecificContextualAnswers(J2CompleteInput):
    params = [
        "context",
        "question"
    ]

   
from typing import Any, Dict, List


def get_answers_from_output(outputs: List[Dict[str, Any]], runnable_type: str) -> List[str]:
    """
    Extracts the answers from the output of a pipeline or component.

    :param outputs: The outputs of the runnable.
    :return: List of answers from the runnable output.
    """
    answers = []
    if runnable_type == "pipeline":
        # Iterate over output from each Pipeline run
        for output in outputs:
            # Iterate over output of component in each Pipeline run
            for component_output in output.values():
                # Only extract answers
                for key in component_output.keys():
                    if "answers" in key:
                        for generated_answer in component_output["answers"]:
                            if generated_answer.data:
                                answers.append(generated_answer.data)
    else:
        # Iterate over output from each Component run
        for output in outputs:
            # Only extract answers
            for key in output.keys():
                if "answers" in key:
                    for generated_answer in output["answers"]:
                        if generated_answer.data:
                            answers.append(generated_answer.data)
    return answers

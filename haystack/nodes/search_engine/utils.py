from typing import List, Any, Optional


def calculate_ranking_scores(list_items: List[Any], boost_first_factor: Optional[int] = None) -> List[float]:
    """
    Assigns scores to items in a list based on their rank position and ensures that the scores add up to 1.
    :param list_items: The list of items to score.
    :param boost_first_factor: The factor to boost the score of the first item by.
    """
    n = len(list_items)
    scores = [0.0] * n

    # Compute the scores based on rank position
    for i, _ in enumerate(list_items):
        scores[i] = (n - i) / ((n * (n + 1)) / 2)

    # Apply the boost factor to the first item
    if boost_first_factor is not None and n > 0:
        scores[0] *= boost_first_factor

    # Normalize the scores so they add up to 1
    total_score = sum(scores)
    normalized_scores = [score / total_score for score in scores]

    return normalized_scores

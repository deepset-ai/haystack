from typing import Any, Dict, List


class ResultsEvaluator:
    def __init__(self, results: Dict[str, Any]):
        self.results = results

    def individual_aggregate_score_report(self):
        pass

    def comparative_aggregate_score_report(self):
        pass

    def individual_detailed_score_report(self):
        pass

    def comparative_detailed_score_report(self):
        pass

    def find_thresholds(self):
        pass

    def find_inputs_below_threshold(self):
        pass

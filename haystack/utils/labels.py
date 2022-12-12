from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from haystack.schema import Label, MultiLabel


def aggregate_labels(
    labels: List[Label],
    add_closed_domain_filter: bool = False,
    add_meta_filters: Optional[Union[str, list]] = None,
    drop_negative_labels: bool = False,
    drop_no_answers: bool = False,
) -> List[MultiLabel]:
    """
    Aggregates Labels into MultiLabel objects (e.g. for evaluation with `Pipeline.eval()`).

    Labels are always aggregated by question and filters defined in the Label objects.
    Beyond that you have options to drop certain labels or to dynamically add filters to control the aggregation process.

    Closed domain aggregation:
    If the questions are being asked only on the document defined within the Label (i.e. SQuAD style), set `add_closed_domain_filter=True` to aggregate by question, filters and document.
    Note that Labels' filters are enriched with the document_id of the Label's document.
    Note that you don't need that step
    - if your labels already contain the document_id in their filters
    - if you're using `Pipeline.eval()`'s `add_isolated_node_eval` feature

    Dynamic metadata aggregation:
    If the questions are being asked on a subslice of your document set, that is not defined with the Label's filters but with an additional meta field,
    populate `add_meta_filters` with the names of Label meta fields to aggregate by question, filters and your custom meta fields.
    Note that Labels' filters are enriched with the specified meta fields defined in the Label.
    Remarks: `add_meta_filters` is only intended for dynamic metadata aggregation (e.g. separate evaluations per document type).
    For standard questions use-cases, where a question is always asked on multiple files individually, consider setting the Label's filters instead.
    For example, if you want to ask a couple of standard questions for each of your products, set filters for "product_id" to your Labels.
    Thus you specify that each Label is always only valid for documents with the respective product_id.

    :param labels: List of Labels to aggregate.
    :param add_closed_domain_filter: When True, adds a filter for the document ID specified in the label.
                        Thus, labels are aggregated in a closed domain fashion based on the question text, filters,
                        and also the id of the document that the label is tied to. See "closed domain aggregation" section for more details.
    :param add_meta_filters: The names of the Label meta fields by which to aggregate in addition to question and filters. For example: ["product_id"].
                        Note that Labels' filters are enriched with the specified meta fields defined in the Label.
    :param drop_negative_labels: When True, labels with incorrect answers and documents are dropped.
    :param drop_no_answers: When True, labels with no answers are dropped.
    :return: A list of MultiLabel objects.
    """
    if add_meta_filters:
        if type(add_meta_filters) == str:
            add_meta_filters = [add_meta_filters]
    else:
        add_meta_filters = []

    # drop no_answers in order to not create empty MultiLabels
    if drop_no_answers:
        labels = [label for label in labels if label.no_answer == False]

    # add filters for closed domain and dynamic metadata aggregation
    for l in labels:
        label_filters_to_add = {}
        if add_closed_domain_filter:
            label_filters_to_add["_id"] = l.document.id

        for meta_key in add_meta_filters:
            meta = l.meta or {}
            curr_meta = meta.get(meta_key, None)
            if curr_meta:
                curr_meta = curr_meta if isinstance(curr_meta, list) else [curr_meta]
                label_filters_to_add[meta_key] = curr_meta

        if label_filters_to_add:
            if l.filters is None:
                l.filters = label_filters_to_add
            else:
                l.filters.update(label_filters_to_add)

    # Filters define the scope a label is valid for the query, so we group the labels by query and filters.
    grouped_labels: Dict[Tuple, List[Label]] = defaultdict(list)
    for l in labels:
        label_filter_keys = [f"{k}={v}" for k, v in l.filters.items()] if l.filters else []
        group_keys: list = [l.query] + label_filter_keys
        group_key = tuple(group_keys)
        grouped_labels[group_key].append(l)

    aggregated_labels = [
        MultiLabel(labels=ls, drop_negative_labels=drop_negative_labels, drop_no_answers=drop_no_answers)
        for ls in grouped_labels.values()
    ]

    return aggregated_labels

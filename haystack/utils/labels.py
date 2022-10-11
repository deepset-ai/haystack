from typing import List, Optional, Union

from haystack.schema import Label, MultiLabel


def aggregate_labels(
    labels: List[Label],
    open_domain: bool = True,
    drop_negative_labels: bool = False,
    drop_no_answers: bool = False,
    aggregate_by_meta: Optional[Union[str, list]] = None,
):
    """
    Aggregates Labels into MultiLabel objects (e.g. for evaluation with `Pipeline.eval()`).

    Labels are always aggregated by question and filters defined in the Label objects.
    Beyond that you have options to dynamically aggregate by other meta fields or to drop certain labels.

    Closed domain aggregation:
    If the questions are being asked only on the document defined within the Label (i.e. SQuAD style), set `open_domain=False` to aggregate by question, filters and document.
    Note that Labels' filters are enriched with the document_id of the Label's document.
    Note that you don't need that step
    - if your labels already contain the document_id in their filters
    - if you're using `Pipeline.eval()`'s `add_isolated_node_eval` feature

    Dynamic metadata aggregation:
    If the questions are being asked on a subslice of your document set, that is not defined with the Label's filters but with an additional meta field,
    populate `aggregate_by_meta` with the names of Label meta fields to aggregate by question, filters and your custom meta fields.
    Note that Labels' filters are enriched with the specified meta fields defined in the Label.
    Remarks: `aggregate_by_meta` is only intended for dynamic metadata aggregation (e.g. separate evaluations per document type).
    For standard questions use-cases, where a question is always asked on multiple files individually, consider setting the Label's filters instead.
    For example, if you want to ask a couple of standard questions for each of your products, set filters for "product_id" to your Labels.
    Thus you specify that each Label is always only valid for documents with the respective product_id.

    :param labels: List of Labels to aggregate.
    :param open_domain: When True, labels are aggregated based on the question and filters alone.
                        Use this if the questions are being asked to your full collection of documents or for standard questions use-cases.
                        When False, labels are aggregated in a closed domain fashion based on the question text, filters,
                        and also the id of the document that the label is tied to. See "closed domain aggregation" section for more details.
    :param aggregate_by_meta: The names of the Label meta fields by which to aggregate in addition to question and filters. For example: ["product_id"]
    :param drop_negative_labels: When True, labels with incorrect answers and documents are dropped.
    :param drop_no_answers: When True, labels with no answers are dropped.
    :return: A list of MultiLabel objects.
    """
    if aggregate_by_meta:
        if type(aggregate_by_meta) == str:
            aggregate_by_meta = [aggregate_by_meta]
    else:
        aggregate_by_meta = []

    # drop no_answers in order to not create empty MultiLabels
    if drop_no_answers:
        labels = [label for label in labels if label.no_answer == False]

    grouped_labels: dict = {}
    for l in labels:
        # This group_keys determines the key by which we aggregate labels. Its contents depend on
        # whether we are in an open / closed domain setting, on filters that are specified for labels,
        # or if there are fields in the meta data that we should group by dynamically (set using group_by_meta).
        label_filter_keys = [f"{k}={''.join(v)}" for k, v in l.filters.items()] if l.filters else []
        group_keys: list = [l.query] + label_filter_keys
        # Filters indicate the scope within which a label is valid.
        # Depending on the aggregation we need to add filters dynamically.
        label_filters_to_add: dict = {}

        if not open_domain:
            group_keys.append(f"_id={l.document.id}")
            label_filters_to_add["_id"] = l.document.id

        for meta_key in aggregate_by_meta:
            meta = l.meta or {}
            curr_meta = meta.get(meta_key, None)
            if curr_meta:
                curr_meta = curr_meta if isinstance(curr_meta, list) else [curr_meta]
                meta_str = f"{meta_key}={''.join(curr_meta)}"
                group_keys.append(meta_str)
                label_filters_to_add[meta_key] = curr_meta

        if label_filters_to_add:
            if l.filters is None:
                l.filters = label_filters_to_add
            else:
                l.filters.update(label_filters_to_add)

        group_key = tuple(group_keys)
        if group_key in grouped_labels:
            grouped_labels[group_key].append(l)
        else:
            grouped_labels[group_key] = [l]

    # Package labels that we grouped together in a MultiLabel object that allows simpler access to some
    # aggregated attributes like `no_answer`
    aggregated_labels = [
        MultiLabel(labels=ls, drop_negative_labels=drop_negative_labels, drop_no_answers=drop_no_answers)
        for ls in grouped_labels.values()
    ]

    return aggregated_labels

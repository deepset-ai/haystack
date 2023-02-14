from typing import Optional, List, Union

import logging

from haystack.schema import FilterType


logger = logging.getLogger(__name__)


# Mock Pinecone instance
CONFIG: dict = {"api_key": None, "environment": None, "indexes": {}}


# Mock Pinecone Index instance
class IndexObject:
    def __init__(
        self,
        index: str,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        dimension: Optional[int] = None,
        metric: Optional[str] = None,
        replicas: Optional[int] = None,
        shards: Optional[int] = None,
        metadata_config: Optional[dict] = None,
    ):
        self.index = index
        self.api_key = api_key
        self.environment = environment
        self.dimension = dimension
        self.metric = metric
        self.replicas = replicas
        self.shards = shards
        self.metadata_config = metadata_config
        self.namespaces: dict = {}


# Mock the Pinecone Index class
class Index:
    def __init__(self, index: str):
        self.index = index
        self.index_config = CONFIG["indexes"][index]

    def upsert(self, vectors: List[tuple], namespace: str = ""):
        if namespace not in self.index_config.namespaces:
            self.index_config.namespaces[namespace] = {}
        upsert_count = 0
        for record in vectors:
            # Extract info from tuple
            _id = record[0]
            vector = record[1]
            metadata = record[2]
            # Checks
            assert type(_id) is str
            assert type(vector) is list
            assert len(vector) == self.index_config.dimension
            assert type(metadata) is dict
            # Create record (eg document)
            new_record: dict = {"id": _id, "values": vector, "metadata": metadata}
            self.index_config.namespaces[namespace][_id] = new_record
            upsert_count += 1
        return {"upserted_count": upsert_count}

    def update(self, namespace: str, id: str, set_metadata: dict):
        # Get existing item metadata
        meta = self.index_config.namespaces[namespace][id]["metadata"]
        # Add new metadata to existing item metadata
        self.index_config.namespaces[namespace][id]["metadata"] = {**meta, **set_metadata}

    def describe_index_stats(self, filter=None):
        namespaces = {}
        for namespace in self.index_config.namespaces.items():
            records = self.index_config.namespaces[namespace[0]]
            if filter:
                filtered_records = []
                for record in records.values():
                    if self._filter(metadata=record["metadata"], filters=filter, top_level=True):
                        filtered_records.append(record)
                records = filtered_records
            namespaces[namespace[0]] = {"vector_count": len(records)}
        return {"dimension": self.index_config.dimension, "index_fullness": 0.0, "namespaces": namespaces}

    def query(
        self,
        vector: List[float],
        top_k: int,
        namespace: str = "",
        include_values: bool = False,
        include_metadata: bool = False,
        filter: Optional[dict] = None,
    ):
        return self.query_filter(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            include_values=include_values,
            include_metadata=include_metadata,
            filter=filter,
        )

    def query_filter(
        self,
        vector: List[float],
        top_k: int,
        namespace: str = "",
        include_values: bool = False,
        include_metadata: bool = False,
        filter: Optional[dict] = None,
    ):
        assert len(vector) == self.index_config.dimension
        response: dict = {"matches": []}
        if namespace not in self.index_config.namespaces:
            return response
        else:
            records = self.index_config.namespaces[namespace]
            namespace_ids = list(records.keys())[:top_k]

            for _id in namespace_ids:
                match = {"id": _id}
                if include_values:
                    match["values"] = records[_id]["values"].copy()
                if include_metadata:
                    match["metadata"] = records[_id]["metadata"].copy()
                match["score"] = 0.0

                if filter is None or (
                    filter is not None and self._filter(records[_id]["metadata"], filter, top_level=True)
                ):
                    # filter if needed
                    response["matches"].append(match)
            return response

    def fetch(self, ids: List[str], namespace: str = ""):
        response: dict = {"namespace": namespace, "vectors": {}}
        if namespace not in self.index_config.namespaces:
            # If we query an empty/non-existent namespace, Pinecone will just return an empty response
            logger.warning("No namespace called '%s'", namespace)
            return response
        records = self.index_config.namespaces[namespace]
        namespace_ids = records.keys()
        for _id in namespace_ids:
            if _id in ids.copy():
                response["vectors"][_id] = {
                    "id": _id,
                    "metadata": records[_id]["metadata"].copy(),
                    "values": records[_id]["values"].copy(),
                }
        return response

    def _filter(
        self,
        metadata: dict,
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]],
        mode: Optional[str] = "$and",
        top_level=False,
    ) -> dict:
        """
        Mock filtering function
        """
        bools = []
        if type(filters) is list:
            list_bools = []
            for _filter in filters:
                res = self._filter(metadata, _filter, mode=mode)
                for key, value in res.items():
                    if key == "$and":
                        list_bools.append(all(value))
                    else:
                        list_bools.append(any(value))
            if mode == "$and":
                bools.append(all(list_bools))
            elif mode == "$or":
                bools.append(any(list_bools))
        else:
            for field, potential_value in filters.items():
                if field in ["$and", "$or"]:
                    bools.append(self._filter(metadata, potential_value, mode=field))
                    mode = field
                    cond = field
                else:
                    if type(potential_value) is dict:
                        sub_bool = []
                        for cond, value in potential_value.items():
                            if len(potential_value.keys()) > 1:
                                sub_filter = {field: {cond: value}}
                                bools.append(self._filter(metadata, sub_filter))
                        if len(sub_bool) > 1:
                            if field == "$or":
                                bools.append(any(sub_bool))
                            else:
                                bools.append(all(sub_bool))
                    elif type(potential_value) is list:
                        cond = "$in"
                        value = potential_value
                    else:
                        cond = "$eq"
                        value = potential_value
                    # main chunk of condition checks
                    if cond == "$eq":
                        if field in metadata and metadata[field] == value:
                            bools.append(True)
                        else:
                            bools.append(False)
                    elif cond == "$ne":
                        if field in metadata and metadata[field] != value:
                            bools.append(True)
                        else:
                            bools.append(False)
                    elif cond == "$in":
                        if field in metadata and metadata[field] in value:
                            bools.append(True)
                        else:
                            bools.append(False)
                    elif cond == "$nin":
                        if field in metadata and metadata[field] not in value:
                            bools.append(True)
                        else:
                            bools.append(False)
                    elif cond == "$gt":
                        if field in metadata and metadata[field] > value:
                            bools.append(True)
                        else:
                            bools.append(False)
                    elif cond == "$lt":
                        if field in metadata and metadata[field] < value:
                            bools.append(True)
                        else:
                            bools.append(False)
                    elif cond == "$gte":
                        if field in metadata and metadata[field] >= value:
                            bools.append(True)
                        else:
                            bools.append(False)
                    elif cond == "$lte":
                        if field in metadata and metadata[field] <= value:
                            bools.append(True)
                        else:
                            bools.append(False)
        if top_level:
            final = []
            for item in bools:
                if type(item) is dict:
                    for key, value in item.items():
                        if key == "$and":
                            final.append(all(value))
                        else:
                            final.append(any(value))
                else:
                    final.append(item)
            if mode == "$and":
                bools = all(final)
            else:
                bools = any(final)
        else:
            if mode == "$and":
                return {"$and": bools}
            else:
                return {"$or": bools}
        return bools

    def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: str = "",
        filters: Optional[FilterType] = None,
        delete_all: bool = False,
    ):
        if filters:
            # Get a filtered list of IDs
            matches = self.query(filters=filters, namespace=namespace, include_values=False, include_metadata=False)[
                "vectors"
            ]
            filter_ids: List[str] = matches.keys()  # .keys() returns an object that supports set operators already
        elif delete_all:
            self.index_config.namespaces[namespace] = {}

        if namespace not in self.index_config.namespaces:
            pass
        elif ids is not None:
            id_list: List[str] = ids
            if filters:
                # We find the intersect between the IDs and filtered IDs
                id_list = set(id_list).intersection(filter_ids)
            records = self.index_config.namespaces[namespace]
            for _id in list(records.keys()):  # list() is needed to be able to del below
                if _id in id_list:
                    del records[_id]
        else:
            # Delete all
            self.index_config.namespaces[namespace] = {}
        return {}

    def _get_config(self):
        return self.index_config


# Mock core Pinecone client functions
def init(api_key: Optional[str] = None, environment: Optional[str] = None):
    CONFIG["api_key"] = api_key
    CONFIG["environment"] = environment
    CONFIG["indexes"] = {}


def list_indexes():
    return list(CONFIG["indexes"].keys())


def create_index(
    name: str,
    dimension: int,
    metric: str = "cosine",
    replicas: int = 1,
    shards: int = 1,
    metadata_config: Optional[dict] = None,
):
    index_object = IndexObject(
        api_key=CONFIG["api_key"],
        environment=CONFIG["environment"],
        index=name,
        dimension=dimension,
        metric=metric,
        replicas=replicas,
        shards=shards,
        metadata_config=metadata_config,
    )
    CONFIG["indexes"][name] = index_object


def delete_index(index: str):
    del CONFIG["indexes"][index]

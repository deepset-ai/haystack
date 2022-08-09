from typing import Optional, List, Dict, Union

import logging

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

    def _filter(self, records: list, filter: Optional[dict]):
        namespace_records = []
        for record in records.values():
            if all(record["metadata"].get(key) in values for key, values in filter.items()):
                namespace_records.append(record)
        return namespace_records

    def describe_index_stats(self, filter=None):
        namespaces = {}
        for namespace in self.index_config.namespaces.items():
            records = self.index_config.namespaces[namespace[0]]
            if filter:
                records = self._filter(records, filter)
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
        assert len(vector) == self.index_config.dimension
        response: dict = {"matches": []}
        if namespace not in self.index_config.namespaces:
            return response
        else:
            records = self.index_config.namespaces[namespace]
            raw_namespace_ids = list(records.keys())[:top_k]
            if filter:
                namespace_ids = []
                for _id in raw_namespace_ids:
                    if all(records[_id]["metadata"].get(key) in values for key, values in filter.items()):
                        namespace_ids.append(_id)
            else:
                namespace_ids = raw_namespace_ids
            for _id in namespace_ids:
                match = {"id": _id}
                if include_values:
                    match["values"] = records[_id]["values"].copy()
                if include_metadata:
                    match["metadata"] = records[_id]["metadata"].copy()
                match["score"] = 0.0
                response["matches"].append(match)
            return response

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
                    filter is not None and self._filter(records[_id]["metadata"], filter, show=False, top_level=True)
                ):
                    # filter if needed
                    response["matches"].append(match)
            return response

    def fetch(self, ids: List[str], namespace: str = ""):
        response: dict = {"namespace": namespace, "vectors": {}}
        if namespace not in self.index_config.namespaces:
            # If we query an empty/non-existent namespace, Pinecone will just return an empty response
            logger.warning(f"No namespace called '{namespace}'")
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
        filters: Dict[str, Union[str, int, float, bool, list]],
        mode: Optional[str] = "$and",
        show=False,
        top_level=False,
    ) -> dict:
        """
        Mock filtering function
        """
        bools = []
        for field, potential_value in filters.items():
            if field in ["$and", "$or"]:
                print("if field in [and, or]")
                bools.append(self._filter(metadata, potential_value, mode=field, show=show))
                mode = field
                cond = field
            else:
                if type(potential_value) is dict:
                    sub_bool = []
                    for cond, value in potential_value.items():
                        if len(potential_value.keys()) > 1:
                            if show:
                                print(f"{potential_value=}")
                            sub_filter = {field: {cond: value}}
                            if show:
                                print(f"{sub_filter=}")
                            bools.append(self._filter(metadata, sub_filter))
                    if len(sub_bool) > 1:
                        if field == "$or":
                            bools.append(any(sub_bool))
                        else:
                            bools.append(all(sub_bool))
                elif type(potential_value) is list:
                    cond = "$in"
                    value = potential_value
                    if show:
                        print(f"$in")
                else:
                    cond = "$eq"
                    value = potential_value
                    if show:
                        print("$eq")
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
        if show:
            print(cond)
        if top_level:
            if mode == "$and":
                if show:
                    print(f"\n{metadata}\n{mode}:{all(bools)} | {filters}\n{bools}\n")
                bools = all(bools)
            else:
                if show:
                    print(f"\n{metadata}\n{mode}:{any(bools)} | {filters}\n{bools}\n")
                bools = any(bools)
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
        filters: Optional[dict] = None,
        delete_all: bool = False,
    ):
        if delete_all:
            self.index_config.namespaces[namespace] = {}

        if namespace not in self.index_config.namespaces:
            pass
        elif ids is not None:
            id_list: List[str] = ids
            records = self.index_config.namespaces[namespace]
            for _id in list(records.keys()):
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

from typing import Optional, List

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
            self.index_config.namespaces[namespace] = []
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
            self.index_config.namespaces[namespace].append(new_record)
            upsert_count += 1
        return {"upserted_count": upsert_count}

    def describe_index_stats(self):
        namespaces = {}
        for namespace in self.index_config.namespaces.items():
            namespaces[namespace[0]] = {"vector_count": len(namespace[1])}
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
            records = self.index_config.namespaces[namespace][:top_k]
            for record in records:
                match = {"id": record["id"]}
                if include_values:
                    match["values"] = record["values"].copy()
                if include_metadata:
                    match["metadata"] = record["metadata"].copy()
                match["score"] = 0.0
                response["matches"].append(match)
            return response

    def fetch(self, ids: List[str], namespace: str = ""):
        response: dict = {"namespace": namespace, "vectors": {}}
        if namespace not in self.index_config.namespaces:
            raise ValueError("Namespace not found")
        records = self.index_config.namespaces[namespace]
        for record in records:
            if record["id"] in ids.copy():
                response["vectors"][record["id"]] = {
                    "id": record["id"],
                    "metadata": record["metadata"].copy(),
                    "values": record["values"].copy(),
                }
        return response

    def delete(self, ids: Optional[List[str]] = None, namespace: str = "", filters: Optional[dict] = None):
        if namespace not in self.index_config.namespaces:
            pass
        elif ids is not None:
            id_list: List[str] = ids
            records = self.index_config.namespaces[namespace]
            for record in records:
                if record["id"] in id_list:
                    records.remove(record)
        else:
            # Delete all
            self.index_config.namespaces[namespace] = []
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

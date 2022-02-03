from typing import Union, List, Optional
from abc import ABC, abstractmethod


class LogOp(ABC):
    def __init__(self, conditions):
        self.conditions = conditions
    @abstractmethod
    def evaluate(self, fields):
        pass
    @classmethod
    def deserialize(cls, items: Union[dict, List[dict]]):
        conditions = []
        if isinstance(items, dict):
            items = [items]
        for item in items:
            for key in item:
                if key == "$not":
                    conditions.append(NotOp.deserialize(item[key]))
                elif key == "$and":
                    conditions.append(AndOp.deserialize(item[key]))
                elif key == "$or":
                    conditions.append(OrOp.deserialize(item[key]))
                else:
                    conditions.append(CompList.deserialize(key, item[key]))
        if cls == LogOp:
            return AndOp(conditions)
        return cls(conditions)

class NotOp:
    def __init__(self, condition):
        self.condition = condition
    def evaluate(self, fields):
        return not self.condition.evaluate(fields)
    @classmethod
    def deserialize(cls, item: dict):
        return cls(LogOp.deserialize(item))
    def serialize_milvusv2(self):
        return "not " + self.condition.serialize_milvusv2()

class AndOp(LogOp):
    def evaluate(self, fields):
        for condition in self.conditions:
            if not condition.evaluate(fields):
                return False
        return True
    def serialize_milvusv2(self):
        if len(self.conditions) == 0:
            return ""
        expression = self.conditions[0].serialize_milvusv2()
        for condition in self.conditions[1:]:
            expression += " and " + condition.serialize_milvusv2()
        return expression

class OrOp(LogOp):
    def evaluate(self, fields):
        for condition in self.conditions:
            if condition.evaluate(fields):
                return True
        return False
    def serialize_milvusv2(self):
        if len(self.conditions) == 0:
            return ""
        expression = self.conditions[0].serialize_milvusv2()
        for condition in self.conditions[1:]:
            expression += " or " + condition.serialize_milvusv2()
        return expression

class CompList:
    def __init__(self, field, conditions):
        self.field = field
        self.conditions = conditions
    def evaluate(self, fields):
        for condition in self.conditions:
            if not condition.evaluate(fields[self.field]):
                return False
        return True
    @classmethod
    def deserialize(cls, field: str, items: Union[dict, List[dict]]):
        if isinstance(items, dict):
            items = [items]
        conditions = []
        if isinstance(items, list) and isinstance(items[0], dict):
            for item in items:
                for key in item:
                    if key.startswith("$"):
                        conditions.append(Comp.deserialize(item[key], key[1:]))
                    else:
                        conditions.append(EqComp.deserialize(item[key]))
        elif isinstance(items, list):
            conditions.append(InComp.deserialize(items))
        else:
            conditions.append(EqComp.deserialize(items))
        return cls(field, conditions)
    def serialize_milvusv2(self):
        if len(self.conditions) == 0:
            return ""
        start = self.conditions[0].serialize_milvusv2(self.field)
        for condition in self.conditions[1:]:
            start += " and " + condition.serialize_milvusv2(self.field)
        return start

class Comp(ABC):
    def __init__(self, to_compare_to) -> None:
        self.to_compare_to = to_compare_to
    @abstractmethod
    def evaluate(self, field):
        pass
    @classmethod
    def deserialize(cls, item: dict, type: Optional[str] = None):
        if type is None:
            return cls(item)
        if type == "eq":
            return EqComp.deserialize(item)
        if type == "ne":
            return NeComp.deserialize(item)
        if type == "gt":
            return GtComp.deserialize(item)
        if type == "gte":
            return GteComp.deserialize(item)
        if type == "lt":
            return LtComp.deserialize(item)
        if type == "lte":
            return LteComp.deserialize(item)
        if type == "in":
            return InComp.deserialize(item)
        if type == "nin":
            return NinComp.deserialize(item)
        raise Exception(f"{type} is not a valid comparison operator")
    @abstractmethod
    def serialize_milvusv2(self, field: str):
        pass

class EqComp(Comp):
    def evaluate(self, field):
        return field == self.to_compare_to
    def serialize_milvusv2(self, field: str):
        if isinstance(self.to_compare_to, str):
            raise ValueError("Milvus does not yet support to filter on string fields")
        return f"{field} == {self.to_compare_to}"
class NeComp(Comp):
    def evaluate(self, field):
        return field != self.to_compare_to
    def serialize_milvusv2(self, field: str):
        if isinstance(self.to_compare_to, str):
            raise ValueError("Milvus does not yet support to filter on string fields")
        return f"""{field} != {self.to_compare_to}"""
class GtComp(Comp):
    def evaluate(self, field):
        return field > self.to_compare_to
    def serialize_milvusv2(self, field: str):
        if isinstance(self.to_compare_to, str):
            raise ValueError("Milvus does not yet support to filter on string fields")
        return f"""{field} > {self.to_compare_to}"""
class GteComp(Comp):
    def evaluate(self, field):
        return field >= self.to_compare_to
    def serialize_milvusv2(self, field: str):
        if isinstance(self.to_compare_to, str):
            raise ValueError("Milvus does not yet support to filter on string fields")
        return f"""{field} >= {self.to_compare_to}"""
class LtComp(Comp):
    def evaluate(self, field):
        return field < self.to_compare_to
    def serialize_milvusv2(self, field: str):
        if isinstance(self.to_compare_to, str):
            raise ValueError("Milvus does not yet support to filter on string fields")
        return f"""{field} < {self.to_compare_to}"""
class LteComp(Comp):
    def evaluate(self, field):
        return field <= self.to_compare_to
    def serialize_milvusv2(self, field: str):
        if isinstance(self.to_compare_to, str):
            raise ValueError("Milvus does not yet support to filter on string fields")
        return f"""{field} <= {self.to_compare_to}"""
class InComp(Comp):
    def evaluate(self, field):
        return field in self.to_compare_to
    def serialize_milvusv2(self, field: str):
        if isinstance(self.to_compare_to, str):
            raise ValueError("Milvus does not yet support to filter on string fields")
        return f"""{field} in {self.to_compare_to}"""
class NinComp(Comp):
    def evaluate(self, field):
        return field not in self.to_compare_to
    def serialize_milvusv2(self, field: str):
        if isinstance(self.to_compare_to, str):
            raise ValueError("Milvus does not yet support to filter on string fields")
        return f"""{field} not in {self.to_compare_to}"""
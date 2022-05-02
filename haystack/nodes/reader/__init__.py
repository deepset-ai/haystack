from haystack.nodes.reader.base import BaseReader
from haystack.nodes.reader.farm import FARMReader
from haystack.nodes.reader.transformers import TransformersReader

from haystack.utils.import_utils import safe_import

TableReader = safe_import("haystack.nodes.reader.table", "TableReader", "table-reader")
RCIReader = safe_import("haystack.nodes.reader.table", "RCIReader", "table-reader")
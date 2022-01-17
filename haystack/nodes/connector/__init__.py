from haystack.utils.import_utils import safe_import

Crawler = safe_import("haystack.nodes.connector", "Crawler", "crawler")  # Has optional dependenciesP
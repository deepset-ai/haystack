from haystack.utils.import_utils import safe_import

FileTypeClassifier = safe_import(
    "haystack.nodes.file_classifier.file_type", "FileTypeClassifier", "preprocessing"
)  # Has optional dependencies

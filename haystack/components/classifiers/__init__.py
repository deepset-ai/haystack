# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.components.classifiers.document_language_classifier import DocumentLanguageClassifier
from haystack.components.classifiers.zero_shot_document_classifier import TransformersZeroShotDocumentClassifier

__all__ = ["DocumentLanguageClassifier", "TransformersZeroShotDocumentClassifier"]

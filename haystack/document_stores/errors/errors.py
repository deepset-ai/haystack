# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


class DocumentStoreError(Exception):
    pass


class DuplicateDocumentError(DocumentStoreError):
    pass


class MissingDocumentError(DocumentStoreError):
    pass

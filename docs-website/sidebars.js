// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

export default {
  docs: [
    {
      type: 'doc',
      id: 'overview/intro',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Overview',
      items: [
        'overview/installation',
        'overview/get-started',
      ],
    },
    {
      type: 'category',
      label: 'Haystack Concepts',
      link: {
        type: 'doc',
        id: 'concepts/concepts-overview'
      },
      items: [
        'concepts/components',
        'concepts/custom-components',
        'concepts/data-classes',
        'concepts/document-store',
        'concepts/pipelines',
        'concepts/agents',
      ],
    },
  ],
};

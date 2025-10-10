// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import OriginalCodeBlock from '@theme-original/CodeBlock';

export default function CodeBlock(props) {
  // props.className will be like "language-js"
  const language = props.className?.replace('language-', '') || '';
  return (
    <div className="theme-code-block-with-lang">
      {language && <span className="code-block-language-badge">{language}</span>}
      <OriginalCodeBlock {...props} />
    </div>
  );
}

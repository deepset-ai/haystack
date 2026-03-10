// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import OriginalCodeBlock from '@theme-original/CodeBlock';

export default function CodeBlock(props) {
  // props.className will be like "language-js"
  const language = props.className?.replace('language-', '') || '';

  // Count the number of lines in the code
  const codeContent = props.children || '';
  const lineCount = typeof codeContent === 'string'
    ? codeContent.trim().split('\n').length
    : 1;

  // Determine which classes to apply based on line count
  const wrapperClasses = ['theme-code-block-with-lang'];
  if (lineCount === 1) {
    wrapperClasses.push('hide-language-badge');
  } else if (lineCount === 2) {
    wrapperClasses.push('hide-action-buttons');
  }

  return (
    <div className={wrapperClasses.join(' ')}>
      {language && <span className="code-block-language-badge">{language}</span>}
      <OriginalCodeBlock {...props} />
    </div>
  );
}

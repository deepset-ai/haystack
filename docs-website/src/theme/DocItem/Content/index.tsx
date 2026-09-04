// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import OriginalContent from '@theme-original/DocItem/Content';
import CopyDropdown from '@site/src/components/CopyDropdown';

export default function ContentWrapper(props) {
  return (
    <>
      <div className="copy-dropdown-sticky">
        <CopyDropdown />
      </div>
      <OriginalContent {...props} />
    </>
  );
}

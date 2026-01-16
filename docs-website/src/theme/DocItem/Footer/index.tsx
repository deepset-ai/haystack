// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import OriginalFooter from '@theme-original/DocItem/Footer';
import CopyDropdown from '@site/src/components/CopyDropdown';

export default function FooterWrapper(props) {
  return (
    <>
      <div className="copy-dropdown-floating">
        <CopyDropdown />
      </div>
      <OriginalFooter {...props} />
    </>
  );
}

// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import OriginalContent from '@theme-original/DocItem/Content';
import { useDoc } from '@docusaurus/plugin-content-docs/client';
import CopyDropdown from '@site/src/components/CopyDropdown';
import PlatformBadge from '@site/src/components/PlatformBadge';

export default function ContentWrapper(props) {
  const { frontMatter } = useDoc();

  return (
    <>
      <div className="copy-dropdown-sticky">
        <CopyDropdown />
      </div>
      <PlatformBadge availability={frontMatter.platform_availability} />
      <OriginalContent {...props} />
    </>
  );
}

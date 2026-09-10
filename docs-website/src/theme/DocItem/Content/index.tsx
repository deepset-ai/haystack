// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import React, { useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import OriginalContent from '@theme-original/DocItem/Content';
import { useDoc } from '@docusaurus/plugin-content-docs/client';
import CopyDropdown from '@site/src/components/CopyDropdown';
import PlatformBadge from '@site/src/components/PlatformBadge';
import styles from '@site/src/components/PlatformBadge/styles.module.css';

export default function ContentWrapper(props) {
  const { frontMatter, metadata } = useDoc();
  const containerRef = useRef(null);
  const [badgeHost, setBadgeHost] = useState(null);

  // The component title is literal markdown (`# ComponentName`) inside the MDX body,
  // not a prop we can wrap — so the badge is portaled next to the rendered <h1> itself.
  useLayoutEffect(() => {
    if (!frontMatter.platform_availability || !containerRef.current) {
      setBadgeHost(null);
      return undefined;
    }
    const heading = containerRef.current.querySelector('h1');
    if (!heading) {
      setBadgeHost(null);
      return undefined;
    }
    const host = document.createElement('span');
    heading.classList.add(styles.headingWithBadge);
    heading.appendChild(host);
    setBadgeHost(host);
    return () => {
      heading.classList.remove(styles.headingWithBadge);
      host.remove();
    };
  }, [frontMatter.platform_availability, metadata.permalink]);

  return (
    <>
      <div className="copy-dropdown-sticky">
        <CopyDropdown />
      </div>
      <div ref={containerRef}>
        <OriginalContent {...props} />
      </div>
      {badgeHost && createPortal(<PlatformBadge availability={frontMatter.platform_availability} />, badgeHost)}
    </>
  );
}

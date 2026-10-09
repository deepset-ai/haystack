// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

type PlatformBadgeProps = {
  available?: boolean;
};

const CheckIcon = () => (
  <svg viewBox="0 0 16 16" fill="none" width="12" height="12" aria-hidden="true">
    <path d="M13.5 4.5L6.5 11.5L2.5 7.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const ChevronIcon = () => (
  <svg viewBox="0 0 16 16" fill="none" width="12" height="12" aria-hidden="true">
    <path d="M6 3.5L11 8L6 12.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

/**
 * Renders directly under the component's <h1>, on doc pages whose frontmatter carries
 * `hep_available` (written automatically by scripts/generate_platform_components_table.py).
 * Absent on any page without that field, so non-component pages render nothing.
 */
const PlatformBadge: React.FC<PlatformBadgeProps> = ({ available }) => {
  if (available) {
    return (
      <Link className={`${styles.badge} ${styles.available}`} to="/docs/platform-components">
        <CheckIcon />
        Available on Haystack Enterprise Platform
        <span className={styles.chevron}>
          <ChevronIcon />
        </span>
      </Link>
    );
  }

  return null;
};

export default PlatformBadge;

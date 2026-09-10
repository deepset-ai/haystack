// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

type Availability = 'available' | 'opensource';

type PlatformBadgeProps = {
  availability?: Availability;
};

const CheckIcon = () => (
  <svg viewBox="0 0 16 16" fill="none" width="11" height="11" aria-hidden="true">
    <path d="M13.5 4.5L6.5 11.5L2.5 7.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const DotIcon = () => (
  <svg viewBox="0 0 16 16" fill="none" width="11" height="11" aria-hidden="true">
    <circle cx="8" cy="8" r="5.2" stroke="currentColor" strokeWidth="1.6" />
  </svg>
);

/**
 * Renders inline next to the component's <h1>, on doc pages whose frontmatter carries
 * `platform_availability` (written automatically by scripts/generate_platform_components_table.py).
 * Absent on any page without that field, so non-component pages render nothing.
 */
const PlatformBadge: React.FC<PlatformBadgeProps> = ({ availability }) => {
  if (availability === 'available') {
    return (
      <Link
        className={`${styles.badge} ${styles.available}`}
        to="/docs/platform-components"
        aria-label="Available in the Haystack Enterprise Platform"
      >
        <CheckIcon />
        Platform
      </Link>
    );
  }

  if (availability === 'opensource') {
    return (
      <span
        className={`${styles.badge} ${styles.opensource}`}
        aria-label="Open source only — not available in the Haystack Enterprise Platform"
      >
        <DotIcon />
        Open Source
      </span>
    );
  }

  return null;
};

export default PlatformBadge;

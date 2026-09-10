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
  <svg viewBox="0 0 16 16" fill="none" width="13" height="13" aria-hidden="true">
    <path d="M13.5 4.5L6.5 11.5L2.5 7.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const ChevronIcon = () => (
  <svg viewBox="0 0 16 16" fill="none" width="13" height="13" aria-hidden="true">
    <path d="M6 3.5L11 8L6 12.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const DotIcon = () => (
  <svg viewBox="0 0 16 16" fill="none" width="13" height="13" aria-hidden="true">
    <circle cx="8" cy="8" r="5.2" stroke="currentColor" strokeWidth="1.6" />
  </svg>
);

/**
 * Renders on component doc pages whose frontmatter carries `platform_availability`
 * (written automatically by scripts/generate_platform_components_table.py). Absent
 * on any page without that field, so non-component pages render nothing.
 */
const PlatformBadge: React.FC<PlatformBadgeProps> = ({ availability }) => {
  if (availability === 'available') {
    return (
      <Link className={`${styles.badge} ${styles.available}`} to="/platform-components">
        <CheckIcon />
        Available in the Platform
        <span className={styles.chevron}>
          <ChevronIcon />
        </span>
      </Link>
    );
  }

  if (availability === 'opensource') {
    return (
      <span className={`${styles.badge} ${styles.opensource}`}>
        <DotIcon />
        Open Source Only
      </span>
    );
  }

  return null;
};

export default PlatformBadge;

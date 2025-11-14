// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import {useVersions} from '@docusaurus/plugin-content-docs/client';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';

export default function Versions() {
  const {siteConfig} = useDocusaurusContext();
  const allVersions = useVersions('default');

  // Filter out unstable versions, "next", and "current" to get stable versions
  const stableVersions = allVersions.filter(v =>
    !v.name.includes('unstable') &&
    v.name !== 'next' &&
    v.name !== 'current'
  );

  // Find the lastVersion (current stable) - it's the one with isLast: true or the first stable version
  const lastVersion = stableVersions.find(v => v.isLast) || stableVersions[0];
  const pastVersions = stableVersions.filter(v => v !== lastVersion);

  return (
    <Layout
      title="Versions"
      description="Haystack Documentation Versions">
      <main className="container margin-vert--xl">
        <h1>Haystack Documentation Versions</h1>

        <div className="margin-bottom--lg">
          <h2 id="next">Next (Unreleased)</h2>
          <p>Documentation for the unreleased version.</p>
          <ul>
            <li>
              <Link to="/docs/next/intro">Documentation</Link>
            </li>
          </ul>
        </div>

        <div className="margin-bottom--lg">
          <h2 id="latest">Current version (Stable)</h2>
          <p>Documentation for the current stable release (v{lastVersion?.name || lastVersion?.label}).</p>
          <ul>
            <li>
              <Link to="/docs/intro">Documentation</Link>
            </li>
          </ul>
        </div>

        {pastVersions.length > 0 && (
          <div className="margin-bottom--lg">
            <h2 id="archive">Past Versions</h2>
            <p>
              Here you can find documentation for previous versions of Haystack.
            </p>
            <ul>
              {pastVersions.map((version) => (
                <li key={version.name}>
                  <Link to={`/docs/${version.name}/intro`}>
                    Version {version.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        )}
      </main>
    </Layout>
  );
}

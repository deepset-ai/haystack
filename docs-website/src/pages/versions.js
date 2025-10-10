// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';

export default function Versions() {
  const {siteConfig} = useDocusaurusContext();
  const versions = require('../../versions.json');
  const currentVersion = versions[0];

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
              <Link to="/docs/next/overview/intro">Documentation</Link>
            </li>
          </ul>
        </div>

        <div className="margin-bottom--lg">
          <h2 id="latest">Current version (Stable)</h2>
          <p>Documentation for the current stable release (v{currentVersion}).</p>
          <ul>
            <li>
              <Link to="/docs/overview/intro">Documentation</Link>
            </li>
          </ul>
        </div>

        {versions.length > 1 && (
          <div className="margin-bottom--lg">
            <h2 id="archive">Past Versions</h2>
            <p>
              Here you can find documentation for previous versions of Haystack.
            </p>
            <ul>
              {versions.slice(1).map((version) => (
                <li key={version}>
                  <Link to={`/docs/${version}/Overview/get-started`}>
                    Version {version}
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

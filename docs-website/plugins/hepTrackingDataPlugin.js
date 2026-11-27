// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Exposes per-page tracking metadata (top-level sidebar category, `hep_available` frontmatter)
 * to `src/clientModules/hepTracking.js` through Docusaurus global data, so it also works
 * when the sidebar is not rendered (mobile) and for pages outside the sidebar.
 */

const DOCS_PLUGIN = 'docusaurus-plugin-content-docs';

function collectDocIds(items, ids = []) {
  for (const item of items) {
    if (item.type === 'doc' || item.type === 'ref') {
      ids.push(item.id);
    } else if (item.type === 'category') {
      if (item.link?.type === 'doc') {
        ids.push(item.link.id);
      }
      collectDocIds(item.items, ids);
    }
  }
  return ids;
}

export default function hepTrackingDataPlugin() {
  return {
    name: 'hep-tracking-data',
    async allContentLoaded({allContent, actions}) {
      const docsContent = allContent[DOCS_PLUGIN]?.default;
      const pages = {};
      for (const version of docsContent?.loadedVersions ?? []) {
        const sectionByDocId = {};
        for (const sidebar of Object.values(version.sidebars ?? {})) {
          for (const item of sidebar) {
            if (item.type === 'category') {
              for (const id of collectDocIds([item])) {
                sectionByDocId[id] = item.label;
              }
            } else if (item.type === 'doc') {
              sectionByDocId[item.id] = item.label ?? item.id;
            }
          }
        }
        for (const doc of version.docs) {
          pages[doc.permalink] = {
            section: sectionByDocId[doc.id] ?? 'Other',
            hep: doc.frontMatter?.hep_available === true,
          };
        }
      }
      const latest = docsContent?.loadedVersions.find((v) => v.isLast);
      actions.setGlobalData({pages, latestVersion: latest?.versionName});
    },
  };
}

// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

// @ts-check

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Haystack Documentation',
  tagline: 'Haystack Docs',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://docs.haystack.deepset.ai',
  baseUrl: '/',

  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',
  onDuplicateRoutes: 'throw',

  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
           // Exclude internal templates from the docs build
           exclude: ['**/_templates/**'],
          editUrl:
            'https://github.com/deepset-ai/haystack/tree/main/docs-website/',
          // Use beforeDefaultRemarkPlugins to ensure our plugin runs before Webpack processes links
          beforeDefaultRemarkPlugins: [require('./src/remark/versionedReferenceLinks')],
          versions: {
            current: {
              label: '2.21-unstable',
              path: 'next',
              banner: 'unreleased',
            },
          },
          lastVersion: '2.20',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  plugins: [
    [
      '@docusaurus/plugin-ideal-image',
      {
        quality: 70,
        max: 1030,
        min: 640,
        steps: 2,
        disableInDev: false,
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'reference',
        path: 'reference',
        routeBasePath: 'reference',
        sidebarPath: './reference-sidebars.js',
        editUrl: 'https://github.com/deepset-ai/haystack/tree/main/docs-website/',
        // Use beforeDefaultRemarkPlugins to ensure our plugin runs before Webpack processes links
        beforeDefaultRemarkPlugins: [require('./src/remark/versionedReferenceLinks')],
        showLastUpdateAuthor: false,
        showLastUpdateTime: false,
        exclude: ['**/_templates/**'],
        versions: {
          current: {
            label: '2.21-unstable',
            path: 'next',
            banner: 'unreleased',
          },
        },
        lastVersion: '2.20',
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      docs: {
        sidebar: {
          autoCollapseCategories: true,
        },
      },
      navbar: {
        title: 'Haystack Documentation',
        logo: {
          alt: 'Haystack Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docsVersionDropdown',
            position: 'left',
            dropdownActiveClassDisabled: true,
          },
          {
            type: 'doc',
            docId: 'intro',
            label: 'Docs',
            position: 'left',
          },
            {
              type: 'doc',
              docsPluginId: 'reference',
              docId: 'api-index',
              label: 'API Reference',
              position: 'left',
            },
          {
            href: 'https://github.com/deepset-ai/haystack/tree/main/docs-website',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Community',
            items: [
              { label: 'Discord',   href: 'https://discord.com/invite/haystack' },
              { label: 'GitHub',    href: 'https://github.com/deepset-ai/haystack' },
              { label: 'Twitter',   href: 'https://twitter.com/haystack_ai' },
            ],
          },
          {
            title: 'Learn',
            items: [
              { label: 'Tutorials',   href: 'https://haystack.deepset.ai/tutorials' },
              { label: 'Cookbooks', href: 'https://haystack.deepset.ai/cookbook' },
            ],
          },
          {
            title: 'More',
            items: [
              { label: 'Integrations',   href: 'https://haystack.deepset.ai/integrations' },
              { label: 'Studio', href: 'https://landing.deepset.ai/deepset-studio-signup' },
            ],
          },
          {
            title: 'Company',
            items: [
              { label: 'About',   href: 'https://deepset.ai/about' },
              { label: 'Careers', href: 'https://deepset.ai/careers' },
              { label: 'Blog',    href: 'https://deepset.ai/blog' },
            ],
          },
        ],
        copyright: `© ${new Date().getFullYear()} deepset GmbH. All rights reserved.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'docker'],
      },
    }),
};

export default config;

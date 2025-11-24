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

  future: {
    experimental_faster: true,
    v4: true,
  },

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
    [
      'docusaurus-plugin-generate-llms-txt',
      {
        // defaults to "llms.txt", but set explicitly for clarity
        outputFile: 'llms.txt',
      },
    ],
    // Local plugin to teach Webpack how to handle `.txt` files like `llms.txt`
    require.resolve('./plugins/txtLoaderPlugin'),
    ['@cmfcmf/docusaurus-search-local',      {
      includeParentCategoriesInPageTitle: true,
      indexDocSidebarParentCategories: 1,
      lunr: {
        titleBoost: 1,
        contentBoost: 1,
        tagsBoost: 3,
        parentCategoriesBoost: 5,
      },
    },
    ]
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
            dropdownItemsAfter: [
              {
                type: 'html',
                value: '<hr style="margin: 0.3rem 0;">',
              },
              {
                href: '/docs/faq#where-can-i-find-tutorials-and-documentation-for-haystack-1x',
                label: '1.x archived documentation',
              },
              {
                href: '/docs/faq#where-is-the-documentation-for-haystack-217-and-older',
                label: '2.x archived documentation',
              },
            ],
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
            href: 'https://github.com/deepset-ai/haystack/blob/main/docs-website/CONTRIBUTING.md',
            label: 'Contribute',
            position: 'right',
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
              {
                html: '<div class="footer-social-icons-container"><div class="footer-social-row"><a href="https://discord.com/invite/haystack" target="_blank" rel="noopener noreferrer" class="footer__link-item" aria-label="Discord"><img src="/img/discord.svg" alt="Discord" class="footer-social-icon" /></a><a href="https://github.com/deepset-ai/haystack" target="_blank" rel="noopener noreferrer" class="footer__link-item" aria-label="GitHub"><img src="/img/github.svg" alt="GitHub" class="footer-social-icon" /></a><a href="https://x.com/haystack_ai" target="_blank" rel="noopener noreferrer" class="footer__link-item" aria-label="X"><img src="/img/x.svg" alt="X" class="footer-social-icon" /></a></div><div class="footer-social-row"><a href="https://www.linkedin.com/company/deepset-ai/" target="_blank" rel="noopener noreferrer" class="footer__link-item" aria-label="LinkedIn"><img src="/img/linkedin.svg" alt="LinkedIn" class="footer-social-icon" /></a><a href="https://www.youtube.com/channel/UC5dfn9m310oyt-cbeegfvZw" target="_blank" rel="noopener noreferrer" class="footer__link-item" aria-label="YouTube"><img src="/img/youtube.svg" alt="YouTube" class="footer-social-icon" /></a></div></div>'
              },
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
          {
            title: 'Legal',
            items: [
              { label: 'Privacy Policy', href: 'https://www.deepset.ai/privacy-policy' },
              { label: 'Imprint', href: 'https://www.deepset.ai/imprint' },
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

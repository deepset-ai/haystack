// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

// Makes the built site readable by AI agents:
//  - writes a Markdown copy of every docs page next to its HTML (`/docs/intro` -> `/docs/intro.md`),
//  - writes `/llms.txt` (https://llmstxt.org) as an index of the latest stable docs, linking the `.md` copies.
// Markdown is produced from the built HTML, so MDX components, tabs and admonitions are already resolved.

const fs = require('fs');
const path = require('path');
const cheerio = require('cheerio');
const TurndownService = require('turndown');

const DOCS_PLUGIN = 'docusaurus-plugin-content-docs';
const AGENT_NOTICE_CLASS = 'agent-markdown-notice';
// Elements that are page chrome, not content.
const STRIP_SELECTOR = [
  'script',
  'style',
  'nav',
  'button',
  'svg',
  '.hash-link',
  '.code-block-language-badge',
  '.theme-doc-footer',
  '.copy-dropdown-container',
  '.copy-dropdown-sticky',
  '.table-of-contents',
  '.pagination-nav',
  '.theme-doc-breadcrumbs',
  '.theme-doc-version-badge',
  '.theme-doc-version-banner',
  `.${AGENT_NOTICE_CLASS}`,
].join(',');

function createTurndown() {
  const turndown = new TurndownService({
    headingStyle: 'atx',
    codeBlockStyle: 'fenced',
    bulletListMarker: '-',
    emDelimiter: '*',
    strongDelimiter: '**',
  });

  turndown.addRule('fencedCodeBlock', {
    filter: (node) => node.nodeName === 'PRE' && node.querySelector('code'),
    replacement: (_content, node) => {
      const language = (node.className.match(/language-(\S+)/) || [])[1] || '';
      // Token lines are <div>s ending in <br>, so rebuild the newlines from them.
      const lines = node.querySelectorAll('.token-line');
      const code = lines.length
        ? Array.from(lines, (line) => line.textContent.replace(/\n$/, '')).join('\n')
        : node.textContent;
      return `\n\`\`\`${language}\n${code.replace(/\n+$/, '')}\n\`\`\`\n\n`;
    },
  });

  turndown.addRule('admonition', {
    filter: (node) => node.nodeName === 'DIV' && node.classList.contains('theme-admonition'),
    replacement: (content, node) => {
      const type = (node.className.match(/theme-admonition-(\w+)/) || [])[1] || 'note';
      const quoted = content.trim().replace(/\n/g, '\n> ');
      return `\n> **${type.toUpperCase()}:** ${quoted}\n\n`;
    },
  });

  return turndown;
}

// Inactive tab panels are in the HTML but hidden; turn every tab into a labelled section.
function flattenTabs($) {
  $('.tabs-container').each((_, container) => {
    const $container = $(container);
    const labels = $container
      .find('[role="tab"]')
      .map((__, tab) => $(tab).text().trim())
      .get();
    const sections = $container
      .find('[role="tabpanel"]')
      .map((index, panel) => {
        const $panel = $(panel).removeAttr('hidden');
        return `<div><p><strong>${labels[index] || `Tab ${index + 1}`}</strong></p>${$panel.html()}</div>`;
      })
      .get();
    $container.replaceWith(sections.join(''));
  });
}

function htmlToMarkdown(html, turndown, siteUrl) {
  const $ = cheerio.load(html);
  const $article = $('article').first();
  if (!$article.length) {
    return null;
  }
  $article.find(STRIP_SELECTOR).remove();
  flattenTabs($);
  // Agents read the file out of context, so site-relative links must be absolute; doc links point at the .md copy.
  $article.find('a[href^="/"], img[src^="/"]').each((_, el) => {
    const attr = el.name === 'a' ? 'href' : 'src';
    const value = $(el).attr(attr);
    const [, pathname, suffix] = value.match(/^([^#?]*)(.*)$/);
    const isDocPage = el.name === 'a' && /^\/(docs|reference)\/(.*\/)?[^/.]+$/.test(pathname);
    $(el).attr(attr, `${siteUrl}${pathname}${isDocPage ? '.md' : ''}${suffix}`);
  });
  const body = turndown.turndown($article.find('.theme-doc-markdown').html() || $article.html());
  return body.replace(/\n{3,}/g, '\n\n').trim();
}

function htmlFileFor(outDir, permalink) {
  const clean = permalink.replace(/\/+$/, '');
  return [path.join(outDir, clean, 'index.html'), path.join(outDir, `${clean}.html`)].find((file) =>
    fs.existsSync(file),
  );
}

function markdownFileFor(outDir, permalink) {
  const clean = permalink.replace(/\/+$/, '');
  return path.join(outDir, clean ? `${clean}.md` : 'index.md');
}

function cleanDescription(description) {
  return (description || '').replace(/\s+/g, ' ').trim();
}

function truncate(text, max) {
  return text.length <= max ? text : `${text.slice(0, max - 1).trimEnd()}…`;
}

// Walks a sidebar into [{title, docs: [doc, ...]}], one section per top-level category.
function sectionsFromSidebar(version) {
  const docsById = new Map(version.docs.map((doc) => [doc.id, doc]));
  const seen = new Set();
  const collect = (items, into) => {
    for (const item of items) {
      const docId = item.type === 'category' ? item.link && item.link.id : item.docId || item.id;
      const doc = docId && docsById.get(docId);
      if (doc && !seen.has(doc.id)) {
        seen.add(doc.id);
        into.push(doc);
      }
      if (item.type === 'category') {
        collect(item.items, into);
      }
    }
  };

  const sections = [];
  const topLevel = Object.values(version.sidebars || {}).flat();
  const loose = { title: null, docs: [] };
  for (const item of topLevel) {
    if (item.type === 'category') {
      const section = { title: item.label, docs: [] };
      collect([item], section.docs);
      sections.push(section);
    } else {
      collect([item], loose.docs);
    }
  }
  if (loose.docs.length) {
    sections.unshift({ title: 'Overview', docs: loose.docs });
  }
  const unlisted = version.docs.filter((doc) => !seen.has(doc.id) && !doc.unlisted);
  return { sections, unlisted };
}

function buildLlmsTxt({ siteConfig, latest }) {
  const lines = [
    `# ${siteConfig.title}`,
    '',
    '> Haystack is an open-source Python framework by deepset for building production-ready LLM applications: ' +
      'agents, retrieval-augmented generation (RAG), question answering and semantic search. ' +
      'This index lists the documentation of the latest stable version. ' +
      'Every page is also available as Markdown by appending `.md` to its URL ' +
      'or by requesting it with the `Accept: text/markdown` header.',
    '',
  ];

  for (const { label, version, withDescriptions } of latest) {
    const { sections, unlisted } = sectionsFromSidebar(version);
    if (unlisted.length) {
      sections.push({ title: 'Other', docs: unlisted });
    }
    for (const section of sections.filter((s) => s.docs.length)) {
      lines.push(`## ${label}: ${section.title}`, '');
      for (const doc of section.docs) {
        const description = withDescriptions ? cleanDescription(doc.description) : '';
        const suffix = description ? `: ${truncate(description, 90)}` : '';
        lines.push(`- [${doc.title}](${siteConfig.url}${markdownPermalink(doc.permalink)})${suffix}`);
      }
      lines.push('');
    }
  }
  return `${lines.join('\n').trimEnd()}\n`;
}

function markdownPermalink(permalink) {
  const clean = permalink.replace(/\/+$/, '');
  return clean ? `${clean}.md` : '/index.md';
}

module.exports = function markdownForAgents() {
  let allContent = {};
  return {
    name: 'markdown-for-agents',

    // postBuild does not receive other plugins' content, so capture it here.
    allContentLoaded({ allContent: loaded }) {
      allContent = loaded;
    },

    async postBuild({ outDir, siteConfig }) {
      const turndown = createTurndown();
      const docsContent = allContent[DOCS_PLUGIN] || {};
      const latest = [];

      let written = 0;
      for (const [pluginId, content] of Object.entries(docsContent)) {
        for (const version of content.loadedVersions) {
          for (const doc of version.docs) {
            const htmlFile = htmlFileFor(outDir, doc.permalink);
            if (!htmlFile) {
              continue;
            }
            const body = htmlToMarkdown(fs.readFileSync(htmlFile, 'utf8'), turndown, siteConfig.url);
            if (!body) {
              continue;
            }
            const pageUrl = `${siteConfig.url}${doc.permalink}`;
            const markdown =
              `> Documentation index: ${siteConfig.url}/llms.txt\n\n` +
              `Source: ${pageUrl}\n\n${body}\n`;
            const target = markdownFileFor(outDir, doc.permalink);
            fs.mkdirSync(path.dirname(target), { recursive: true });
            fs.writeFileSync(target, markdown);
            written += 1;
          }
        }
        const last = content.loadedVersions.find((version) => version.isLast);
        if (last) {
          latest.push({
            label: pluginId === 'default' ? 'Docs' : 'API reference',
            version: last,
            // API reference descriptions are boilerplate and would push llms.txt past agent size limits.
            withDescriptions: pluginId === 'default',
          });
        }
      }

      fs.writeFileSync(path.join(outDir, 'llms.txt'), buildLlmsTxt({ siteConfig, latest }));
      console.log(`[markdown-for-agents] wrote ${written} Markdown pages and llms.txt`);
    },
  };
};

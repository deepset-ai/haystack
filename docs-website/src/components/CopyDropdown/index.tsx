// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import React, { useState, useRef, useEffect, useMemo } from 'react';
import TurndownService from 'turndown';
import styles from './styles.module.css';

// Icon imports
import CopyIcon from '@site/static/img/copy.svg';
import ChevronDownIcon from '@site/static/img/chevron-down.svg';
import MarkdownIcon from '@site/static/img/markdown.svg';
import PDFIcon from '@site/static/img/pdf.svg';

// Create and configure Turndown service
function createTurndownService(): TurndownService {
  const turndownService = new TurndownService({
    headingStyle: 'atx',
    codeBlockStyle: 'fenced',
    bulletListMarker: '-',
    emDelimiter: '*',
    strongDelimiter: '**',
  });

  // Custom rule for code blocks with language detection
  turndownService.addRule('fencedCodeBlock', {
    filter: (node) => {
      return (
        node.nodeName === 'PRE' &&
        node.firstChild !== null &&
        node.firstChild.nodeName === 'CODE'
      );
    },
    replacement: (_content, node) => {
      const codeElement = node.firstChild as HTMLElement;
      const className = codeElement.className || '';
      const language = className.match(/language-(\S+)/)?.[1] || '';
      const code = codeElement.textContent || '';
      return `\n\`\`\`${language}\n${code.trim()}\n\`\`\`\n\n`;
    },
  });

  // Custom rule for inline code
  turndownService.addRule('inlineCode', {
    filter: (node) => {
      return (
        node.nodeName === 'CODE' &&
        node.parentNode !== null &&
        node.parentNode.nodeName !== 'PRE'
      );
    },
    replacement: (content) => {
      return `\`${content}\``;
    },
  });

  // Custom rule for Docusaurus admonitions
  turndownService.addRule('admonition', {
    filter: (node) => {
      return (
        node.nodeName === 'DIV' &&
        (node as HTMLElement).classList.contains('theme-admonition')
      );
    },
    replacement: (content, node) => {
      const element = node as HTMLElement;
      const type = element.classList.contains('alert--warning')
        ? 'warning'
        : element.classList.contains('alert--danger')
          ? 'danger'
          : element.classList.contains('alert--info')
            ? 'info'
            : element.classList.contains('alert--success')
              ? 'tip'
              : 'note';
      return `\n> **${type.toUpperCase()}:** ${content.trim()}\n\n`;
    },
  });

  // Remove unwanted elements using a filter function
  turndownService.remove((node) => {
    if (node.nodeType !== Node.ELEMENT_NODE) return false;
    const element = node as HTMLElement;
    const tagName = element.tagName.toLowerCase();
    
    // Remove by tag name
    if (['script', 'style', 'nav'].includes(tagName)) return true;
    
    // Remove by class name
    const classesToRemove = [
      'theme-doc-footer',
      'copy-dropdown-container',
      'table-of-contents',
      'pagination-nav',
      'theme-doc-breadcrumbs',
      'theme-doc-version-badge',
      'hash-link',
    ];
    
    return classesToRemove.some((cls) => element.classList.contains(cls));
  });

  return turndownService;
}

interface CopyDropdownProps {
  className?: string;
}

export default function CopyDropdown({ className }: CopyDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const [docTitle, setDocTitle] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Memoize the Turndown service instance
  const turndownService = useMemo(() => createTurndownService(), []);

  // Set the document title from the page
  useEffect(() => {
    if (typeof document !== 'undefined') {
      // Get title from h1 or document title
      const h1 = document.querySelector('article h1, .markdown h1');
      const title = h1?.textContent || document.title.split(' | ')[0] || document.title;
      setDocTitle(title);
    }
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Get the page content as markdown using Turndown
  const getPageMarkdown = (): string => {
    const article = document.querySelector('article');
    if (!article) return '';

    // Clone the article to manipulate without affecting the page
    const clone = article.cloneNode(true) as HTMLElement;

    // Remove elements we don't want before conversion
    clone
      .querySelectorAll(
        '.theme-doc-footer, .copy-dropdown-container, script, style, .hash-link, .table-of-contents'
      )
      .forEach((el) => el.remove());

    // Convert HTML to Markdown using Turndown
    const content = turndownService.turndown(clone);

    // Build the final markdown with header
    let markdown = `# ${docTitle}\n\n`;
    markdown += `URL: ${window.location.href}\n\n`;
    markdown += '---\n\n';
    markdown += content;

    return markdown.trim();
  };

  const handleCopyPage = async () => {
    const markdown = getPageMarkdown();
    try {
      await navigator.clipboard.writeText(markdown);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
    setIsOpen(false);
  };

  const handleViewMarkdown = () => {
    const markdown = getPageMarkdown();
    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    window.open(url, '_blank');
    setIsOpen(false);
  };

  const handleExportPDF = () => {
    window.print();
    setIsOpen(false);
  };

  return (
    <div className={`${styles.copyDropdownContainer} ${className || ''}`} ref={dropdownRef}>
      <button
        className={`${styles.copyButton} ${copySuccess ? styles.success : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-haspopup="true"
      >
        <CopyIcon />
        <span>{copySuccess ? 'Copied!' : 'Copy'}</span>
        <ChevronDownIcon />
      </button>
      
      {isOpen && (
        <div className={styles.dropdownMenu}>
          <button className={styles.menuItem} onClick={handleCopyPage}>
            <CopyIcon />
            <div className={styles.menuItemContent}>
              <span className={styles.menuItemTitle}>Copy page</span>
              <span className={styles.menuItemDescription}>Copy page as Markdown for LLMs</span>
            </div>
          </button>
          
          <button className={styles.menuItem} onClick={handleViewMarkdown}>
            <MarkdownIcon />
            <div className={styles.menuItemContent}>
              <span className={styles.menuItemTitle}>View as Markdown</span>
              <span className={styles.menuItemDescription}>View this page as plain text</span>
            </div>
          </button>
          
          <div className={styles.menuDivider} />
          
          <button className={styles.menuItem} onClick={handleExportPDF}>
            <PDFIcon />
            <div className={styles.menuItemContent}>
              <span className={styles.menuItemTitle}>Export as PDF</span>
              <span className={styles.menuItemDescription}>Save this page as a PDF file</span>
            </div>
          </button>
        </div>
      )}
    </div>
  );
}

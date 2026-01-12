// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import React, { useState, useRef, useEffect } from 'react';
import styles from './styles.module.css';

// Icon components
const CopyIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </svg>
);

const ChevronDownIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

const MarkdownIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
    <polyline points="10 9 9 9 8 9" />
  </svg>
);

const PDFIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <path d="M9 15v-2h2c.6 0 1 .4 1 1s-.4 1-1 1H9z" />
    <path d="M9 11h2" />
  </svg>
);

interface CopyDropdownProps {
  className?: string;
}

export default function CopyDropdown({ className }: CopyDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const [docTitle, setDocTitle] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);

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

  // Get the page content as markdown
  const getPageMarkdown = () => {
    const article = document.querySelector('article');
    if (!article) return '';

    // Clone the article to manipulate without affecting the page
    const clone = article.cloneNode(true) as HTMLElement;
    
    // Remove elements we don't want
    clone.querySelectorAll('.theme-doc-footer, .copy-dropdown-container, script, style').forEach(el => el.remove());
    
    // Get text content with basic markdown-like formatting
    let markdown = `# ${docTitle}\n\n`;
    markdown += `URL: ${window.location.href}\n\n`;
    markdown += '---\n\n';
    
    // Process headers and content
    const processNode = (node: Node): string => {
      if (node.nodeType === Node.TEXT_NODE) {
        return node.textContent || '';
      }
      
      if (node.nodeType !== Node.ELEMENT_NODE) return '';
      
      const el = node as HTMLElement;
      const tagName = el.tagName.toLowerCase();
      
      // Skip hidden elements
      if (el.style.display === 'none' || el.hidden) return '';
      
      let result = '';
      
      switch (tagName) {
        case 'h1':
          result = `# ${el.textContent?.trim()}\n\n`;
          break;
        case 'h2':
          result = `## ${el.textContent?.trim()}\n\n`;
          break;
        case 'h3':
          result = `### ${el.textContent?.trim()}\n\n`;
          break;
        case 'h4':
          result = `#### ${el.textContent?.trim()}\n\n`;
          break;
        case 'h5':
          result = `##### ${el.textContent?.trim()}\n\n`;
          break;
        case 'h6':
          result = `###### ${el.textContent?.trim()}\n\n`;
          break;
        case 'p':
          result = `${el.textContent?.trim()}\n\n`;
          break;
        case 'ul':
        case 'ol':
          Array.from(el.children).forEach((li, index) => {
            const prefix = tagName === 'ol' ? `${index + 1}. ` : '- ';
            result += `${prefix}${li.textContent?.trim()}\n`;
          });
          result += '\n';
          break;
        case 'pre':
        case 'code':
          if (tagName === 'pre' || el.parentElement?.tagName.toLowerCase() !== 'pre') {
            result = `\`\`\`\n${el.textContent?.trim()}\n\`\`\`\n\n`;
          }
          break;
        case 'a':
          result = `[${el.textContent?.trim()}](${el.getAttribute('href')})`;
          break;
        case 'strong':
        case 'b':
          result = `**${el.textContent?.trim()}**`;
          break;
        case 'em':
        case 'i':
          result = `*${el.textContent?.trim()}*`;
          break;
        case 'blockquote':
          result = `> ${el.textContent?.trim()}\n\n`;
          break;
        default:
          Array.from(el.childNodes).forEach(child => {
            result += processNode(child);
          });
      }
      
      return result;
    };
    
    Array.from(clone.childNodes).forEach(node => {
      markdown += processNode(node);
    });
    
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

import React, {useEffect, useState} from 'react';
import Root from '@theme-original/Root';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

export default function RootWrapper(props) {
  const {siteConfig} = useDocusaurusContext();

  // Determine product from URL path or environment
  const getProduct = () => {
    if (typeof window !== 'undefined') {
      const path = window.location.pathname;
      if (path.startsWith('/professional')) return 'professional';
      if (path.startsWith('/enterprise')) return 'enterprise';
    }
    return siteConfig?.customFields?.docsProduct ||
      (typeof process !== 'undefined' && process.env.DOCS_PRODUCT) ||
      'haystack';
  };

  const [product, setProduct] = useState(getProduct);

  // Update product when location changes
  useEffect(() => {
    const handleLocationChange = () => {
      const newProduct = getProduct();
      console.log('[Root.js] Location changed, setting product to:', newProduct);
      setProduct(newProduct);
    };

    // Listen for route changes
    if (typeof window !== 'undefined') {
      window.addEventListener('popstate', handleLocationChange);
      // Also check on initial load and navigation
      handleLocationChange();
    }

    return () => {
      if (typeof window !== 'undefined') {
        window.removeEventListener('popstate', handleLocationChange);
      }
    };
  }, []);

  useEffect(() => {
    console.log('[Root.js] Current product:', product);
    console.log('[Root.js] Current path:', typeof window !== 'undefined' ? window.location.pathname : 'N/A');
    if (typeof document === 'undefined') {
      return;
    }

    document.documentElement.setAttribute('data-docs-product', product);

    const filterToc = () => {
      console.log('[Root.js] filterToc called, product:', product);
      const items = document.querySelectorAll('.table-of-contents__item');
      console.log('[Root.js] Found TOC items:', items.length);

      items.forEach(item => {
        const link = item.querySelector('.table-of-contents__link');
        if (!link) {
          return;
        }
        const href = link.getAttribute('href') || '';
        const isEnterprise = href.startsWith('#enterprise-');
        const isProfessional = href.startsWith('#professional-');

        console.log('[Root.js] TOC item:', {href, isEnterprise, isProfessional, product});

        if (
          (product === 'professional' && isEnterprise) ||
          (product === 'enterprise' && isProfessional)
        ) {
          console.log('[Root.js] Removing TOC item:', href);
          item.parentElement?.removeChild(item);
        }
      });
    };

    filterToc();

    const observer = new MutationObserver(filterToc);
    observer.observe(document.body, {childList: true, subtree: true});

    return () => {
      document.documentElement.removeAttribute('data-docs-product');
      observer.disconnect();
    };
  }, [product]);

  return <Root {...props} />;
}

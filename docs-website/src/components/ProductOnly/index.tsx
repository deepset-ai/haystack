// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

type ProductOnlyProps = {
  product?: string;
  products?: string[];
  children: React.ReactNode;
};

const ProductOnly: React.FC<ProductOnlyProps> = ({product, products, children}) => {
  const {siteConfig} = useDocusaurusContext();
  const currentProduct =
    siteConfig?.customFields?.docsProduct ??
    (typeof process !== 'undefined' && process.env?.DOCS_PRODUCT) ??
    'haystack';

  // Support both single product and multiple products
  const allowedProducts = products || (product ? [product] : []);
  const isVisible = allowedProducts.includes(currentProduct);

  // Create a data attribute with all products for CSS targeting
  const productList = allowedProducts.join(',');

  // Always render content but use CSS to hide it
  // This ensures anchors exist for TOC but are visually hidden
  return (
    <div data-product-only={productList} style={{display: isVisible ? 'block' : 'none'}}>
      {children}
    </div>
  );
};

export default ProductOnly;

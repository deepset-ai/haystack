import React from 'react';
import Html from '@theme-original/Html';

export default function HtmlWrapper(props) {
  const docsProduct =
    (props?.siteConfig && props.siteConfig?.customFields?.docsProduct) ||
    (typeof process !== 'undefined' && process.env?.DOCS_PRODUCT) ||
    'haystack';

  return (
    <Html
      {...props}
      htmlAttributes={{
        ...props.htmlAttributes,
        'data-docs-product': docsProduct,
      }}
    />
  );
}

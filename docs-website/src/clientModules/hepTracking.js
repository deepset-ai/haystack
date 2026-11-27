// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import globalData from '@generated/globalData';

// Placement-level tracking for Haystack Enterprise Platform (HEP) CTAs.
// Events go to the GTM dataLayer; GTM/GA4 configuration lives outside this repo.

// Every link to these hosts is a CTA.
const CTA_HOSTS = new Set([
  'cloud.deepset.ai',
  'docs.cloud.deepset.ai',
  'landing.deepset.ai',
]);

// The marketing site also hosts unrelated pages (privacy, imprint, careers, blog...), so only HEP pages count.
const MARKETING_HOSTS = new Set(['deepset.ai', 'www.deepset.ai']);

function isMarketingProductPath(path) {
  return (
    path === '/haystack-platform' ||
    path.startsWith('/haystack-platform/') ||
    /\/haystack-enterprise(-starter)?$/.test(path)
  );
}

// haystack.deepset.ai mostly hosts tutorials and cookbooks, so only its enterprise posts count as CTAs.
const HAYSTACK_SITE_HOST = 'haystack.deepset.ai';
const HAYSTACK_SITE_ENTERPRISE_PATH = /^\/blog\/[^/]*haystack-enterprise/;

// Page slug (last path segment, version prefix stripped) -> callout location
const CALLOUT_LOCATIONS = {
  intro: 'callout_intro',
  deployment: 'callout_deployment',
  'get-started': 'callout_get_started',
  installation: 'callout_installation',
  tracing: 'callout_tracing',
};

const PAGE_LOCATIONS = {
  'deployment-haystack-enterprise-platform': 'hep_deployment_page',
  'tracing-haystack-enterprise-platform': 'hep_tracing_page',
};

const VERSION_PREFIX = /^\/(?:docs|reference)\/(next|\d+\.\d+[^/]*)(?=\/|$)/;
const VIEW_RATIO = 0.5;
const VIEW_MS = 1000;

const trackingData = globalData['hep-tracking-data']?.default ?? {};
const pageData = trackingData.pages ?? {};

function push(payload) {
  window.dataLayer = window.dataLayer || [];
  window.dataLayer.push(payload);
}

function normalizePath(pathname) {
  return pathname.length > 1 ? pathname.replace(/\/+$/, '') : pathname;
}

function getVersion(pathname) {
  const match = VERSION_PREFIX.exec(pathname);
  return match ? match[1] : trackingData.latestVersion ?? 'latest';
}

function getSlug(pathname) {
  return normalizePath(pathname).replace(VERSION_PREFIX, '/docs').split('/').pop();
}

function getSection(pathname) {
  const page = pageData[normalizePath(pathname)] ?? pageData[`${normalizePath(pathname)}/`];
  if (page) {
    return page.section;
  }
  return /^\/reference(\/|$)/.test(pathname) ? 'API Reference' : 'Other';
}

function isHepAvailable(pathname) {
  const page = pageData[normalizePath(pathname)] ?? pageData[`${normalizePath(pathname)}/`];
  return page ? page.hep : false;
}

function getCtaType(url) {
  const path = url.pathname.replace(/\/+$/, '');
  if (url.hostname === HAYSTACK_SITE_HOST) {
    return 'enterprise_support';
  }
  if (url.hostname === 'docs.cloud.deepset.ai') {
    return path.startsWith('/docs/quick-start-guide') ? 'quickstart' : 'platform_docs';
  }
  if (url.hostname === 'cloud.deepset.ai' || url.hostname === 'landing.deepset.ai') {
    return 'other';
  }
  if (path === '/haystack-platform' || path.startsWith('/haystack-platform/')) {
    return 'product_page';
  }
  if (/\/haystack-enterprise(-starter)?$/.test(path)) {
    return 'enterprise_support';
  }  return 'other';
}

function getCtaLocation(anchor, pathname) {
  if (anchor.closest('.navbar')) {
    return 'navbar';
  }
  if (anchor.closest('footer')) {
    return 'footer';
  }
  const slug = getSlug(pathname);
  if (anchor.closest('.theme-admonition')) {
    return CALLOUT_LOCATIONS[slug] ?? 'callout';
  }
  if (PAGE_LOCATIONS[slug]) {
    return PAGE_LOCATIONS[slug];
  }
  if (slug === 'platform-components') {
    const firstParagraph = document.querySelector('.theme-doc-markdown p');
    if (firstParagraph && firstParagraph.contains(anchor)) {
      return 'platform_components_intro';
    }
  }
  return 'inline';
}

function toCtaUrl(anchor) {
  let url;
  try {
    url = new URL(anchor.href, window.location.href);
  } catch {
    return null;
  }
  const isCta =
    CTA_HOSTS.has(url.hostname) ||
    (MARKETING_HOSTS.has(url.hostname) && isMarketingProductPath(url.pathname.replace(/\/+$/, ''))) ||
    (url.hostname === HAYSTACK_SITE_HOST && HAYSTACK_SITE_ENTERPRISE_PATH.test(url.pathname));
  return isCta ? url : null;
}

function handleClick(event) {
  if (event.type === 'auxclick' && event.button !== 1) {
    return;
  }
  const target = event.target instanceof Element ? event.target : null;
  const anchor = target?.closest('a[href]');
  if (!anchor) {
    return;
  }
  const {pathname} = window.location;
  const docsVersion = getVersion(pathname);

  if (anchor.dataset.cta === 'badge') {
    push({
      event: 'hep_badge_click',
      component_name: anchor.dataset.component,
      docs_version: docsVersion,
    });
    return;
  }

  const url = toCtaUrl(anchor);
  if (!url) {
    return;
  }
  // Synchronous push, no preventDefault: the outbound navigation is never delayed.
  push({
    event: 'hep_cta_click',
    cta_location: getCtaLocation(anchor, pathname),
    cta_type: getCtaType(url),
    // Drop query and hash: HubSpot appends visitor IDs (__hstc, __hssc, __hsfp) that must not reach GA4.
    link_url: `${url.origin}${url.pathname}`,
    docs_version: docsVersion,
    page_path: pathname,
  });
}

let viewObserver = null;
let viewTimers = new Map();
let firedViews = new Set();

function resetViewTracking() {
  viewObserver?.disconnect();
  viewObserver = null;
  viewTimers.forEach((timer) => clearTimeout(timer));
  viewTimers = new Map();
  firedViews = new Set();
}

function fireView(element, pathname) {
  if (element.dataset.cta === 'badge') {
    const key = 'badge';
    if (!firedViews.has(key)) {
      firedViews.add(key);
      push({event: 'hep_badge_view', component_name: element.dataset.component, docs_version: getVersion(pathname)});
    }
    return;
  }
  const url = toCtaUrl(element);
  if (!url) {
    return;
  }
  const location = getCtaLocation(element, pathname);
  if (firedViews.has(location)) {
    return;
  }
  firedViews.add(location);
  push({
    event: 'hep_cta_view',
    cta_location: location,
    docs_version: getVersion(pathname),
    page_path: pathname,
  });
}

function observeViews(pathname) {
  resetViewTracking();
  if (!('IntersectionObserver' in window)) {
    return;
  }
  viewObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        const element = entry.target;
        if (entry.isIntersecting && entry.intersectionRatio >= VIEW_RATIO) {
          if (!viewTimers.has(element)) {
            viewTimers.set(
              element,
              setTimeout(() => {
                viewTimers.delete(element);
                viewObserver?.unobserve(element);
                fireView(element, pathname);
              }, VIEW_MS),
            );
          }
        } else if (viewTimers.has(element)) {
          clearTimeout(viewTimers.get(element));
          viewTimers.delete(element);
        }
      }
    },
    {threshold: [0, VIEW_RATIO]},
  );
  document.querySelectorAll('a[href]').forEach((anchor) => {
    if (anchor.dataset.cta === 'badge' || (toCtaUrl(anchor) && !anchor.closest('.navbar'))) {
      viewObserver.observe(anchor);
    }
  });
}

if (ExecutionEnvironment.canUseDOM) {
  document.addEventListener('click', handleClick);
  document.addEventListener('auxclick', handleClick);
}

export function onRouteDidUpdate({location, previousLocation}) {
  const {pathname} = location;
  if (previousLocation && previousLocation.pathname === pathname) {
    return;
  }
  push({
    event: 'docs_page_context',
    docs_section: getSection(pathname),
    docs_version: getVersion(pathname),
    hep_available: isHepAvailable(pathname),
    page_path: pathname,
  });
  // Let React commit the new page before scanning for CTAs.
  setTimeout(() => observeViews(pathname), 0);
}

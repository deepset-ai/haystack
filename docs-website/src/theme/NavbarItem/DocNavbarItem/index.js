import React, {useMemo} from 'react';
import {useLocation} from '@docusaurus/router';
import {
  useActiveDocContext,
  useActivePluginAndVersion,
  useLayoutDoc,
} from '@docusaurus/plugin-content-docs/client';
import DefaultNavbarItem from '@theme/NavbarItem/DefaultNavbarItem';

const DEFAULT_PLUGIN_ID = 'default';
const DOCS_PATH_PREFIX = '/docs';
const REFERENCE_PATH_PREFIX = '/reference';

export default function DocNavbarItem({
  docId,
  label: staticLabel,
  docsPluginId,
  activeBasePath,
  ...props
}) {
  const resolvedPluginId = docsPluginId ?? DEFAULT_PLUGIN_ID;
  const {pathname} = useLocation();
  const activePluginAndVersion = useActivePluginAndVersion();
  const activePluginId = activePluginAndVersion?.activePlugin?.pluginId;
  const activePluginMatches =
    !activePluginId || activePluginId === resolvedPluginId;
  const activeDocContext = useActiveDocContext(resolvedPluginId);
  const doc = useLayoutDoc(docId, resolvedPluginId);

  const expectedBasePath = useMemo(() => {
    if (activeBasePath) {
      return activeBasePath;
    }
    return docsPluginId === 'reference' ? REFERENCE_PATH_PREFIX : DOCS_PATH_PREFIX;
  }, [activeBasePath, docsPluginId]);

  const shouldHighlight = useMemo(() => {
    if (!doc || !expectedBasePath || !activePluginMatches) {
      return false;
    }
    if (
      resolvedPluginId === DEFAULT_PLUGIN_ID &&
      pathname.startsWith(REFERENCE_PATH_PREFIX)
    ) {
      return false;
    }
    const normalized = expectedBasePath.endsWith('/')
      ? expectedBasePath
      : `${expectedBasePath}/`;
    const matchesBasePath =
      pathname === expectedBasePath || pathname.startsWith(normalized);
    if (!matchesBasePath) {
      return false;
    }
    const pageActive = activeDocContext?.activeDoc?.path === doc.path;
    const sidebarMatches =
      !!activeDocContext?.activeDoc?.sidebar &&
      activeDocContext.activeDoc.sidebar === doc.sidebar;
    return pageActive || sidebarMatches;
  }, [
    activeDocContext,
    activePluginMatches,
    doc,
    expectedBasePath,
    pathname,
    resolvedPluginId,
  ]);

  if (!doc) {
    return null;
  }

  return (
    <DefaultNavbarItem
      exact
      {...props}
      isActive={() => shouldHighlight}
      label={staticLabel ?? doc.id}
      to={doc.path}
    />
  );
}

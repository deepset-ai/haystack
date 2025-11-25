import React, {useMemo} from 'react';
import {
  useVersions,
  useActiveDocContext,
  useDocsVersionCandidates,
  useDocsPreferredVersion,
  useActivePluginAndVersion,
} from '@docusaurus/plugin-content-docs/client';
import {translate} from '@docusaurus/Translate';
import {useHistorySelector} from '@docusaurus/theme-common';
import DefaultNavbarItem from '@theme/NavbarItem/DefaultNavbarItem';
import DropdownNavbarItem from '@theme/NavbarItem/DropdownNavbarItem';

const DEFAULT_PLUGIN_ID = 'default';
const REFERENCE_PLUGIN_ID = 'reference';

function getVersionItems(versions, configs) {
  if (configs) {
    const versionMap = new Map(versions.map((version) => [version.name, version]));
    const toVersionItem = (name, config) => {
      const version = versionMap.get(name);
      if (!version) {
        throw new Error(
          `No docs version exist for name '${name}', please verify your 'docsVersionDropdown' navbar item versions config.\nAvailable version names:\n- ${versions
            .map((v) => `${v.name}`)
            .join('\n- ')}`,
        );
      }
      return {version, label: config?.label ?? version.label};
    };
    if (Array.isArray(configs)) {
      return configs.map((name) => toVersionItem(name, undefined));
    }
    return Object.entries(configs).map(([name, config]) =>
      toVersionItem(name, config),
    );
  }
  return versions.map((version) => ({version, label: version.label}));
}

function useVersionItems({docsPluginId, configs, sharedVersionNames}) {
  const versions = useVersions(docsPluginId);
  const baseItems = getVersionItems(versions, configs);
  if (!sharedVersionNames) {
    return baseItems;
  }
  const sharedItems = baseItems.filter(({version}) =>
    sharedVersionNames.has(version.name),
  );
  return sharedItems.length > 0 ? sharedItems : baseItems;
}

function getVersionMainDoc(version) {
  return version.docs.find((doc) => doc.id === version.mainDocId);
}

function getVersionTargetDoc(version, activeDocContext) {
  return (
    activeDocContext.alternateDocVersions[version.name] ??
    getVersionMainDoc(version)
  );
}

function useDisplayedVersionItem({docsPluginId, versionItems}) {
  const candidates = useDocsVersionCandidates(docsPluginId);
  const candidateItems = candidates
    .map((candidate) => versionItems.find((vi) => vi.version === candidate))
    .filter((vi) => vi !== undefined);
  return candidateItems[0] ?? versionItems[0] ?? null;
}

function useSynchronizedPreferredVersion(pluginId) {
  const {savePreferredVersionName} = useDocsPreferredVersion(pluginId);
  return savePreferredVersionName;
}

export default function DocsVersionDropdownNavbarItem({
  mobile,
  docsPluginId,
  dropdownActiveClassDisabled,
  dropdownItemsBefore,
  dropdownItemsAfter,
  versions: configs,
  isActive: providedIsActive,
  ...props
}) {
  const activePluginContext = useActivePluginAndVersion();
  const inferredPluginId = activePluginContext?.activePlugin?.pluginId;
  const effectiveDocsPluginId = inferredPluginId ?? docsPluginId ?? DEFAULT_PLUGIN_ID;

  const search = useHistorySelector((history) => history.location.search);
  const hash = useHistorySelector((history) => history.location.hash);
  const activeDocContext = useActiveDocContext(effectiveDocsPluginId);

  const savePreferredForActivePlugin = useSynchronizedPreferredVersion(
    effectiveDocsPluginId,
  );
  const savePreferredForDefault = useSynchronizedPreferredVersion(
    DEFAULT_PLUGIN_ID,
  );
  const savePreferredForReference = useSynchronizedPreferredVersion(
    REFERENCE_PLUGIN_ID,
  );

  const defaultVersions = useVersions(DEFAULT_PLUGIN_ID);
  const referenceVersions = useVersions(REFERENCE_PLUGIN_ID);

  const sharedVersionNames = useMemo(() => {
    if (!defaultVersions.length || !referenceVersions.length) {
      return undefined;
    }
    const referenceNameSet = new Set(referenceVersions.map((version) => version.name));
    const names = defaultVersions
      .map((version) => version.name)
      .filter((name) => referenceNameSet.has(name));
    return new Set(names);
  }, [defaultVersions, referenceVersions]);

  const versionAvailability = useMemo(
    () => ({
      [DEFAULT_PLUGIN_ID]: new Set(
        defaultVersions.map((version) => version.name),
      ),
      [REFERENCE_PLUGIN_ID]: new Set(
        referenceVersions.map((version) => version.name),
      ),
    }),
    [defaultVersions, referenceVersions],
  );

  const versionItems = useVersionItems({
    docsPluginId: effectiveDocsPluginId,
    configs,
    sharedVersionNames,
  });

  const displayedVersionItem = useDisplayedVersionItem({
    docsPluginId: effectiveDocsPluginId,
    versionItems,
  });

  if (!displayedVersionItem) {
    return null;
  }

  const resolvedIsActive = useMemo(() => {
    if (providedIsActive) {
      return providedIsActive;
    }
    if (dropdownActiveClassDisabled) {
      return () => false;
    }
    const currentPluginId = activePluginContext?.activePlugin?.pluginId;
    return () => currentPluginId === effectiveDocsPluginId;
  }, [
    providedIsActive,
    dropdownActiveClassDisabled,
    activePluginContext,
    effectiveDocsPluginId,
  ]);

  function versionItemToLink({version, label}) {
    const targetDoc = getVersionTargetDoc(version, activeDocContext);
    return {
      label,
      to: `${targetDoc.path}${search}${hash}`,
      isActive: () => version === activeDocContext.activeVersion,
      onClick: () => {
        const pluginsToSync = [
          {id: effectiveDocsPluginId, save: savePreferredForActivePlugin},
          {id: DEFAULT_PLUGIN_ID, save: savePreferredForDefault},
          {id: REFERENCE_PLUGIN_ID, save: savePreferredForReference},
        ];
        const seen = new Set();
        pluginsToSync.forEach(({id, save}) => {
          if (!save || seen.has(id)) {
            return;
          }
          seen.add(id);
          const availableNames = versionAvailability[id];
          if (!availableNames || availableNames.has(version.name)) {
            save(version.name);
          }
        });
      },
    };
  }

  const items = [
    ...dropdownItemsBefore,
    ...versionItems.map(versionItemToLink),
    ...dropdownItemsAfter,
  ];

  if (items.length <= 1) {
    const targetDoc = getVersionTargetDoc(
      displayedVersionItem.version,
      activeDocContext,
    );
    return (
      <DefaultNavbarItem
        {...props}
        mobile={mobile}
        label={displayedVersionItem.label}
        to={targetDoc.path}
        isActive={resolvedIsActive}
      />
    );
  }

  const dropdownLabel =
    mobile && items.length > 1
      ? translate({
          id: 'theme.navbar.mobileVersionsDropdown.label',
          message: 'Versions',
          description:
            'The label for the navbar versions dropdown on mobile view',
        })
      : displayedVersionItem.label;

  const dropdownTo =
    mobile && items.length > 1
      ? undefined
      : getVersionTargetDoc(displayedVersionItem.version, activeDocContext).path;

  return (
    <DropdownNavbarItem
      {...props}
      mobile={mobile}
      label={dropdownLabel}
      to={dropdownTo}
      items={items}
      isActive={resolvedIsActive}
      className="navbar-version-badge-dropdown"
    />
  );
}

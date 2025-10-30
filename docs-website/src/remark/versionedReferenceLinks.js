/**
 * Remark plugin to automatically add version context to reference links
 *
 * Transforms: /reference/slug -> /reference/{version}/slug
 * Where {version} is automatically determined from the current doc's version context
 *
 * Note: The latest version does NOT get a version prefix in the URL
 */

const path = require('path');
const fs = require('fs');

// Cache the latest version at module level, but allow it to be read lazily
let latestVersion = null;

// Dynamically read the latest version from versions.json
function getLatestVersion() {
  if (latestVersion !== null) {
    return latestVersion;
  }

  try {
    const versionsPath = path.join(__dirname, '../../versions.json');
    const versions = JSON.parse(fs.readFileSync(versionsPath, 'utf8'));
    // The first version in versions.json is the latest version
    latestVersion = versions[0];
    return latestVersion;
  } catch (error) {
    console.warn('Could not read versions.json, defaulting latest version detection', error);
    return null;
  }
}

function versionedReferenceLinks() {
  return (tree, file) => {
    // Read the latest version inside the processor function
    const currentLatestVersion = getLatestVersion();

    // Try to get version from file metadata (set by Docusaurus)
    let version = file.data?.version;

    // If not in metadata, extract from file path
    if (!version) {
      // file.history[0] contains the file path
      const filePath = file.history?.[0] || '';

      // Debug logging (will appear in build logs)
      console.log('[versionedReferenceLinks] Processing file:', filePath);
      console.log('[versionedReferenceLinks] file.data:', file.data);

      // Match patterns like: versioned_docs/version-2.19/ or reference_versioned_docs/version-2.19/
      // Handle both relative and absolute paths
      const versionMatch = filePath.match(/versioned_docs[/\\]version-([^/\\]+)[/\\]/);

      if (versionMatch) {
        version = versionMatch[1]; // e.g., "2.19"
        console.log('[versionedReferenceLinks] Detected version from path:', version);
      } else {
        // Check if it's in the current/next docs (no version folder)
        version = 'current';
        console.log('[versionedReferenceLinks] No version in path, using current');
      }
    } else {
      console.log('[versionedReferenceLinks] Version from metadata:', version);
    }

    console.log('[versionedReferenceLinks] Final version:', version, 'Latest version:', currentLatestVersion);

    // Manually visit all nodes in the tree
    const visit = (node, callback) => {
      callback(node);
      if (node.children) {
        node.children.forEach(child => visit(child, callback));
      }
    };

    visit(tree, (node) => {
      // Check if this is a link node
      if (node.type === 'link' || node.type === 'linkReference') {
        if (node.url && node.url.startsWith('/reference/')) {
          // Check if it already has a version
          const hasVersion = /^\/reference\/(next|v?\d+\.\d+)\//.test(node.url);

          if (!hasVersion) {
            // Latest version: no version prefix needed (served at /reference/)
            // Current/next version: add /next/
            // Other versions: add version number
            if (version === currentLatestVersion) {
              // Latest version - no changes needed, keep as /reference/...
              return;
            } else if (version === 'current') {
              // Current version - add /next/
              node.url = node.url.replace('/reference/', '/reference/next/');
            } else {
              // Older version - add version number
              node.url = node.url.replace('/reference/', `/reference/${version}/`);
            }
          }
        }
      }
    });
  };
}

module.exports = versionedReferenceLinks;

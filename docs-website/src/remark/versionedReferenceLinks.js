/**
 * Remark plugin to automatically add version context to reference links
 * 
 * Transforms: /reference/slug -> /reference/{version}/slug
 * Where {version} is automatically determined from the current doc's version context
 */

function versionedReferenceLinks() {
  return (tree, file) => {
    // Get version from file metadata (set by Docusaurus)
    const version = file.data?.version || 'next';
    
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
            // Add version after /reference/
            node.url = node.url.replace('/reference/', `/reference/${version}/`);
          }
        }
      }
    });
  };
}

module.exports = versionedReferenceLinks;

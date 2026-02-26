// Imports a Docusaurus sidebar JS file and prints its default export as JSON.
// Usage: node extract_sidebar.mjs <path-to-sidebars.js>

import { pathToFileURL } from "url";
import { resolve } from "path";

const [, , sidebarPath] = process.argv;

if (!sidebarPath) {
  process.stderr.write("Usage: node extract_sidebar.mjs <path-to-sidebars.js>\n");
  process.exit(1);
}

const mod = await import(pathToFileURL(resolve(sidebarPath)).href);
process.stdout.write(JSON.stringify(mod.default));

// Imports a Docusaurus sidebar JS file, renders it as a JSON and writes it to a file.
// Usage: node extract_sidebar.mjs <path-to-sidebars.js> <output-path>

import { pathToFileURL } from "url";
import { resolve } from "path";
import { writeFileSync } from "fs";

const [, , sidebarPath, outputPath] = process.argv;

if (!sidebarPath || !outputPath) {
  process.stderr.write("Usage: node extract_sidebar.mjs <path-to-sidebars.js> <output-path>\n");
  process.exit(1);
}

const mod = await import(pathToFileURL(resolve(sidebarPath)).href);
writeFileSync(resolve(outputPath), JSON.stringify(mod.default, null, 2));

/** Package the public game as one HTML file for offline play or a sandboxed iframe.
 * Uses the same simulation, rendering and input code as the local model client.
 * Browser edition makes no model-service requests and has no embedded credentials.
 */
import { build } from 'vite';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const destination = resolve(process.argv[2] || resolve(root, 'game/dist/cinder-station.html'));
const bundles = await build({
  configFile: false,
  root: resolve(root, 'game'),
  logLevel: 'warn',
  build: {
    write: false,
    target: 'es2022',
    minify: 'esbuild',
    sourcemap: false,
    cssCodeSplit: false,
    lib: { entry: resolve(root, 'game/src/main.ts'), name: 'CinderStation', formats: ['iife'] },
  },
});
const output = (Array.isArray(bundles) ? bundles : [bundles]).flatMap(bundle => bundle.output);
const javascript = output.filter(file => file.type === 'chunk').map(file => file.code).join('\n');
const css = output.filter(file => file.type === 'asset' && file.fileName.endsWith('.css'))
  .map(file => String(file.source)).join('\n');
if (!javascript || !css) throw new Error('Expected both bundled game JavaScript and CSS.');
const template = await readFile(resolve(root, 'game/index.html'), 'utf8');
const entry = '<script type="module" src="/src/main.ts"></script>';
if (!template.includes(entry)) throw new Error('Game entry point was not found in the HTML template.');
const html = template.replace('<html lang="en">', '<html lang="en" data-static-demo="true">')
  .replace('</head>', `<style>${css.replace(/<\/style/gi, '<\\/style')}</style></head>`)
  .replace(entry, () => `<script>${javascript.replace(/<\/script/gi, '<\\/script')}</script>`);
await mkdir(dirname(destination), { recursive: true });
await writeFile(destination, html, 'utf8');
console.log(`Built standalone Cinder Station (${Buffer.byteLength(html).toLocaleString()} bytes).`);

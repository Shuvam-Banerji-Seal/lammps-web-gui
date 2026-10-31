#!/usr/bin/env node
/**
 * Builds the MCP folder of the site into dist/ (run by `npm run build` after `vite build`):
 *
 *   dist/mcp/lammps-mcp.mjs   the MCP stdio server (mcp/main.ts) bundled with its knowledge; plain Node, no packages
 *   dist/mcp/lammps-mcp.tgz   the same file as an npm package, so `npx -y <url>` runs it
 *   dist/mcp/knowledge.json   the knowledge (mcp/knowledge.ts) for agents that fetch URLs
 *   dist/mcp/index.json       manifest: tools, install commands, links
 *   dist/mcp/index.html       the page for people
 *   dist/llms.txt             a short map of the site
 *
 * The TypeScript sources are loaded through Vite (ssrLoadModule) and bundled with Vite's SSR build, so this needs no
 * other tools than the repository's own.
 */
import { execFileSync } from 'node:child_process';
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync, chmodSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build, createServer } from 'vite';

const root = fileURLToPath(new URL('..', import.meta.url));
const dist = join(root, 'dist');
const out = join(dist, 'mcp');
mkdirSync(out, { recursive: true });

// 1. the knowledge, the tool list and the static texts, from the TypeScript sources
const vite = await createServer({
  configFile: false, root, logLevel: 'error', appType: 'custom',
  server: { middlewareMode: true, watch: null, hmr: false },
  optimizeDeps: { noDiscovery: true, include: [] },
});
const { readSources } = await vite.ssrLoadModule('/mcp/sources.ts');
const { buildKnowledge } = await vite.ssrLoadModule('/mcp/knowledge.ts');
const { createMcpServer } = await vite.ssrLoadModule('/mcp/server.ts');
const site = await vite.ssrLoadModule('/mcp/site.ts');
const knowledge = buildKnowledge(readSources(root));
const listed = await createMcpServer(knowledge).handle({ jsonrpc: '2.0', id: 1, method: 'tools/list' });
const tools = listed.result.tools;
await vite.close();

writeFileSync(join(out, site.MCP_FILES.knowledge), JSON.stringify(knowledge));
writeFileSync(join(out, site.MCP_FILES.manifest), `${JSON.stringify(site.manifest(knowledge, tools), null, 2)}\n`);
writeFileSync(join(out, 'index.html'), site.pageHtml(knowledge, tools));
writeFileSync(join(dist, 'llms.txt'), site.llmsTxt(knowledge, tools));

// 2. the server: mcp/main.ts with the knowledge built in, one ES module for Node
const VIRTUAL = 'virtual:lammps-mcp-knowledge';
const bundleDir = mkdtempSync(join(tmpdir(), 'lammps-mcp-'));
await build({
  configFile: false, root, logLevel: 'warn',
  plugins: [{
    name: 'lammps-mcp-knowledge',
    resolveId: (id) => (id === VIRTUAL ? `\0${VIRTUAL}` : null),
    load: (id) => (id === `\0${VIRTUAL}` ? `export default ${JSON.stringify(knowledge)};` : null),
  }],
  ssr: { noExternal: true, target: 'node' },
  build: {
    ssr: 'mcp/main.ts', outDir: bundleDir, emptyOutDir: true, target: 'node18', minify: false, copyPublicDir: false,
    rollupOptions: { output: { format: 'es', entryFileNames: 'lammps-mcp.mjs', codeSplitting: false } },
  },
});
const header = `#!/usr/bin/env node
// LAMMPS Web GUI MCP server ${knowledge.version} — ${site.mcpUrls(knowledge).page}
// Model Context Protocol over stdio: add "node lammps-mcp.mjs" (or "npx -y ${site.mcpUrls(knowledge).tarball}") to an MCP client.
// Built from https://github.com/Shuvam-Banerji-Seal/lammps-web-gui (mcp/); licence: LicenseRef-LAMMPS-Web-GUI-Educational-1.0
// (the repository's LICENSE). The LAMMPS documentation index inside (command names, page titles and first sentences)
// is from LAMMPS doc/src at commit ${knowledge.docIndex.commit} (GPL-2.0, https://github.com/lammps/lammps).
`;
const body = readFileSync(join(bundleDir, 'lammps-mcp.mjs'), 'utf8').replace(/^#!.*\n/, '');
const serverFile = join(out, site.MCP_FILES.server);
writeFileSync(serverFile, header + body);
chmodSync(serverFile, 0o755);
rmSync(bundleDir, { recursive: true, force: true });

// 3. the npm tarball for npx: the server as the package's only bin
const pkgDir = mkdtempSync(join(tmpdir(), 'lammps-mcp-pkg-'));
const pkgName = 'lammps-web-gui-mcp';
writeFileSync(join(pkgDir, 'package.json'), `${JSON.stringify({
  name: pkgName,
  version: knowledge.version,
  description: 'LAMMPS Web GUI MCP server: LAMMPS documentation, script checking, browser-engine support, templates and examples for coding agents (stdio)',
  type: 'module',
  bin: { [pkgName]: site.MCP_FILES.server },
  files: [site.MCP_FILES.server, 'LICENSE', 'LAMMPS-GPL-2.0.txt', 'README.md'],
  engines: { node: '>=18' },
  license: 'SEE LICENSE IN LICENSE',
  homepage: site.mcpUrls(knowledge).page,
  repository: { type: 'git', url: 'git+https://github.com/Shuvam-Banerji-Seal/lammps-web-gui.git' },
}, null, 2)}\n`);
copyFileSync(serverFile, join(pkgDir, site.MCP_FILES.server));
copyFileSync(join(root, 'LICENSE'), join(pkgDir, 'LICENSE'));
copyFileSync(join(root, 'third_party', 'lammps', 'LICENSE'), join(pkgDir, 'LAMMPS-GPL-2.0.txt'));
writeFileSync(join(pkgDir, 'README.md'), `# ${pkgName}\n\nThe LAMMPS Web GUI MCP server (stdio). See ${site.mcpUrls(knowledge).page}.\n\nThe LAMMPS documentation index it contains is from LAMMPS (GPL-2.0, LAMMPS-GPL-2.0.txt); the rest is covered by LICENSE.\n`);
execFileSync('npm', ['pack', '--silent', '--pack-destination', pkgDir], { cwd: pkgDir, stdio: ['ignore', 'pipe', 'inherit'] });
const tgz = readdirSync(pkgDir).find((f) => f.endsWith('.tgz'));
if (!tgz) throw new Error('npm pack wrote no tarball');
copyFileSync(join(pkgDir, tgz), join(out, site.MCP_FILES.tarball));
rmSync(pkgDir, { recursive: true, force: true });

// 4. smoke test: the published file answers initialize, tools/list and a tool call over stdio, or the build fails
const replies = execFileSync(process.execPath, [serverFile], {
  input: [
    { jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'build', version: '0' } } },
    { jsonrpc: '2.0', method: 'notifications/initialized' },
    { jsonrpc: '2.0', id: 2, method: 'tools/list' },
    { jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'validate_lammps_script', arguments: { script: 'units lj\nrun 0\n' } } },
  ].map((m) => JSON.stringify(m)).join('\n') + '\n',
  encoding: 'utf8', stdio: ['pipe', 'pipe', 'ignore'], timeout: 30000,
}).trim().split('\n').map((l) => JSON.parse(l));
const byId = new Map(replies.map((r) => [r.id, r]));
if (byId.get(1)?.result?.serverInfo?.name !== knowledge.name || byId.get(2)?.result?.tools?.length !== tools.length
  || !/^Script check:/.test(byId.get(3)?.result?.content?.[0]?.text ?? '')) {
  throw new Error(`the bundled MCP server did not answer as expected: ${JSON.stringify(replies).slice(0, 500)}`);
}

const kb = (f) => `${(readFileSync(join(out, f)).length / 1024).toFixed(0)} KB`;
console.log(`mcp: ${tools.length} tools; ${site.MCP_FILES.server} ${kb(site.MCP_FILES.server)}, ${site.MCP_FILES.tarball} ${kb(site.MCP_FILES.tarball)}, ${site.MCP_FILES.knowledge} ${kb(site.MCP_FILES.knowledge)} -> dist/mcp/`);

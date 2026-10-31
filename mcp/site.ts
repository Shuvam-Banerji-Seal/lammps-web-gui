/**
 * The static side of the MCP folder on the site (written by mcp/build.mjs into dist/): the /mcp/ page for people,
 * /mcp/index.json (a manifest an agent can read with a plain GET), and /llms.txt.
 *
 * GitHub Pages serves files only, and every MCP network transport needs a server that answers POST requests, so the
 * site cannot be an MCP endpoint itself. It publishes the server instead (lammps-mcp.mjs, and the same file as an npm
 * tarball for npx); an MCP client runs it on the user's machine over stdio.
 */
import type { Knowledge } from './knowledge';

export interface ToolInfo { name: string; title?: string; description: string }

export const MCP_FILES = {
  server: 'lammps-mcp.mjs',
  tarball: 'lammps-mcp.tgz',
  knowledge: 'knowledge.json',
  manifest: 'index.json',
} as const;

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

export const mcpUrls = (k: Knowledge) => {
  const base = `${k.site}mcp/`;
  return {
    page: base,
    server: `${base}${MCP_FILES.server}`,
    tarball: `${base}${MCP_FILES.tarball}`,
    knowledge: `${base}${MCP_FILES.knowledge}`,
    manifest: `${base}${MCP_FILES.manifest}`,
  };
};

/**
 * The npx arguments. npm 12 refuses to install a package from a tarball URL unless --allow-remote=all is given
 * (its allow-remote setting defaults to "none"); npm 10 and 11 accept the flag and ignore it.
 */
export const npxArgs = (k: Knowledge): string[] => ['-y', '--allow-remote=all', mcpUrls(k).tarball];

/** How to connect, per client: the commands and configuration files shown on the page and in the manifest. */
export const connectRecipes = (k: Knowledge) => {
  const u = mcpUrls(k);
  const args = npxArgs(k);
  const cli = `npx ${args.join(' ')}`;
  return {
    claudeCode: `claude mcp add lammps -- ${cli}`,
    codex: `codex mcp add lammps -- ${cli}`,
    // Claude Desktop, Cursor (.cursor/mcp.json), Windsurf, Cline, Gemini CLI (settings.json)
    json: JSON.stringify({ mcpServers: { lammps: { command: 'npx', args } } }, null, 2),
    vscode: JSON.stringify({ servers: { lammps: { type: 'stdio', command: 'npx', args } } }, null, 2),
    opencode: JSON.stringify({ mcp: { lammps: { type: 'local', command: ['npx', ...args], enabled: true } } }, null, 2),
    download: `curl -fsSLo lammps-mcp.mjs ${u.server}\nnode lammps-mcp.mjs   # an MCP stdio server: give your client "node /path/to/lammps-mcp.mjs"`,
  };
};

export const manifest = (k: Knowledge, tools: ToolInfo[]) => {
  const u = mcpUrls(k);
  const r = connectRecipes(k);
  return {
    name: k.name,
    title: 'LAMMPS Web GUI MCP server',
    version: k.version,
    description: 'Model Context Protocol server for writing LAMMPS input: the LAMMPS documentation, a script checker, the browser engine\'s supported commands and styles, starter scripts, example notebooks and CMake build scripts.',
    transport: 'stdio',
    why_not_http: 'The site is static (GitHub Pages): it cannot answer the POST requests of the MCP HTTP transport, so the server runs locally over stdio.',
    install: { npx: { command: 'npx', args: npxArgs(k) }, node: { download: u.server, command: 'node', args: ['lammps-mcp.mjs'] } },
    clients: {
      claude_code: r.claudeCode, codex: r.codex, mcp_servers_json: JSON.parse(r.json), vscode_mcp_json: JSON.parse(r.vscode),
      opencode_json: JSON.parse(r.opencode),
    },
    without_mcp: `Agents without MCP can read ${u.manifest}, ${u.knowledge} and ${k.site}llms.txt with plain GET requests.`,
    tools: tools.map((t) => ({ name: t.name, title: t.title, description: t.description })),
    static: {
      knowledge: u.knowledge,
      llms_txt: `${k.site}llms.txt`,
      examples: `${k.site}notebooks/index.json`,
      lammps_docs: k.docIndex.docs,
      lammps_doc_source: k.docIndex.raw,
      lammps_commit: k.docIndex.commit,
    },
    site: k.site,
  };
};

export const llmsTxt = (k: Knowledge, tools: ToolInfo[]): string => {
  const u = mcpUrls(k);
  return `# LAMMPS Web GUI

> A browser-only workbench for the LAMMPS molecular dynamics code: a script builder, a script checker, a CMake build
> helper, a WebGL structure and trajectory viewer, and a notebook that runs LAMMPS input in the browser with an engine
> checked against native LAMMPS (commit ${k.docIndex.commit.slice(0, 7)}).

## MCP server

- [MCP page](${u.page}): how to connect a coding agent (stdio server; \`npx ${npxArgs(k).join(' ')}\`)
- [Manifest](${u.manifest}): tools, install commands and static data as JSON
- [Knowledge](${u.knowledge}): everything the server knows, as one JSON file (doc index, browser engine support, templates, examples, guides)

Tools: ${tools.map((t) => t.name).join(', ')}.

## Docs

- [LAMMPS documentation](${k.docIndex.docs}) (the server fetches pages from ${k.docIndex.raw})
${k.guides.map((g) => `- ${g.title}: ${g.description}`).join('\n')}

## Examples

${k.examples.map((e) => `- [${e.name}](${k.site}notebooks/${e.file}): ${e.summary}`).join('\n')}
`;
};

export const pageHtml = (k: Knowledge, tools: ToolInfo[]): string => {
  const u = mcpUrls(k);
  const r = connectRecipes(k);
  const code = (s: string) => `<pre><code>${esc(s)}</code></pre>`;
  const engine = k.engine.styles;
  const count = (key: string) => (engine[key] ?? []).length;
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LAMMPS MCP server</title>
<meta name="description" content="Connect a coding agent to the LAMMPS Web GUI: LAMMPS documentation, script checking, browser-engine support, templates and examples over the Model Context Protocol.">
<link rel="icon" href="../logo.svg">
<style>
:root { --bg: #f7f8fa; --fg: #1b1f24; --muted: #57606a; --card: #ffffff; --line: #d8dee4; --accent: #0b6bcb; --code: #eef1f4; }
@media (prefers-color-scheme: dark) { :root { --bg: #0f1216; --fg: #e6e9ed; --muted: #9aa4af; --card: #171b21; --line: #2a313a; --accent: #5aa9ff; --code: #1e242c; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 32px 0 8px; }
p, li { color: var(--fg); } .muted { color: var(--muted); }
a { color: var(--accent); }
.codewrap { position: relative; }
pre { background: var(--code); border: 1px solid var(--line); border-radius: 6px; padding: 10px 64px 10px 12px; overflow-x: auto; font-size: 13px; }
.copy { position: absolute; top: 6px; right: 6px; font: 12px system-ui, sans-serif; padding: 3px 8px; border: 1px solid var(--line); border-radius: 4px; background: var(--card); color: var(--fg); cursor: pointer; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; vertical-align: top; padding: 8px; border-top: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; }
td:first-child { white-space: nowrap; font-family: ui-monospace, monospace; font-size: 13px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; margin: 12px 0; }
@media (max-width: 600px) { td:first-child { white-space: normal; word-break: break-word; } }
</style>
</head>
<body>
<main>
<p class="muted"><a href="../">LAMMPS Web GUI</a> / MCP</p>
<h1>LAMMPS MCP server</h1>
<p>Connect any coding agent that speaks the <a href="https://modelcontextprotocol.io">Model Context Protocol</a> to what this site knows about writing LAMMPS input. That includes Claude Code, Codex, Cursor, VS Code, Windsurf, Cline, Gemini CLI and opencode. Agents without MCP can read the same data as <a href="#files">plain files</a>. The server provides:</p>
<ul>
<li><strong>The LAMMPS documentation:</strong> ${k.docIndex.entries.length} commands and styles, with each page's full text at commit ${esc(k.docIndex.commit.slice(0, 7))}.</li>
<li><strong>The script checker.</strong></li>
<li><strong>What the in-browser engine runs:</strong> ${count('pair_style')} pair styles, ${count('fix')} fixes and ${count('compute')} computes.</li>
<li><strong>Starter scripts, example notebooks and CMake build scripts.</strong></li>
</ul>

<h2>Connect</h2>
<div class="card">
<p><strong>Claude Code</strong></p>${code(r.claudeCode)}
<p><strong>Codex CLI</strong></p>${code(r.codex)}
<p><strong>Claude Desktop, Cursor, Windsurf, Cline, Gemini CLI</strong>: add this to the client's <code>mcpServers</code> configuration.</p>${code(r.json)}
<p><strong>VS Code</strong> (<code>.vscode/mcp.json</code>)</p>${code(r.vscode)}
<p><strong>opencode</strong> (<code>opencode.json</code>)</p>${code(r.opencode)}
<p><strong>Without npx</strong>: download one file and run it with Node 18 or later. It has no dependencies.</p>${code(r.download)}
</div>
<p class="muted">npm 12 installs a package from a URL only with <code>--allow-remote=all</code>. Older npm versions accept the flag and ignore it.</p>
<p class="muted">Why a local server? The site is hosted as static files, and the MCP HTTP transport needs a server that answers POST requests. The server is therefore published here as one JavaScript file. Your MCP client runs it on your machine over stdio. Only <code>get_lammps_doc</code> uses the network, to fetch LAMMPS documentation pages from GitHub.</p>

<h2>Tools</h2>
<table>
<thead><tr><th>Tool</th><th>What it does</th></tr></thead>
<tbody>
${tools.map((t) => `<tr><td>${esc(t.name)}</td><td>${esc(t.description)}</td></tr>`).join('\n')}
</tbody>
</table>
<p>Resources include the guides (${k.guides.map((g) => esc(g.id)).join(', ')}), the templates, the example notebooks and the engine support list. Any documentation page is available as <code>lammps-doc://{name}</code>. The prompt <code>write_lammps_input</code> walks an agent through writing and checking a script.</p>

<h2 id="files">Plain files</h2>
<p>Agents that only fetch URLs can read the same data directly:</p>
<ul>
<li><a href="${esc(u.manifest)}">${esc(MCP_FILES.manifest)}</a>: tools, install commands and links</li>
<li><a href="${esc(u.knowledge)}">${esc(MCP_FILES.knowledge)}</a>: the doc index, engine support, templates, examples and guides</li>
<li><a href="../llms.txt">llms.txt</a>: a short map of the site</li>
</ul>
<p class="muted">Version ${esc(k.version)}. The documentation index and the fetched pages are from <a href="https://github.com/lammps/lammps">LAMMPS</a> (GPL-2.0). The server and the rest of this site are covered by the project's licence.</p>
</main>
<script>
// a Copy button on each code block
for (const pre of document.querySelectorAll('pre')) {
  const b = document.createElement('button');
  b.className = 'copy'; b.type = 'button'; b.textContent = 'Copy';
  b.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(pre.querySelector('code').textContent); b.textContent = 'Copied'; }
    catch { b.textContent = 'Select and copy'; }
    setTimeout(() => { b.textContent = 'Copy'; }, 1500);
  });
  // the button sits on a wrapper so it stays put when a long line scrolls
  const wrap = document.createElement('div');
  wrap.className = 'codewrap';
  pre.replaceWith(wrap);
  wrap.append(pre, b);
}
</script>
</body>
</html>
`;
};

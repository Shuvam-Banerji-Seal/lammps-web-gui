/**
 * The MCP server of the site (mcp/server.ts, published as /mcp/lammps-mcp.mjs by mcp/build.mjs): JSON-RPC handling,
 * every tool, the resources and the prompt, against the knowledge built from the repository (mcp/knowledge.ts), with
 * the LAMMPS documentation fetch replaced by a fake. Also the static files (mcp/site.ts) and the doc index.
 */
import { describe, expect, it } from 'vitest';
import { join } from 'node:path';
import { buildKnowledge, type Knowledge } from '../mcp/knowledge';
import { readSources } from '../mcp/sources';
import { createMcpServer, PROTOCOL_VERSIONS } from '../mcp/server';
import { llmsTxt, manifest, pageHtml } from '../mcp/site';

const k: Knowledge = buildKnowledge(readSources(join(__dirname, '..')));

const fetched: string[] = [];
const fakeFetch = async (url: string) => {
  fetched.push(url);
  if (url.endsWith('/missing.rst')) return { ok: false, status: 404, text: async () => '' };
  const page = url.slice(url.lastIndexOf('/') + 1);
  return { ok: true, status: 200, text: async () => `${page}\n${'x'.repeat(5000)}END` };
};
const server = createMcpServer(k, { fetch: fakeFetch });

let nextId = 1;
const rpc = async (method: string, params?: Record<string, unknown>) => {
  const r = await server.handle({ jsonrpc: '2.0', id: nextId++, method, ...(params ? { params } : {}) });
  expect(r).not.toBeNull();
  return r!;
};
const tool = async (name: string, args: Record<string, unknown> = {}) => {
  const r = await rpc('tools/call', { name, arguments: args });
  expect(r.error).toBeUndefined();
  return r.result as { content: { type: string; text: string }[]; structuredContent?: Record<string, unknown>; isError?: boolean };
};

describe('MCP protocol', () => {
  it('initialize answers the requested revision when it knows it, else its newest', async () => {
    const a = (await rpc('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '0' } })).result as Record<string, unknown>;
    expect(a.protocolVersion).toBe('2025-06-18');
    expect(a.serverInfo).toMatchObject({ name: 'lammps-web-gui', version: k.version });
    expect(a.capabilities).toMatchObject({ tools: {}, resources: {}, prompts: {} });
    const b = (await rpc('initialize', { protocolVersion: '1999-01-01' })).result as Record<string, unknown>;
    expect(b.protocolVersion).toBe(PROTOCOL_VERSIONS[0]);
  });

  it('notifications get no reply; unknown methods and bad messages get JSON-RPC errors', async () => {
    expect(await server.handle({ jsonrpc: '2.0', method: 'notifications/initialized' })).toBeNull();
    expect((await rpc('no/such/method')).error).toMatchObject({ code: -32601 });
    expect((await server.handle({ id: 3, method: 'ping' }))!.error).toMatchObject({ code: -32600 });
    expect((await rpc('ping')).result).toEqual({});
    expect((await rpc('tools/call', { name: 'no_such_tool' })).error).toMatchObject({ code: -32602 });
  });

  it('lists ten tools, each with an object input schema and a description', async () => {
    const tools = ((await rpc('tools/list')).result as { tools: { name: string; description: string; inputSchema: { type: string } }[] }).tools;
    expect(tools.map((t) => t.name)).toEqual([
      'search_lammps_docs', 'get_lammps_doc', 'validate_lammps_script', 'check_browser_support', 'list_templates', 'get_template',
      'list_examples', 'get_example', 'generate_cmake_build', 'read_guide',
    ]);
    for (const t of tools) {
      expect(t.inputSchema.type).toBe('object');
      expect(t.description.length).toBeGreaterThan(40);
    }
  });
});

describe('LAMMPS documentation', () => {
  it('the index maps every name to a page with a title, from the engine commit', () => {
    expect(k.docIndex.commit).toMatch(/^b332f28/);
    expect(k.docIndex.entries.length).toBeGreaterThan(1000);
    for (const e of k.docIndex.entries) expect(k.docIndex.pages[e.page]?.title, e.name).toBeTruthy();
    const page = (n: string) => k.docIndex.entries.find((e) => e.name === n)?.page;
    expect(page('fix nvt')).toBe('fix_nh');
    expect(page('pair_style lj/cut')).toBe('pair_lj');
    expect(page('read_data')).toBe('read_data');
  });

  it('search finds styles by name and by topic, with browser support', async () => {
    const rdf = (await tool('search_lammps_docs', { query: 'rdf', limit: 3 })).structuredContent!.results as { name: string; browser: string }[];
    expect(rdf[0]).toMatchObject({ name: 'compute rdf', browser: 'supported' });
    const thermo = (await tool('search_lammps_docs', { query: 'langevin thermostat' })).structuredContent!.results as { name: string }[];
    expect(thermo.map((h) => h.name)).toContain('fix langevin');
    const none = await tool('search_lammps_docs', { query: 'zzzqqq' });
    expect(none.content[0].text).toMatch(/No LAMMPS command or style matches/);
  });

  it('get_lammps_doc resolves script names, aliases and pages, fetches the page at the commit and pages long text', async () => {
    fetched.length = 0;
    const r = await tool('get_lammps_doc', { name: 'fix nvt', max_chars: 1000 });
    expect(fetched).toEqual([`https://raw.githubusercontent.com/lammps/lammps/${k.docIndex.commit}/doc/src/fix_nh.rst`]);
    expect(r.content[0].text).toContain('https://docs.lammps.org/fix_nh.html');
    expect(r.content[0].text).toContain('Browser engine support for fix nvt: supported.');
    expect(r.content[0].text).toMatch(/call again with offset=1000/);
    expect(r.structuredContent).toMatchObject({ page: 'fix_nh', offset: 0 });
    // the cached page, read on from the offset
    const end = await tool('get_lammps_doc', { name: 'fix_nh', offset: 5000 });
    expect(end.content[0].text).toMatch(/END$/);
    expect(fetched).toHaveLength(1);
    // "pair lj/cut" and an accelerator suffix name the same page as pair_style lj/cut
    expect((await tool('get_lammps_doc', { name: 'pair lj/cut/omp' })).structuredContent).toMatchObject({ page: 'pair_lj', name: 'pair_style lj/cut' });
  });

  it('get_lammps_doc reports an unknown name and a failed fetch as tool errors', async () => {
    const unknown = await tool('get_lammps_doc', { name: 'fix nosuchstyle' });
    expect(unknown.isError).toBe(true);
    expect(unknown.content[0].text).toMatch(/search_lammps_docs/);
    const failing = createMcpServer(k, { fetch: async () => { throw new Error('offline'); } });
    const r = (await failing.handle({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'get_lammps_doc', arguments: { name: 'read_data' } } }))!;
    expect(r.result).toMatchObject({ isError: true });
    expect(JSON.stringify(r.result)).toContain('https://docs.lammps.org/read_data.html');
  });
});

describe('script tools', () => {
  it('validate_lammps_script reports rules with their docs page', async () => {
    const r = await tool('validate_lammps_script', { script: 'region box block 0 5 0 5 0 5\ncreate_box 1 box\nunits lj\n' });
    const d = r.structuredContent!.diagnostics as { rule: string; level: string; doc?: string }[];
    expect(d.find((x) => x.rule === 'order/units-after-box')).toMatchObject({ level: 'error' });
    expect(r.content[0].text).toMatch(/^Script check: \d+ errors/);
  });

  it('every template is a complete script the checker passes', async () => {
    expect(k.templates.length).toBeGreaterThan(0);
    for (const t of k.templates) {
      const s = (await tool('get_template', { id: t.id })).structuredContent!.script as string;
      expect(s).toMatch(/\brun\b|\bminimize\b/);
      const counts = (await tool('validate_lammps_script', { script: s })).structuredContent!.counts as { errors: number };
      expect(counts.errors, t.id).toBe(0);
    }
    expect((await tool('get_template', { id: 'nope' })).isError).toBe(true);
  });

  it('check_browser_support names what the engine does not run, through hybrid styles and accelerator suffixes', async () => {
    const ok = await tool('check_browser_support', { script: k.templates[0].script });
    expect(ok.structuredContent).toMatchObject({ supported: true });
    const r = await tool('check_browser_support', {
      script: 'units metal\npair_style hybrid/overlay lj/cut/omp 2.5 hdnnp 6.0 dir model\nfix 1 all nve\nfix 2 all plumed plumedfile p.dat\n',
    });
    expect(r.structuredContent).toMatchObject({ supported: false });
    const items = (r.structuredContent!.problems as { item: string }[]).map((p) => p.item);
    expect(items).toEqual(['pair_style hdnnp', 'fix plumed']);
  });

  it('examples come with their script and input files', async () => {
    const list = (await tool('list_examples')).structuredContent!.examples as { file: string; files: { name: string }[] }[];
    expect(list.length).toBe(k.examples.length);
    const water = list.find((e) => e.file === 'water-spce-data.lmpnb')!;
    expect(water.files.map((f) => f.name)).toContain('water_spce.data');
    const s = (await tool('get_example', { file: 'water-spce-data', format: 'script' })).structuredContent!.script as string;
    expect(s).toContain('read_data');
  });

  it('generate_cmake_build writes the package flags and refuses unknown names', async () => {
    const r = await tool('generate_cmake_build', { packages: ['molecule', 'KSPACE'], accelerator: 'openmp', os: 'linux' });
    expect(r.structuredContent!.flags).toEqual(expect.arrayContaining(['-D PKG_MOLECULE=yes', '-D PKG_KSPACE=yes']));
    expect(r.content[0].text).toContain('cmake');
    const bad = await tool('generate_cmake_build', { packages: ['NOPE'] });
    expect(bad.isError).toBe(true);
  });

  it('read_guide returns each guide', async () => {
    for (const g of k.guides) expect((await tool('read_guide', { id: g.id })).content[0].text).toBe(g.text);
    expect((await tool('read_guide', { id: 'writing-inputs' })).content[0].text).toContain('https://docs.lammps.org/Commands_structure.html');
  });
});

describe('resources and prompts', () => {
  it('lists and reads guides, templates, examples, engine support and doc pages', async () => {
    const res = ((await rpc('resources/list')).result as { resources: { uri: string }[] }).resources.map((r) => r.uri);
    expect(res).toContain('lammps-web-gui://guides/writing-inputs');
    expect(res).toContain(`lammps-web-gui://templates/${k.templates[0].id}`);
    const read = async (uri: string) => ((await rpc('resources/read', { uri })).result as { contents: { text: string; mimeType: string }[] }).contents[0];
    expect((await read('lammps-web-gui://engine/support')).mimeType).toBe('application/json');
    expect(JSON.parse((await read('lammps-web-gui://engine/support')).text).styles.pair_style).toContain('lj/cut');
    expect((await read('lammps-doc://fix%20nvt')).text.startsWith('fix_nh.rst')).toBe(true);
    expect((await rpc('resources/read', { uri: 'lammps-web-gui://nothing/here' })).error).toMatchObject({ code: -32002 });
    const templates = ((await rpc('resources/templates/list')).result as { resourceTemplates: { uriTemplate: string }[] }).resourceTemplates;
    expect(templates[0].uriTemplate).toBe('lammps-doc://{name}');
  });

  it('the write_lammps_input prompt names the workflow', async () => {
    const p = (await rpc('prompts/get', { name: 'write_lammps_input', arguments: { goal: 'liquid argon', browser: 'yes' } })).result as { messages: { content: { text: string } }[] };
    const t = p.messages[0].content.text;
    for (const step of ['search_lammps_docs', 'get_lammps_doc', 'validate_lammps_script', 'check_browser_support']) expect(t).toContain(step);
    expect((await rpc('prompts/get', { name: 'write_lammps_input', arguments: {} })).error).toMatchObject({ code: -32602 });
  });
});

describe('static files of /mcp/', () => {
  it('the page, the manifest and llms.txt name every tool and the npx command', async () => {
    const tools = ((await rpc('tools/list')).result as { tools: { name: string; description: string }[] }).tools;
    const html = pageHtml(k, tools);
    const m = manifest(k, tools);
    const llms = llmsTxt(k, tools);
    for (const t of tools) {
      expect(html).toContain(t.name);
      expect(llms).toContain(t.name);
    }
    expect(m.tools).toHaveLength(tools.length);
    // npm 12 installs from a tarball URL only with --allow-remote=all
    expect(m.install.npx.args).toEqual(['-y', '--allow-remote=all', 'https://shuvam-banerji-seal.github.io/lammps-web-gui/mcp/lammps-mcp.tgz']);
    expect(html).toContain('claude mcp add lammps -- npx -y --allow-remote=all https://shuvam-banerji-seal.github.io/lammps-web-gui/mcp/lammps-mcp.tgz');
    expect(m.clients.vscode_mcp_json).toEqual({ servers: { lammps: { type: 'stdio', command: 'npx', args: m.install.npx.args } } });
    expect(html).toMatch(/<title>LAMMPS MCP server<\/title>/);
  });
});

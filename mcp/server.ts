/**
 * The LAMMPS Web GUI MCP server: Model Context Protocol (https://modelcontextprotocol.io) tools, resources and a prompt
 * that give a coding agent what the site knows about writing LAMMPS input — the LAMMPS documentation (index, and every
 * page fetched from GitHub at the engine's parity commit), the script validator, which commands and styles the browser
 * engine runs, the starter templates and example notebooks, and CMake build scripts.
 *
 * Transport-free: handle() takes one JSON-RPC message and returns the reply (null for a notification). mcp/main.ts
 * runs it over stdio; the tests call it directly. The knowledge comes from mcp/knowledge.ts (types only here, so the
 * engine is not bundled).
 */
import type { Knowledge, DocIndex } from './knowledge';
import { validateScript, diagnosticCounts } from '../src/lammps/validate';
import { DEFAULT_COMPILER_OPTIONS, generateBuildScript, type CompilerOptions } from '../src/lammps/compiler';
import { scriptStatementsDetailed, tokenizeLine } from '../src/lammps/scriptParser';

/** Protocol revisions this server speaks, newest first. */
export const PROTOCOL_VERSIONS = ['2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05'];

type Json = null | boolean | number | string | Json[] | { [k: string]: Json };
interface Request { jsonrpc: '2.0'; id?: string | number | null; method: string; params?: Record<string, unknown> }
interface Reply { jsonrpc: '2.0'; id: string | number | null; result?: unknown; error?: { code: number; message: string; data?: unknown } }
interface ToolResult { content: { type: 'text'; text: string }[]; structuredContent?: Record<string, unknown>; isError?: boolean }

export interface ServerOptions {
  /** fetch for the LAMMPS documentation pages (globalThis.fetch by default). */
  fetch?: (url: string) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
}

class ToolError extends Error {}

const STYLE_COMMANDS: Record<string, string> = {
  pair: 'pair_style', bond: 'bond_style', angle: 'angle_style', dihedral: 'dihedral_style', improper: 'improper_style',
  kspace: 'kspace_style', atom: 'atom_style', min: 'min_style',
};
/** Accelerator variants are aliases of the plain style (the engine accepts them as such). */
const ACCEL = /\/(gpu|intel|kk|kk\/host|kk\/device|omp|opt)$/;
const HYBRID = new Set(['hybrid', 'hybrid/overlay', 'hybrid/scaled', 'hybrid/molecular']);

const text = (t: string, structured?: Record<string, unknown>): ToolResult =>
  ({ content: [{ type: 'text', text: t }], ...(structured ? { structuredContent: structured } : {}) });

const str = (args: Record<string, unknown>, key: string, required = true): string => {
  const v = args[key];
  if (typeof v === 'string' && v.trim()) return v;
  if (required) throw new ToolError(`missing string argument "${key}"`);
  return '';
};
const int = (args: Record<string, unknown>, key: string, dflt: number, lo: number, hi: number): number => {
  const v = args[key];
  if (v === undefined || v === null) return dflt;
  const n = typeof v === 'number' ? v : Number(v);
  if (!Number.isInteger(n)) throw new ToolError(`argument "${key}" must be an integer`);
  return Math.min(hi, Math.max(lo, n));
};

export const createMcpServer = (k: Knowledge, opts: ServerOptions = {}) => {
  const doFetch = opts.fetch ?? ((url: string) => globalThis.fetch(url));
  const idx: DocIndex = k.docIndex;
  const byName = new Map<string, string>();
  for (const e of idx.entries) if (!byName.has(e.name)) byName.set(e.name, e.page);
  const docUrl = (page: string) => `${idx.docs}${page}.html`;
  const notebookUrl = `${k.site}`;
  /** Script Builder labels and categories, as extra search words for the doc page they cite. */
  const builderWords = new Map<string, string>();
  for (const b of k.builder) {
    const m = /docs\.lammps\.org\/([^/#?]+)\.html/.exec(b.doc ?? '');
    if (m) builderWords.set(m[1], `${builderWords.get(m[1]) ?? ''} ${b.label} ${b.category}`.toLowerCase());
  }
  const engineCommands = new Set(k.engine.commands);
  const engineStyles = new Map(Object.entries(k.engine.styles).map(([cmd, list]) => [cmd, new Set(list)]));
  const refused = new Map(Object.entries(k.engine.refused).map(([cmd, list]) => [cmd, new Set(list)]));

  /** "fix nvt", "pair lj/cut", "pair_style lj/cut/omp" -> the index name; a page name -> itself. */
  const resolve = (raw: string): { name: string | null; page: string } | null => {
    let q = raw.trim().replace(/\s+/g, ' ');
    const words = q.split(' ');
    if (words.length === 2 && STYLE_COMMANDS[words[0]]) q = `${STYLE_COMMANDS[words[0]]} ${words[1]}`;
    for (const cand of [q, q.replace(ACCEL, '')]) {
      const page = byName.get(cand);
      if (page) return { name: cand, page };
    }
    const asPage = q.replace(/\.(html|rst)$/, '').replace(/^https?:\/\/docs\.lammps\.org\//, '');
    if (idx.pages[asPage]) return { name: null, page: asPage };
    return null;
  };

  /** Whether the browser engine runs a command or style ("fix nvt", "read_data"). */
  const support = (name: string): 'supported' | 'not supported' | 'refused' | 'unknown' => {
    const [cmd, style] = name.split(' ');
    if (style === undefined) return engineCommands.has(cmd) ? 'supported' : 'not supported';
    const plain = style.replace(ACCEL, '');
    if (plain === 'none' && cmd.endsWith('_style')) return 'supported';
    if (refused.get(cmd)?.has(plain)) return 'refused';
    const set = engineStyles.get(cmd);
    if (!set) return engineCommands.has(cmd) ? 'unknown' : 'not supported';
    return set.has(plain) ? 'supported' : 'not supported';
  };

  // the only requests this server makes: a documentation page (a plain file name) of LAMMPS on GitHub at a fixed commit
  const RAW = /^https:\/\/raw\.githubusercontent\.com\/lammps\/lammps\/[0-9a-f]{7,40}\/doc\/src\/$/;
  const pageCache = new Map<string, string>();
  const fetchPage = async (page: string): Promise<string> => {
    const hit = pageCache.get(page);
    if (hit !== undefined) return hit;
    if (!RAW.test(idx.raw) || !/^[A-Za-z0-9_-]+$/.test(page)) throw new ToolError(`refusing to fetch documentation page "${page}" from ${idx.raw}`);
    const url = `${idx.raw}${page}.rst`;
    let r;
    try {
      r = await doFetch(url);
    } catch (e) {
      throw new ToolError(`could not fetch ${url} (${e instanceof Error ? e.message : String(e)}); the page is also at ${docUrl(page)}`);
    }
    if (!r.ok) throw new ToolError(`${url} answered HTTP ${r.status}; the page is also at ${docUrl(page)}`);
    const body = await r.text();
    pageCache.set(page, body);
    return body;
  };

  // ---------------------------------------------------------------- tools

  const searchDocs = (args: Record<string, unknown>): ToolResult => {
    const query = str(args, 'query').toLowerCase().trim();
    const limit = int(args, 'limit', 10, 1, 50);
    const tokens = query.split(/[\s,]+/).filter(Boolean);
    const hits: { name: string; page: string; score: number }[] = [];
    for (const e of idx.entries) {
      const name = e.name.toLowerCase();
      const p = idx.pages[e.page];
      const hay = `${p?.title ?? ''} ${p?.summary ?? ''}`.toLowerCase();
      const extra = builderWords.get(e.page) ?? '';
      let score = name === query ? 100 : name.includes(query) ? 40 : 0;
      for (const t of tokens) {
        if (name.split(/[ /_]/).includes(t)) score += 12;
        else if (name.includes(t)) score += 6;
        if (hay.includes(t)) score += 3;
        if (extra.includes(t)) score += 3;
      }
      if (tokens.length > 1 && tokens.every((t) => name.includes(t) || hay.includes(t) || extra.includes(t))) score += 10;
      if (score > 0) hits.push({ name: e.name, page: e.page, score });
    }
    hits.sort((a, b) => b.score - a.score || a.name.length - b.name.length || a.name.localeCompare(b.name));
    const top = hits.slice(0, limit).map((h) => ({
      name: h.name, page: h.page, title: idx.pages[h.page]?.title ?? '', summary: idx.pages[h.page]?.summary ?? '',
      url: docUrl(h.page), browser: support(h.name),
    }));
    if (!top.length) return text(`No LAMMPS command or style matches "${query}". Try a style name (lj/cut, nvt) or a topic word (thermostat, rdf).`, { results: [] });
    const lines = top.map((h) => `- ${h.name} — ${h.summary || h.title} (${h.url}; browser engine: ${h.browser})`);
    return text(`LAMMPS documentation (commit ${idx.commit.slice(0, 7)}) matching "${query}":\n${lines.join('\n')}\n\nget_lammps_doc returns a page's full text.`, { results: top });
  };

  const getDoc = async (args: Record<string, unknown>): Promise<ToolResult> => {
    const name = str(args, 'name');
    const offset = int(args, 'offset', 0, 0, 10_000_000);
    const max = int(args, 'max_chars', 20000, 1000, 100000);
    const hit = resolve(name);
    if (!hit) throw new ToolError(`no LAMMPS documentation page for "${name}"; search_lammps_docs finds names (e.g. "fix nvt", "pair_style lj/cut", "read_data")`);
    const body = await fetchPage(hit.page);
    const part = body.slice(offset, offset + max);
    const more = offset + max < body.length ? `\n\n[${body.length - offset - max} more characters: call again with offset=${offset + max}]` : '';
    const builder = k.builder.filter((b) => (b.doc ?? '').includes(`/${hit.page}.html`));
    const forms = builder.length
      ? `\nScript Builder forms for this page: ${builder.map((b) => `${b.label} (${b.example.join(' / ')})`).join('; ')}`
      : '';
    const head = `${idx.pages[hit.page]?.title ?? hit.page} — ${docUrl(hit.page)}\nSource: LAMMPS doc/src/${hit.page}.rst at commit ${idx.commit.slice(0, 7)} (GPL-2.0).`
      + `${hit.name ? `\nBrowser engine support for ${hit.name}: ${support(hit.name)}.` : ''}${forms}\n\n`;
    return text(`${head}${part}${more}`, { page: hit.page, url: docUrl(hit.page), offset, length: body.length, ...(hit.name ? { name: hit.name, browser: support(hit.name) } : {}) });
  };

  const validate = (args: Record<string, unknown>): ToolResult => {
    const script = str(args, 'script');
    const diags = validateScript(script);
    const counts = diagnosticCounts(diags);
    const lines = diags.map((d) => `- line ${d.line} ${d.level} [${d.rule}] ${d.command}: ${d.message}${d.doc ? ` (${d.doc})` : ''}`);
    const head = `Script check: ${counts.errors ?? 0} errors, ${counts.warnings ?? 0} warnings, ${counts.infos ?? 0} notes.`;
    return text(diags.length ? `${head}\n${lines.join('\n')}` : `${head} No problems found.`, { counts, diagnostics: diags.map((d) => ({ ...d })) });
  };

  const checkSupport = (args: Record<string, unknown>): ToolResult => {
    const script = str(args, 'script');
    const problems: { line: number; item: string; status: string }[] = [];
    const seen = new Set<string>();
    const note = (line: number, item: string, status: string) => {
      if (status === 'supported' || status === 'unknown') return;
      const key = `${item}|${status}`;
      if (seen.has(key)) return;
      seen.add(key);
      problems.push({ line, item, status });
    };
    for (const st of scriptStatementsDetailed(script)) {
      const tok = tokenizeLine(st.text);
      const cmd = tok[0];
      if (!cmd || /[$]/.test(cmd)) continue;
      note(st.line, cmd, engineCommands.has(cmd) ? 'supported' : 'not supported');
      const style = (w: string | undefined, kind: string) => {
        if (!w || /[${}]/.test(w) || w.startsWith('v_')) return;
        note(st.line, `${kind} ${w}`, support(`${kind} ${w}`));
      };
      if (['pair_style', 'bond_style', 'angle_style', 'dihedral_style', 'improper_style', 'kspace_style', 'min_style', 'atom_style'].includes(cmd)) {
        const w = tok[1];
        if (w === 'none' || w === 'zero') continue;
        style(w, cmd);
        if (w && HYBRID.has(w)) {
          // the sub-styles: the arguments that name a documented style of this kind
          for (const a of tok.slice(2)) if (byName.has(`${cmd} ${a.replace(ACCEL, '')}`)) style(a, cmd);
        }
      } else if (cmd === 'fix' || cmd === 'compute' || cmd === 'dump') style(tok[3], cmd);
    }
    const ok = problems.length === 0;
    const lines = problems.map((p) => `- line ${p.line}: ${p.item} — ${p.status} in the browser engine`);
    return text(ok
      ? `Every command and style in this script is supported by the browser engine. Paste it into the notebook at ${notebookUrl} (Notebook tab) to run it; the engine still stops with an error that names any keyword combination it does not implement.`
      : `${problems.length} item(s) the browser engine does not run:\n${lines.join('\n')}\n\nNative LAMMPS runs them; in the browser, replace them or run the script with a local LAMMPS build (generate_cmake_build).`,
    { supported: ok, problems });
  };

  const listTemplates = (): ToolResult => text(
    k.templates.map((t) => `- ${t.id}: ${t.label} — ${t.description}`).join('\n'),
    { templates: k.templates.map(({ id, label, description }) => ({ id, label, description })) },
  );
  const getTemplate = (args: Record<string, unknown>): ToolResult => {
    const id = str(args, 'id');
    const t = k.templates.find((x) => x.id === id);
    if (!t) throw new ToolError(`no template "${id}"; list_templates names them (${k.templates.map((x) => x.id).join(', ')})`);
    return text(`# ${t.label}\n# ${t.description}\n${t.script}`, { id: t.id, script: t.script });
  };

  const listExamples = (): ToolResult => text(
    k.examples.map((e) => `- ${e.file}: ${e.name} — ${e.summary}${e.files.length ? ` (input files: ${e.files.map((f) => f.name).join(', ')})` : ''}`).join('\n'),
    { examples: k.examples.map(({ file, name, summary, files }) => ({ file, name, summary, files, url: `${k.site}notebooks/${file}` })) },
  );
  const getExample = (args: Record<string, unknown>): ToolResult => {
    const file = str(args, 'file');
    const e = k.examples.find((x) => x.file === file || x.file === `${file}.lmpnb`);
    if (!e) throw new ToolError(`no example "${file}"; list_examples names them`);
    const format = str(args, 'format', false) || 'markdown';
    const url = `${k.site}notebooks/${e.file}`;
    const files = e.files.length ? `\nInput files (inside the notebook file ${url}): ${e.files.map((f) => `${f.name} (${f.bytes} bytes)`).join(', ')}` : '';
    const body = format === 'script' ? e.script : e.markdown;
    return text(`${e.name}\n${e.summary}${files}\n\n${body}`, { file: e.file, url, script: e.script });
  };

  const cmake = (args: Record<string, unknown>): ToolResult => {
    const packages = Array.isArray(args.packages) ? args.packages.map(String).map((p) => p.toUpperCase().replace(/^PKG_/, '')) : null;
    const preset = str(args, 'preset', false);
    const known = new Set(k.compiler.packages.map((p) => p.name));
    const unknown = (packages ?? []).filter((p) => !known.has(p));
    if (unknown.length) throw new ToolError(`unknown LAMMPS package(s): ${unknown.join(', ')}; known: ${[...known].join(', ')}`);
    if (preset && !k.compiler.presets.some((p) => p.id === preset)) throw new ToolError(`unknown preset "${preset}"; presets: ${k.compiler.presets.map((p) => p.id).join(', ')}`);
    const acc = str(args, 'accelerator', false);
    if (acc && !k.compiler.accelerators.some((a) => a.id === acc)) throw new ToolError(`unknown accelerator "${acc}"; one of ${k.compiler.accelerators.map((a) => a.id).join(', ')}`);
    const os = str(args, 'os', false);
    if (os && os !== 'linux' && os !== 'windows') throw new ToolError('os must be linux or windows');
    const buildType = str(args, 'build_type', false);
    if (buildType && !['Release', 'Debug', 'RelWithDebInfo'].includes(buildType)) throw new ToolError('build_type must be Release, Debug or RelWithDebInfo');
    const o: CompilerOptions = {
      ...DEFAULT_COMPILER_OPTIONS,
      options: { ...DEFAULT_COMPILER_OPTIONS.options },
      presetId: packages ? (preset || '') : (preset || DEFAULT_COMPILER_OPTIONS.presetId),
      manualPackages: packages ?? [],
      ...(acc ? { accelerator: acc as CompilerOptions['accelerator'] } : {}),
      ...(os ? { os: os as CompilerOptions['os'] } : {}),
      ...(buildType ? { buildType: buildType as CompilerOptions['buildType'] } : {}),
      ...(typeof args.mpi === 'boolean' ? { withMpi: args.mpi } : {}),
      ...(args.jobs !== undefined ? { jobs: int(args, 'jobs', 8, 1, 256) } : {}),
      ...(str(args, 'branch', false) ? { branch: str(args, 'branch') } : {}),
    };
    const r = generateBuildScript(o);
    return text(`${r.text}${r.warnings.length ? `\n\n# Warnings:\n${r.warnings.map((w) => `# - ${w}`).join('\n')}` : ''}`, { flags: r.flags, warnings: r.warnings, script: r.text });
  };

  const readGuide = (args: Record<string, unknown>): ToolResult => {
    const id = str(args, 'id');
    const g = k.guides.find((x) => x.id === id);
    if (!g) throw new ToolError(`no guide "${id}"; guides: ${k.guides.map((x) => x.id).join(', ')}`);
    return text(g.text, { id: g.id, title: g.title });
  };

  const readOnly = { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false };
  const TOOLS: { name: string; title: string; description: string; inputSchema: Json; annotations: Record<string, boolean>; run: (a: Record<string, unknown>) => ToolResult | Promise<ToolResult> }[] = [
    {
      name: 'search_lammps_docs', title: 'Search the LAMMPS documentation',
      description: 'Find LAMMPS commands and styles (all of the documentation index at the engine\'s LAMMPS commit) by name or topic, e.g. "nose-hoover thermostat", "rdf", "lj/cut", "read_data". Each hit gives the doc page, a one-sentence summary and whether the in-browser engine runs it.',
      inputSchema: { type: 'object', properties: { query: { type: 'string', description: 'Style name, command or topic words' }, limit: { type: 'integer', minimum: 1, maximum: 50, default: 10 } }, required: ['query'] },
      annotations: readOnly, run: searchDocs,
    },
    {
      name: 'get_lammps_doc', title: 'Read a LAMMPS documentation page',
      description: 'The full text (reStructuredText) of the LAMMPS documentation page for a command or style — syntax, keywords, defaults, restrictions — fetched from github.com/lammps/lammps at the engine\'s commit. Name it as in an input script ("fix nvt", "pair_style lj/cut/coul/long", "kspace_style pppm", "read_data") or by page ("fix_nh"). Long pages come in parts: pass offset to continue.',
      inputSchema: { type: 'object', properties: { name: { type: 'string' }, offset: { type: 'integer', minimum: 0, default: 0 }, max_chars: { type: 'integer', minimum: 1000, maximum: 100000, default: 20000 } }, required: ['name'] },
      annotations: { ...readOnly, openWorldHint: true }, run: getDoc,
    },
    {
      name: 'validate_lammps_script', title: 'Check a LAMMPS input script',
      description: 'Runs the LAMMPS Web GUI script checker on an input script: command order (units and atom_style before the box, pair_coeff after pair_style, ...), missing definitions, deprecated or removed commands, and units-dependent mistakes. Every rule cites its docs.lammps.org page.',
      inputSchema: { type: 'object', properties: { script: { type: 'string', description: 'The LAMMPS input script text' } }, required: ['script'] },
      annotations: readOnly, run: validate,
    },
    {
      name: 'check_browser_support', title: 'Can the browser engine run this script?',
      description: 'Lists the commands and styles of an input script that the in-browser LAMMPS engine of the LAMMPS Web GUI does not run (it implements a large subset of LAMMPS, checked against native LAMMPS). Empty means the script can run in the site\'s Notebook.',
      inputSchema: { type: 'object', properties: { script: { type: 'string' } }, required: ['script'] },
      annotations: readOnly, run: checkSupport,
    },
    {
      name: 'list_templates', title: 'List starter input scripts',
      description: 'The Script Builder\'s starter pipelines (LJ melt, NVT/NPT equilibration, minimization, ...): each a complete input script that passes the script checker.',
      inputSchema: { type: 'object', properties: {} }, annotations: readOnly, run: listTemplates,
    },
    {
      name: 'get_template', title: 'Get a starter input script',
      description: 'The full input script of a starter template (ids from list_templates).',
      inputSchema: { type: 'object', properties: { id: { type: 'string' } }, required: ['id'] },
      annotations: readOnly, run: getTemplate,
    },
    {
      name: 'list_examples', title: 'List example notebooks',
      description: 'Worked example notebooks of the site (water from a data file, CO2 and butane from molecule templates, crystals from CIF files, ...) with their input files.',
      inputSchema: { type: 'object', properties: {} }, annotations: readOnly, run: listExamples,
    },
    {
      name: 'get_example', title: 'Get an example notebook',
      description: 'An example notebook as Markdown (explanations and the input in fenced blocks) or as one input script (format "script").',
      inputSchema: { type: 'object', properties: { file: { type: 'string' }, format: { type: 'string', enum: ['markdown', 'script'], default: 'markdown' } }, required: ['file'] },
      annotations: readOnly, run: getExample,
    },
    {
      name: 'generate_cmake_build', title: 'Write a LAMMPS CMake build script',
      description: 'A shell (Linux) or PowerShell (Windows) script that clones LAMMPS and builds it with CMake: packages by preset or by name (e.g. ["MOLECULE","KSPACE","RIGID"]), an accelerator (openmp, gpu-cuda, kokkos-cuda, ...), MPI, build type.',
      inputSchema: {
        type: 'object',
        properties: {
          packages: { type: 'array', items: { type: 'string' } },
          preset: { type: 'string', enum: k.compiler.presets.map((p) => p.id) },
          accelerator: { type: 'string', enum: k.compiler.accelerators.map((a) => a.id) },
          os: { type: 'string', enum: ['linux', 'windows'] },
          mpi: { type: 'boolean' },
          build_type: { type: 'string', enum: ['Release', 'Debug', 'RelWithDebInfo'] },
          jobs: { type: 'integer', minimum: 1, maximum: 256 },
          branch: { type: 'string', description: 'LAMMPS git branch or tag (default develop)' },
        },
      },
      annotations: readOnly, run: cmake,
    },
    {
      name: 'read_guide', title: 'Read a guide',
      description: `Guides on writing LAMMPS input and on the browser engine: ${k.guides.map((g) => `"${g.id}" (${g.title})`).join(', ')}.`,
      inputSchema: { type: 'object', properties: { id: { type: 'string', enum: k.guides.map((g) => g.id) } }, required: ['id'] },
      annotations: readOnly, run: readGuide,
    },
  ];

  // ---------------------------------------------------------------- resources and prompts

  const RES = 'lammps-web-gui://';
  const resources = (): Record<string, unknown>[] => [
    ...k.guides.map((g) => ({ uri: `${RES}guides/${g.id}`, name: g.id, title: g.title, description: g.description, mimeType: 'text/markdown' })),
    ...k.templates.map((t) => ({ uri: `${RES}templates/${t.id}`, name: `template ${t.id}`, title: t.label, description: t.description, mimeType: 'text/plain' })),
    ...k.examples.map((e) => ({ uri: `${RES}examples/${e.file}`, name: `example ${e.file}`, title: e.name, description: e.summary, mimeType: 'text/markdown' })),
    { uri: `${RES}engine/support`, name: 'engine support', title: 'Browser engine: supported commands and styles', mimeType: 'application/json' },
  ];
  const readResource = async (uri: string): Promise<{ uri: string; mimeType: string; text: string }> => {
    const rest = uri.startsWith(RES) ? uri.slice(RES.length) : null;
    const [kind, id] = rest ? [rest.slice(0, rest.indexOf('/')), rest.slice(rest.indexOf('/') + 1)] : ['', ''];
    if (kind === 'guides') { const g = k.guides.find((x) => x.id === id); if (g) return { uri, mimeType: 'text/markdown', text: g.text }; }
    if (kind === 'templates') { const t = k.templates.find((x) => x.id === id); if (t) return { uri, mimeType: 'text/plain', text: t.script }; }
    if (kind === 'examples') { const e = k.examples.find((x) => x.file === id); if (e) return { uri, mimeType: 'text/markdown', text: e.markdown }; }
    if (kind === 'engine' && id === 'support') return { uri, mimeType: 'application/json', text: JSON.stringify(k.engine, null, 1) };
    const doc = /^lammps-doc:\/\/(.+)$/.exec(uri);
    if (doc) {
      const hit = resolve(decodeURIComponent(doc[1]));
      if (hit) return { uri, mimeType: 'text/x-rst', text: await fetchPage(hit.page) };
    }
    throw new RpcError(-32002, `Resource not found: ${uri}`);
  };

  const PROMPT = {
    name: 'write_lammps_input', title: 'Write a LAMMPS input script',
    description: 'Write and check a LAMMPS input script for a simulation goal, using the documentation and the script checker.',
    arguments: [
      { name: 'goal', description: 'What the simulation should do (system, ensemble, outputs)', required: true },
      { name: 'browser', description: '"yes" if it should run in the LAMMPS Web GUI browser engine', required: false },
    ],
  };
  const promptText = (goal: string, browser: boolean) => [
    `Write a LAMMPS input script for: ${goal}`,
    '',
    'Work from the documentation, not from memory:',
    '1. read_guide "writing-inputs" for the structure of an input script;',
    '2. search_lammps_docs for the commands and styles you need, and get_lammps_doc for the syntax of each one you use (keywords, defaults, restrictions);',
    '3. start from a close template (list_templates / get_template) or example (list_examples) when one fits;',
    '4. run validate_lammps_script on the result and fix every error and warning;',
    browser
      ? `5. run check_browser_support and replace anything the browser engine does not run; then give the script with a note that it runs in the Notebook at ${notebookUrl}.`
      : '5. say which LAMMPS packages the script needs (generate_cmake_build can write the build script).',
  ].join('\n');

  // ---------------------------------------------------------------- JSON-RPC

  class RpcError extends Error {
    constructor(readonly code: number, message: string) { super(message); }
  }

  const call = async (method: string, params: Record<string, unknown>): Promise<unknown> => {
    switch (method) {
      case 'initialize': {
        const asked = typeof params.protocolVersion === 'string' ? params.protocolVersion : '';
        return {
          protocolVersion: PROTOCOL_VERSIONS.includes(asked) ? asked : PROTOCOL_VERSIONS[0],
          capabilities: { tools: { listChanged: false }, resources: { listChanged: false, subscribe: false }, prompts: { listChanged: false } },
          serverInfo: { name: k.name, title: 'LAMMPS Web GUI', version: k.version, websiteUrl: k.site },
          instructions: `LAMMPS input-writing help from the LAMMPS Web GUI (${k.site}). Look commands up with search_lammps_docs and get_lammps_doc (the LAMMPS documentation at commit ${idx.commit.slice(0, 7)}) rather than from memory, check scripts with validate_lammps_script, and use check_browser_support when a script should run in the site's in-browser engine.`,
        };
      }
      case 'ping': return {};
      case 'tools/list': return { tools: TOOLS.map(({ run: _run, ...t }) => t) };
      case 'tools/call': {
        const name = String(params.name ?? '');
        const tool = TOOLS.find((t) => t.name === name);
        if (!tool) throw new RpcError(-32602, `Unknown tool: ${name}`);
        const args = (params.arguments && typeof params.arguments === 'object' ? params.arguments : {}) as Record<string, unknown>;
        try {
          return await tool.run(args);
        } catch (e) {
          if (e instanceof ToolError) return { content: [{ type: 'text', text: e.message }], isError: true };
          throw e;
        }
      }
      case 'resources/list': return { resources: resources() };
      case 'resources/templates/list':
        return { resourceTemplates: [{ uriTemplate: 'lammps-doc://{name}', name: 'LAMMPS documentation page', description: 'A LAMMPS doc page by command, style ("fix nvt") or page name ("fix_nh"), as reStructuredText', mimeType: 'text/x-rst' }] };
      case 'resources/read': return { contents: [await readResource(String(params.uri ?? ''))] };
      case 'prompts/list': return { prompts: [PROMPT] };
      case 'prompts/get': {
        if (params.name !== PROMPT.name) throw new RpcError(-32602, `Unknown prompt: ${String(params.name)}`);
        const a = (params.arguments ?? {}) as Record<string, string>;
        if (!a.goal) throw new RpcError(-32602, 'Missing required argument: goal');
        return { description: PROMPT.description, messages: [{ role: 'user', content: { type: 'text', text: promptText(a.goal, /^(y|yes|true|1)$/i.test(a.browser ?? '')) } }] };
      }
      case 'completion/complete': return { completion: { values: [], hasMore: false } };
      case 'logging/setLevel': return {};
      default: throw new RpcError(-32601, `Method not found: ${method}`);
    }
  };

  /** One JSON-RPC message in, its reply out (null for notifications and replies). */
  const handle = async (msg: unknown): Promise<Reply | null> => {
    const m = msg as Partial<Request>;
    if (!m || typeof m !== 'object' || m.jsonrpc !== '2.0' || typeof m.method !== 'string') {
      if (m && typeof m === 'object' && 'id' in m && ('result' in m || 'error' in m)) return null;
      return { jsonrpc: '2.0', id: (m && typeof m === 'object' && 'id' in m ? (m.id as string | number | null) : null) ?? null, error: { code: -32600, message: 'Invalid Request' } };
    }
    const isNotification = !('id' in m) || m.id === undefined;
    try {
      const result = await call(m.method, (m.params ?? {}) as Record<string, unknown>);
      return isNotification ? null : { jsonrpc: '2.0', id: m.id ?? null, result };
    } catch (e) {
      if (isNotification) return null;
      const code = e instanceof RpcError ? e.code : -32603;
      return { jsonrpc: '2.0', id: m.id ?? null, error: { code, message: e instanceof Error ? e.message : String(e) } };
    }
  };

  return { handle, tools: TOOLS.map((t) => t.name) };
};

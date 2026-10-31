/**
 * What the MCP server (mcp/server.ts) knows, gathered at build time (mcp/build.mjs) from the app's own sources: the
 * LAMMPS documentation index (mcp/data/lammps-doc-index.json), the Script Builder catalog, the starter templates, the
 * example notebooks, the browser engine's supported commands and styles, the CMake package catalog and the guides.
 * The result is plain JSON: the server embeds it, and the site serves it at /mcp/knowledge.json.
 *
 * This module imports the engine (for its style lists), so only the build and the tests load it; the server imports
 * its types only.
 */
import { ALL_COMMANDS, defaultParams, type ParamType } from '../src/lammps/catalog';
import { SCRIPT_TEMPLATES, buildTemplate } from '../src/lammps/templates';
import { generateScript } from '../src/lammps/generator';
import { ACCELERATORS, BUILD_OPTIONS, LMP_PACKAGES, PRESETS } from '../src/lammps/compiler';
import { styleNames } from '../src/engine/styles';
import { SUPPORTED_COMMANDS } from '../src/engine/session';
import { REFUSED_STYLES } from '../src/engine/refusedStyles';
import { ATOM_STYLES } from '../src/engine/commands/setup';
import { DUMP_STYLES, GZ_STYLES, LAZY_DUMP_STYLES, ZSTD_STYLES } from '../src/engine/commands/run';
import { LAZY_GROUPS } from '../src/engine/lazyStyles';

export interface DocIndex {
  source: string;
  /** The LAMMPS commit the index and the fetched pages come from (the engine's parity commit). */
  commit: string;
  docs: string;
  /** Base URL of the raw .rst pages at that commit. */
  raw: string;
  entries: { name: string; page: string }[];
  pages: Record<string, { title: string; summary: string }>;
}

export interface BuilderCommand {
  id: string;
  command: string;
  label: string;
  section: string;
  category: string;
  doc?: string;
  deprecated?: string;
  params: { key: string; label: string; type: ParamType; options?: string[]; default?: string; help?: string }[];
  /** The line(s) the Script Builder writes with the default parameters. */
  example: string[];
}

export interface Guide { id: string; title: string; description: string; text: string }

export interface Example {
  file: string;
  name: string;
  summary: string;
  /** The code cells, in order: the input script the notebook runs. */
  script: string;
  /** The notebook as Markdown (text cells, and code cells as fenced blocks). */
  markdown: string;
  /** Input files the notebook carries (name and size; the content is in the notebook file). */
  files: { name: string; bytes: number }[];
}

export interface Knowledge {
  name: string;
  version: string;
  /** The deployed site, with a trailing slash. */
  site: string;
  docIndex: DocIndex;
  builder: BuilderCommand[];
  templates: { id: string; label: string; description: string; script: string }[];
  examples: Example[];
  engine: {
    /** Input-script commands the browser engine runs. */
    commands: string[];
    /** Supported style names per command (pair_style, fix, compute, atom_style, dump, ...). */
    styles: Record<string, string[]>;
    /** Styles the engine registers only to refuse with a specific reason. */
    refused: Record<string, string[]>;
  };
  compiler: {
    packages: { name: string; description: string; category: string; heavy?: boolean }[];
    accelerators: { id: string; label: string; vendor: string; packages: string[]; notes?: string }[];
    presets: { id: string; label: string; description: string; packages: string[] }[];
    options: { key: string; label: string; values: string[]; default: string; help?: string }[];
  };
  guides: Guide[];
}

/** Raw inputs that live in files: the doc index, the example notebooks (public/notebooks) and the guide texts. */
export interface KnowledgeSources {
  version: string;
  site: string;
  /** mcp/data/lammps-doc-index.json (written by mcp/scripts/doc-index.mjs). */
  docIndex: DocIndex;
  /** public/notebooks/index.json, and each listed .lmpnb file's text by name. */
  notebookIndex: { notebooks: { file: string; name: string; summary: string }[] };
  notebooks: Record<string, string>;
  guides: Guide[];
}

const REFUSED_KEYS: Record<string, string> = {
  PAIRS: 'pair_style', BONDS: 'bond_style', ANGLES: 'angle_style', DIHEDRALS: 'dihedral_style', IMPROPERS: 'improper_style',
  KSPACES: 'kspace_style', FIXES: 'fix', COMPUTES: 'compute', MINIMIZERS: 'min_style',
};

interface NotebookCell { kind: string; text: string }

const example = (entry: { file: string; name: string; summary: string }, text: string): Example => {
  const doc = JSON.parse(text) as { cells: NotebookCell[]; inputFiles?: Record<string, string> };
  const code = doc.cells.filter((c) => c.kind !== 'markdown').map((c) => c.text.trimEnd());
  const markdown = doc.cells.map((c) => (c.kind === 'markdown' ? c.text.trimEnd() : `\`\`\`lammps\n${c.text.trimEnd()}\n\`\`\``)).join('\n\n');
  const files = Object.entries(doc.inputFiles ?? {}).map(([name, body]) => ({ name, bytes: body.length }));
  return { ...entry, script: `${code.join('\n\n')}\n`, markdown: `${markdown}\n`, files };
};

export const buildKnowledge = (src: KnowledgeSources): Knowledge => ({
  name: 'lammps-web-gui',
  version: src.version,
  site: src.site,
  docIndex: src.docIndex,
  builder: ALL_COMMANDS.map((d) => ({
    id: d.id, command: d.command, label: d.label, section: d.section, category: d.category,
    ...(d.doc ? { doc: d.doc } : {}),
    ...(d.deprecated ? { deprecated: d.deprecated } : {}),
    params: d.params.map((p) => ({
      key: p.key, label: p.label, type: p.type,
      ...(p.options ? { options: p.options.map((o) => o.value) } : {}),
      ...(p.default !== undefined ? { default: p.default } : {}),
      ...(p.help ? { help: p.help } : {}),
    })),
    example: d.build(defaultParams(d)),
  })),
  templates: SCRIPT_TEMPLATES.map((t) => ({
    id: t.id, label: t.label, description: t.description, script: generateScript(buildTemplate(t)).text,
  })),
  examples: src.notebookIndex.notebooks.filter((e) => src.notebooks[e.file] !== undefined).map((e) => example(e, src.notebooks[e.file])),
  engine: {
    // SUPPORTED_COMMANDS and the commands of the lazily loaded groups (prd, tad, temper, pair_write, ...)
    commands: [...new Set([...SUPPORTED_COMMANDS, ...LAZY_GROUPS.flatMap((g) => [...(g.names.COMMANDS ?? [])])])].sort(),
    styles: { ...styleNames(), atom_style: [...ATOM_STYLES, 'hybrid'], dump: [...DUMP_STYLES, ...LAZY_DUMP_STYLES, ...GZ_STYLES, ...ZSTD_STYLES] },
    refused: Object.fromEntries(Object.entries(REFUSED_STYLES).map(([k, v]) => [REFUSED_KEYS[k] ?? k, [...(v ?? [])]])),
  },
  compiler: {
    packages: LMP_PACKAGES.map((p) => ({ ...p })),
    accelerators: ACCELERATORS.map((a) => ({ id: a.id, label: a.label, vendor: a.vendor, packages: [...a.packages], ...(a.notes ? { notes: a.notes } : {}) })),
    presets: PRESETS.map((p) => ({ id: p.id, label: p.label, description: p.description, packages: [...p.packages] })),
    options: BUILD_OPTIONS.map((o) => ({ ...o })),
  },
  guides: src.guides,
});

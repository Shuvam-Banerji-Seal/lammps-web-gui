import {
  COMMAND_BY_ID,
  SECTION_LABELS,
  ScriptBranch,
  ScriptModel,
  ScriptSection,
  ScriptStep,
  SECTION_ORDER,
} from './catalog';

export interface GeneratedScript {
  /** Full input text ready to save as in.lammps */
  text: string;
  /** Emitted (non-empty) command lines with their originating step uid. */
  emitted: { uid: string; line: string }[];
  /** Non-fatal issues: empty required params, disabled steps, etc. */
  warnings: string[];
}

/**
 * How the step list maps onto emitted lines.
 *
 * - `pipeline` (default): emit in exactly the order shown in the flowchart.
 *   LAMMPS input is executed top-to-bottom, so this is the only order that
 *   round-trips faithfully and the only one that keeps multi-stage scripts
 *   correct (`run` → `reset_timestep` → `run`, `run` → `write_data`, a second
 *   `dump` opened after the first stage, …).
 * - `section`: legacy grouping by canonical section. Kept for the explicit
 *   "Sort by section" action, which physically reorders the steps so the
 *   flowchart still matches the script.
 */
export type EmitOrder = 'pipeline' | 'section';

export interface GenerateOptions {
  order?: EmitOrder;
}

const isFilled = (v: string | undefined): boolean =>
  v !== undefined && v.trim() !== '';

/* ------------------------------------------------------------------ */
/* Branch resolution                                                   */
/* ------------------------------------------------------------------ */

/** A step in the resolved execution path, tagged with its owning branch. */
export interface ResolvedStep {
  step: ScriptStep;
  /** null = trunk; otherwise the ScriptBranch.id this step came from. */
  branchId: string | null;
}

export const branchesOf = (model: ScriptModel): ScriptBranch[] =>
  model.branches ?? [];

/** Fork points present in the model, in trunk order (null fork first). */
export const forkPoints = (model: ScriptModel): (string | null)[] => {
  const seen = new Set<string>();
  const order: (string | null)[] = [];
  let hasStartFork = false;
  for (const b of branchesOf(model)) {
    if (b.forkAfter === null) hasStartFork = true;
    else seen.add(b.forkAfter);
  }
  if (hasStartFork) order.push(null);
  for (const s of model.steps) if (seen.has(s.uid)) order.push(s.uid);
  return order;
};

/** The branch taken at a fork point, or null when the trunk is followed. */
export const takenBranchAt = (
  model: ScriptModel,
  forkAfter: string | null,
): ScriptBranch | null => {
  const active = new Set(model.activeBranchIds ?? []);
  return (
    branchesOf(model).find(b => b.forkAfter === forkAfter && active.has(b.id)) ??
    null
  );
};

/**
 * Walk the trunk, splicing in whichever branch is taken at each fork point.
 *
 * A non-rejoining branch ends the walk: its steps replace the trunk tail,
 * which is what makes a branch a genuinely divergent concept rather than an
 * insertion.
 */
export const resolvePath = (model: ScriptModel): ResolvedStep[] => {
  const out: ResolvedStep[] = [];
  const push = (b: ScriptBranch) => {
    for (const s of b.steps) out.push({ step: s, branchId: b.id });
  };

  const atStart = takenBranchAt(model, null);
  if (atStart) {
    push(atStart);
    if (!atStart.rejoin) return out;
  }

  for (const step of model.steps) {
    out.push({ step, branchId: null });
    const b = takenBranchAt(model, step.uid);
    if (!b) continue;
    push(b);
    if (!b.rejoin) return out;
  }
  return out;
};

/** Trunk steps that the taken branches cut off (shown ghosted in the UI). */
export const skippedTrunkUids = (model: ScriptModel): Set<string> => {
  const resolved = new Set(
    resolvePath(model)
      .filter(r => r.branchId === null)
      .map(r => r.step.uid),
  );
  return new Set(model.steps.filter(s => !resolved.has(s.uid)).map(s => s.uid));
};

/* ------------------------------------------------------------------ */
/* Ordering helpers                                                    */
/* ------------------------------------------------------------------ */

const sectionRank = (section: ScriptSection): number => {
  const i = SECTION_ORDER.indexOf(section);
  return i < 0 ? SECTION_ORDER.length : i;
};

/**
 * Stable sort of a step list into canonical section order. Exposed so the
 * builder can offer an explicit "Sort by section" action — the steps really
 * move, so the flowchart never disagrees with the emitted script.
 */
export const sortStepsBySection = (steps: ScriptStep[]): ScriptStep[] =>
  steps
    .map((step, i) => ({ step, i }))
    .sort((a, b) => {
      const sa = COMMAND_BY_ID[a.step.defId]?.section;
      const sb = COMMAND_BY_ID[b.step.defId]?.section;
      const ra = sa ? sectionRank(sa) : SECTION_ORDER.length;
      const rb = sb ? sectionRank(sb) : SECTION_ORDER.length;
      return ra === rb ? a.i - b.i : ra - rb;
    })
    .map(e => e.step);

/** True when the steps already sit in canonical section order. */
export const isSectionSorted = (steps: ScriptStep[]): boolean => {
  let last = -1;
  for (const s of steps) {
    const sec = COMMAND_BY_ID[s.defId]?.section;
    if (!sec) continue;
    const r = sectionRank(sec);
    if (r < last) return false;
    last = r;
  }
  return true;
};

/* ------------------------------------------------------------------ */
/* Script generation                                                   */
/* ------------------------------------------------------------------ */

/**
 * Render the script for a model.
 *
 * Rules:
 * - MANUAL OVERRIDE: when model.manualText is set to non-empty text, it is
 *   emitted verbatim and the step list is bypassed (a warning says so).
 * - Steps are emitted in PIPELINE order — exactly the order the flowchart
 *   shows — because LAMMPS executes an input file top to bottom. A `# ---- `
 *   banner is written whenever the section changes, so canonically ordered
 *   scripts still read like the hand-written examples in the docs.
 * - Whichever branch is taken at each fork point is spliced in; a branch that
 *   does not rejoin replaces the trunk tail.
 * - Disabled steps are skipped SILENTLY (the flowchart shows them dashed);
 *   only structurally broken steps produce warnings.
 * - Steps whose required params are blank are skipped + warned about.
 * - A header comment block documents title, generator, and the taken branches.
 */
export const generateScript = (
  model: ScriptModel,
  opts: GenerateOptions = {},
): GeneratedScript => {
  if (model.manualText !== undefined && model.manualText.trim() !== '') {
    return {
      text: model.manualText,
      emitted: [],
      warnings: [
        'Manual edit mode — the builder step list is bypassed. ' +
          'Exit manual editing to regenerate from your steps.',
      ],
    };
  }

  const warnings: string[] = [];
  const emitted: { uid: string; line: string }[] = [];
  const chunks: string[] = [];

  const taken = branchesOf(model).filter(b =>
    (model.activeBranchIds ?? []).includes(b.id),
  );

  chunks.push(
    '# ----------------------------------------------------------------',
    `# ${model.title || 'LAMMPS simulation'}`,
    '# Generated by Molecule3D — LAMMPS Workbench by Shuvam Banerji Seal',
    `# ${new Date().toISOString().slice(0, 10)}`,
  );
  for (const b of taken) {
    chunks.push(`# Branch: ${b.label}${b.rejoin ? ' (rejoins main line)' : ''}`);
  }
  chunks.push(
    '# ----------------------------------------------------------------',
    '',
  );

  let path = resolvePath(model);
  if (opts.order === 'section') {
    const sorted = sortStepsBySection(path.map(r => r.step));
    const branchOf = new Map(path.map(r => [r.step.uid, r.branchId]));
    path = sorted.map(step => ({ step, branchId: branchOf.get(step.uid) ?? null }));
  }

  let lastSection: ScriptSection | null = null;
  let wroteAnyBanner = false;

  for (const { step } of path) {
    if (!step.enabled) continue;
    const def = COMMAND_BY_ID[step.defId];
    if (!def) {
      warnings.push(`Unknown command id "${step.defId}" — step skipped.`);
      continue;
    }

    // Required-param check happens BEFORE the banner so a skipped step never
    // leaves an orphan section header behind.
    const missing = def.params.filter(
      pd =>
        pd.type !== 'flag' &&
        !isFilled(step.params[pd.key]) &&
        isRequired(def.id, pd.key),
    );
    if (missing.length > 0) {
      warnings.push(
        `${def.label}: missing ${missing.map(m => m.label).join(', ')} — skipped.`,
      );
      continue;
    }

    const lines = def.build(step.params).filter(l => l.trim() !== '');
    if (lines.length === 0) continue;

    if (def.section !== lastSection) {
      if (wroteAnyBanner) chunks.push('');
      chunks.push(`# ---- ${SECTION_LABELS[def.section]} ----`);
      lastSection = def.section;
      wroteAnyBanner = true;
    }

    if (step.note?.trim()) chunks.push(`# ${step.note.trim()}`);
    for (const l of lines) {
      chunks.push(l);
      emitted.push({ uid: step.uid, line: l });
    }
  }

  chunks.push('');
  chunks.push('# End of generated input');
  chunks.push('');

  return { text: chunks.join('\n'), emitted, warnings };
};

/** Params that must be non-empty per command (beyond ParamDef.required). */
const REQUIRED: Record<string, string[]> = {
  'read_data': ['file'],
  'read_restart': ['file'],
  'create_box': ['ntypes', 'region'],
  'region_block': ['id'],
  'region_sphere': ['id'],
  'region_cylinder': ['id'],
  'region_prism': ['id'],
  'create_bonds': ['args'],
  'change_box': ['args'],
  'read_dump': ['file', 'fields'],
  'molecule_cmd': ['file'],
  'pair_style_hybrid': ['substyles'],
  'pair_write': ['args'],
  'write_dump': ['fields'],
  'fix_ave_time': ['values'],
  'fix_ave_histo': ['values'],
  'fix_print_out': ['text'],
  'compute_reduce': ['input'],
  'fix_move': ['args'],
  'fix_modify_cmd': ['fixid'],
  'compute_modify': ['compid'],
  'unfix_cmd': ['fixid'],
  'undump_cmd': ['dumpid'],
  'uncompute_cmd': ['compid'],
  'variable_atom': ['name', 'expr'],
  'if_cmd': ['condition', 'thenBlock'],
  'label_cmd': ['id'],
  'jump_cmd': ['file'],
  'next_cmd': ['vars'],
  'include_cmd': ['file'],
  'shell_cmd': ['cmd'],
};

const isRequired = (defId: string, key: string): boolean =>
  (REQUIRED[defId] ?? []).includes(key);

/* ------------------------------------------------------------------ */
/* Flowchart derivation                                                */
/* ------------------------------------------------------------------ */

export interface FlowNode {
  uid: string;
  defId: string;
  label: string;
  sublabel: string;
  section: string;
  enabled: boolean;
  /** null = trunk step; otherwise the branch this node belongs to. */
  branchId?: string | null;
  /** Branch label, for renderers that annotate divergent segments. */
  branchLabel?: string;
}

export interface FlowEdge {
  from: string;
  to: string;
  /** Edge sits at a fork point — renderers may draw it as a diamond. */
  fork?: boolean;
}

export interface FlowGraph {
  start: boolean;
  end: boolean;
  nodes: FlowNode[];
  edges: FlowEdge[];
  /** Fork points with the branch taken at each (for legends/exports). */
  forks: { afterUid: string | null; takenBranchId: string | null; total: number }[];
}

const nodeFor = (step: ScriptStep, branch: ScriptBranch | null): FlowNode => {
  const def = COMMAND_BY_ID[step.defId];
  return {
    uid: step.uid,
    defId: step.defId,
    label: def.command === 'fix' ? fixLabel(def.id) : def.command,
    sublabel: shortParams(def, step),
    section: def.section,
    enabled: step.enabled,
    branchId: branch?.id ?? null,
    branchLabel: branch?.label,
  };
};

/**
 * Derive the pipeline graph for the CURRENTLY TAKEN path. Disabled steps
 * appear as dashed nodes so the user can see what they switched off; steps
 * that a divergent branch cut off are simply absent (the builder ghosts them
 * separately via `skippedTrunkUids`).
 */
export const deriveFlowchart = (model: ScriptModel): FlowGraph => {
  const nodes: FlowNode[] = [];
  const edges: FlowEdge[] = [];
  const branchById = new Map(branchesOf(model).map(b => [b.id, b]));

  const path = resolvePath(model).filter(r => COMMAND_BY_ID[r.step.defId]);
  path.forEach((r, i) => {
    nodes.push(nodeFor(r.step, r.branchId ? branchById.get(r.branchId) ?? null : null));
    if (i > 0) {
      const prev = path[i - 1];
      edges.push({
        from: prev.step.uid,
        to: r.step.uid,
        fork: prev.branchId === null && r.branchId !== null,
      });
    }
  });

  const forks = forkPoints(model).map(afterUid => ({
    afterUid,
    takenBranchId: takenBranchAt(model, afterUid)?.id ?? null,
    total: branchesOf(model).filter(b => b.forkAfter === afterUid).length,
  }));

  return { start: nodes.length > 0, end: nodes.length > 0, nodes, edges, forks };
};

const fixLabel = (defId: string): string =>
  // fix.<style> ids encode the style for readability
  `fix ${defId.split('.')[1] ?? ''}`.trim();

const shortParams = (
  def: import('./catalog').CommandDef,
  step: ScriptStep
): string => {
  const parts: string[] = [];
  for (const pd of def.params.slice(0, 3)) {
    const v = step.params[pd.key];
    if (v && v.trim()) parts.push(String(v).slice(0, 18));
  }
  return parts.join(' · ');
};

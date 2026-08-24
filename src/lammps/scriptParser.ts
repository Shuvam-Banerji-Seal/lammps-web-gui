/**
 * LAMMPS input-script importer — reverses the generator.
 *
 * Parses an `in.*` script into a ScriptModel so the flowchart, parameter
 * editors and warnings all light up for EXISTING scripts.
 *
 * Strategy (per statement):
 *  1. Tokenize the line (quote-aware; `&` continuations joined; comments
 *     stripped) — LAMMPS parsing rules per docs.lammps.org/Commands_parse.html.
 *  2. Score every catalog CommandDef sharing the statement's command keyword
 *     by matching its BUILD SIGNATURE against the tokens:
 *      - Each def's build() is invoked with per-param sentinels
 *        (enums → their first option, everything else → a unique slot mark).
 *      - Literal pattern tokens must match the input verbatim.
 *      - Slot tokens consume one input token each; a trailing string/text
 *        slot absorbs the remainder.
 *      - Commands that take a user-chosen ID (fix/dump/compute/region/…)
 *        treat the token after the command keyword as a wildcard so
 *        `fix myNvt all nvt …` still matches the fix_nvt definition.
 *      - Enum slots validate against their option list for scoring, so
 *        `pair_style hybrid/overlay …` picks pair_style_hybrid over
 *        pair_style_popular.
 *  3. Best-scoring def wins; params are extracted from the consumed slots.
 *  4. Unmatched statements become `raw_line` steps — nothing is lost.
 */

import {
  ALL_COMMANDS,
  COMMAND_BY_ID,
  CommandDef,
  ParamDef,
  ScriptModel,
  ScriptStep,
  defaultParams,
} from './catalog';

export interface ImportResult {
  model: ScriptModel;
  stats: { total: number; recognized: number; raw: number };
}

/** Commands whose first argument is a user-chosen ID (wildcard on import). */
const ID_FLEX = new Set([
  'fix', 'dump', 'compute', 'region', 'group', 'variable', 'label',
  'molecule', 'undump', 'unfix', 'uncompute', 'dump_modify',
  'compute_modify', 'fix_modify',
]);
/** Param keys that hold the user-chosen ID (flex token maps back into it). */
const ID_PARAM_KEYS = new Set(['id', 'name', 'fixid', 'dumpid', 'compid']);

const SLOT = '\u0000';
const slotOf = (i: number) => `${SLOT}${i}`;

/** Quote-aware tokenizer: keeps "…" as one token. */
export const tokenizeLine = (line: string): string[] => {
  const tokens: string[] = [];
  const re = /"([^"]*)"|'([^']*)'|(\S+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(line)) !== null) {
    tokens.push(m[0]);
  }
  return tokens;
};

/** Strip a trailing comment (naive: first # outside quotes). */
const stripComment = (line: string): string => {
  let inQuote: string | null = null;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuote) {
      if (c === inQuote) inQuote = null;
    } else if (c === '"' || c === "'") {
      inQuote = c;
    } else if (c === '#') {
      return line.slice(0, i);
    }
  }
  return line;
};

/** Join `&`-continued lines, strip comments/blanks → logical statements. */
export const scriptStatements = (text: string): string[] => {
  const out: string[] = [];
  const rawLines = text.split(/\r?\n/);
  let buffer = '';
  for (const raw of rawLines) {
    const noComment = stripComment(raw).trimEnd();
    if (buffer === '' && noComment.trim() === '') continue;
    if (buffer === '') buffer = noComment.trim();
    else buffer += ' ' + noComment.trim();
    if (buffer.endsWith('&')) {
      buffer = buffer.slice(0, -1).trimEnd();
      continue;
    }
    if (buffer.trim()) out.push(buffer.trim());
    buffer = '';
  }
  if (buffer.trim()) out.push(buffer.trim());
  return out;
};

interface PatternVariant {
  tokens: string[];
  isSlot: boolean[];
  paramKeys: (string | null)[];
  absorbIndex: number;
}

interface Pattern {
  def: CommandDef;
  variants: PatternVariant[];     // minimal first, full second
  literalCount: number;
  enumKeys: Set<string>;          // params that are enums (for scoring)
}

const patternCache = new Map<string, Pattern | null>();

const buildVariant = (def: CommandDef, minimal: boolean): PatternVariant | null => {
  const params: Record<string, string> = {};
  def.params.forEach((pd, i) => {
    // Every non-flag param becomes a SLOT; enums are validated for scoring
    // in matchLine (so `pair_style hybrid/overlay` prefers the hybrid def).
    // Minimal variant: optional empty-default strings stay truly empty so
    // conditional tokens (`v.units && …`) vanish from the pattern.
    params[pd.key] =
      pd.type === 'flag'
        ? (pd.default ?? 'no')
        : minimal && (pd.default ?? '') === ''
          ? ''
          : slotOf(i);
  });

  let built: string[];
  try {
    built = def.build(params).filter(l => l.trim() !== '');
  } catch {
    return null;
  }
  if (built.length !== 1) return null; // multi-line builders are not import-matched

  const tokens = tokenizeLine(built[0]);
  const isSlot: boolean[] = [];
  const paramKeys: (string | null)[] = [];
  const enumKeys = new Set<string>();

  tokens.forEach(tok => {
    const slotMatch = tok.includes(SLOT);
    if (slotMatch) {
      const idx = parseInt(tok.slice(SLOT.length), 10);
      const pd: ParamDef | undefined = def.params[idx];
      isSlot.push(true);
      paramKeys.push(pd ? pd.key : null);
      if (pd?.type === 'enum') enumKeys.add(pd.key);
    } else {
      isSlot.push(false);
      paramKeys.push(null);
    }
  });

  // ID-flex: wildcard the token right after the command keyword. When the
  // def has an explicit ID param, the token maps back into it so region
  // names, fix IDs etc. survive the import.
  if (ID_FLEX.has(def.command) && tokens.length > 1 && !isSlot[1]) {
    isSlot[1] = true;
    const first = def.params[0];
    paramKeys[1] = first && ID_PARAM_KEYS.has(first.key) ? first.key : null;
  }

  // Trailing absorb: ensure the LAST param — when it is a string/text slot —
  // is present at the pattern tail even if the minimal build omitted it
  // (optional-empty). Without this, `velocity … loop geom` has nothing to
  // absorb the trailing keywords.
  const lastPd = def.params[def.params.length - 1];
  if (lastPd && (lastPd.type === 'string' || lastPd.type === 'text') && tokens.length > 0) {
    const endsWithIt =
      isSlot[tokens.length - 1] && paramKeys[tokens.length - 1] === lastPd.key;
    if (!endsWithIt) {
      tokens.push(slotOf(def.params.length - 1));
      isSlot.push(true);
      paramKeys.push(lastPd.key);
    }
  }

  // Trailing absorb: last param is a string/text slot → eats the rest.
  let absorbIndex = -1;
  const last = def.params[def.params.length - 1];
  if (last && (last.type === 'string' || last.type === 'text') && tokens.length > 0) {
    const lastIdx = tokens.length - 1;
    if (isSlot[lastIdx] && paramKeys[lastIdx] === last.key) absorbIndex = lastIdx;
  }

  return { tokens, isSlot, paramKeys, absorbIndex };
};

const buildPattern = (def: CommandDef): Pattern | null => {
  const cached = patternCache.get(def.id);
  if (cached !== undefined) return cached;

  const minimal = buildVariant(def, true);
  const full = buildVariant(def, false);
  const variants = [minimal, full].filter((v): v is PatternVariant => v !== null);
  // De-duplicate identical variants
  const unique: PatternVariant[] = variants.filter(
    (v, i) => variants.findIndex(o => o.tokens.join('\u0001') === v.tokens.join('\u0001')) === i,
  );
  if (unique.length === 0) {
    patternCache.set(def.id, null);
    return null;
  }

  const literalCount = Math.min(
    ...unique.map(v => v.tokens.filter((t, i) => !v.isSlot[i]).length),
  );
  const enumKeys = new Set<string>();
  def.params.forEach(pd => { if (pd.type === 'enum') enumKeys.add(pd.key); });

  const pattern: Pattern = { def, variants: unique, literalCount, enumKeys };
  patternCache.set(def.id, pattern);
  return pattern;
};

export interface LineMatch {
  def: CommandDef;
  params: Record<string, string>;
  score: number;
}

/** Match one statement's tokens against the catalog. */
export const matchLine = (tokens: string[]): LineMatch | null => {
  if (tokens.length === 0) return null;
  const keyword = tokens[0];

  let best: LineMatch | null = null;
  for (const def of ALL_COMMANDS) {
    if (def.command !== keyword) continue;
    const pat = buildPattern(def);
    if (!pat) continue;

    for (const variant of pat.variants) {
      const required = variant.absorbIndex >= 0 ? variant.absorbIndex : variant.tokens.length;
      if (tokens.length < required) continue;
      if (variant.absorbIndex < 0 && tokens.length !== variant.tokens.length) continue;

      let ok = true;
      let score = variant.tokens.filter((t, i) => !variant.isSlot[i]).length;
      const values: Record<string, string> = {};

      for (let t = 0; t < required && ok; t++) {
        if (variant.isSlot[t]) {
          const key = variant.paramKeys[t];
          if (key) {
            values[key] = tokens[t];
            if (pat.enumKeys.has(key)) {
              const pd = def.params.find(p => p.key === key);
              if (pd?.options?.some(o => o.value === tokens[t])) score += 2;
            }
          }
        } else if (variant.tokens[t] !== tokens[t]) {
          ok = false;
        }
      }
      if (!ok) continue;

      if (variant.absorbIndex >= 0) {
        const key = variant.paramKeys[variant.absorbIndex];
        if (key) values[key] = tokens.slice(variant.absorbIndex).join(' ');
      }

      if (!best || score > best.score) {
        best = { def, params: values, score };
      }
    }
  }
  return best;
};

let rawCounter = 1;

/** Import a full script into a ScriptModel. */
export const parseScript = (text: string, title = 'Imported script'): ImportResult => {
  const statements = scriptStatements(text);
  const steps: ScriptStep[] = [];
  let recognized = 0;

  for (const stmt of statements) {
    const tokens = tokenizeLine(stmt);
    const match = matchLine(tokens);
    if (match) {
      recognized += 1;
      steps.push({
        uid: `imp-${rawCounter++}`,
        defId: match.def.id,
        params: { ...defaultParams(match.def), ...match.params },
        enabled: true,
      });
    } else {
      steps.push({
        uid: `imp-${rawCounter++}`,
        defId: 'raw_line',
        params: { ...defaultParams(COMMAND_BY_ID.raw_line), line: stmt },
        enabled: true,
      });
    }
  }

  return {
    model: { title, steps },
    stats: { total: statements.length, recognized, raw: statements.length - recognized },
  };
};

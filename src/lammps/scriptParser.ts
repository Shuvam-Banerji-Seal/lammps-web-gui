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
 *      - Enum and flag params are also tried at each of their options, so
 *        tokens that build() emits only for one option (`create_atoms …
 *        region ID`, `region … side out`, `kspace_style none`) are matchable.
 *  3. A candidate is ACCEPTED only if rebuilding the def with the captured
 *     params reproduces the statement's tokens exactly — import → regenerate
 *     can therefore never change a command. Among accepted candidates the
 *     best score wins (ties: more non-empty captured params).
 *  4. Unmatched statements become `raw_line` steps — nothing is lost.
 */

import { mapUnquoted, splitCommands } from '../engine/script';
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

/**
 * Quote-aware tokenizer: a triple-, double- or single-quoted argument is one
 * token (quotes kept). docs.lammps.org/Commands_parse.html: "If you want
 * text with spaces to be treated as a single argument, it can be enclosed in
 * either single (') or double (") or triple (""") quotes."
 */
export const tokenizeLine = (line: string): string[] => {
  const tokens: string[] = [];
  const re = /"""[\s\S]*?"""|"([^"]*)"|'([^']*)'|(\S+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(line)) !== null) {
    tokens.push(m[0]);
  }
  return tokens;
};

/** A logical statement plus the 1-based source line it started on. */
export interface SourceStatement {
  text: string;
  line: number;
}

/**
 * Logical statements with the 1-based source line each started on, split by
 * one character scan over the whole text (src/engine/script.ts) that follows
 * docs.lammps.org/Commands_parse.html: '&' continues a line (also inside
 * quotes; "a comment after a trailing “&” character will prevent the command
 * from continuing"); '#' starts a comment only outside quotes; triple quotes
 * keep their line breaks and make "&" characters unnecessary.
 */
export const scriptStatementsDetailed = (text: string): SourceStatement[] =>
  splitCommands(text.replace(/\r\n?/g, '\n')).map((c) => ({
    // whitespace outside quotes only separates words, so normalise it
    text: mapUnquoted(c.text, (part) => part.replace(/\s+/g, ' ')).trim(),
    line: c.line,
  }));

/** Join `&`-continued lines, strip comments/blanks → logical statements. */
export const scriptStatements = (text: string): string[] =>
  scriptStatementsDetailed(text).map(s => s.text);

interface PatternVariant {
  tokens: string[];
  isSlot: boolean[];
  paramKeys: (string | null)[];
  /** Literal text around a slot inside its token (e.g. the quotes of "SLOT"). */
  affix: ([string, string] | null)[];
  /** Index of a final string/text slot that absorbs the remaining tokens, or -1. */
  absorbIndex: number;
  /** Enum / flag values baked into this variant (literal in its tokens). */
  fixed: Record<string, string>;
}

interface Pattern {
  def: CommandDef;
  variants: PatternVariant[];
  enumKeys: Set<string>;          // params that are enums (for scoring)
}

const patternCache = new Map<string, Pattern | null>();

/** Most option combinations tried per def (beyond it only the first two enum/flag params vary). */
const MAX_COMBOS = 64;

const isTextParam = (pd: ParamDef | undefined) => !!pd && (pd.type === 'string' || pd.type === 'text');

const buildVariant = (def: CommandDef, minimal: boolean, fixed: Record<string, string>): PatternVariant | null => {
  const params: Record<string, string> = {};
  def.params.forEach((pd, i) => {
    // Every non-flag param becomes a SLOT unless this variant fixes it to
    // one of its options; enums are validated for scoring in matchLine.
    // Minimal variant: optional empty-default strings stay truly empty so
    // conditional tokens (`v.units && …`) vanish from the pattern.
    params[pd.key] =
      pd.key in fixed
        ? fixed[pd.key]
        : pd.type === 'flag'
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
  const affix: ([string, string] | null)[] = [];

  tokens.forEach(tok => {
    const at = tok.indexOf(SLOT);
    if (at >= 0) {
      const m = /^\d+/.exec(tok.slice(at + SLOT.length));
      const idx = m ? parseInt(m[0], 10) : -1;
      const pd: ParamDef | undefined = def.params[idx];
      isSlot.push(true);
      paramKeys.push(pd ? pd.key : null);
      const prefix = tok.slice(0, at);
      const suffix = m ? tok.slice(at + SLOT.length + m[0].length) : '';
      affix.push(prefix || suffix ? [prefix, suffix] : null);
    } else {
      isSlot.push(false);
      paramKeys.push(null);
      affix.push(null);
    }
  });

  // ID-flex: wildcard the token right after the command keyword. When the
  // def has an explicit ID param, the token maps back into it so region
  // names, fix IDs etc. survive the import.
  if (ID_FLEX.has(def.command) && tokens.length > 1 && !isSlot[1]) {
    isSlot[1] = true;
    affix[1] = null;
    const first = def.params[0];
    paramKeys[1] = first && ID_PARAM_KEYS.has(first.key) ? first.key : null;
  }

  // Trailing absorb slot: when the LAST param is a string/text that has no
  // slot ANYWHERE in this pattern (the minimal build omitted it), append one
  // so trailing keywords (`velocity … loop geom`) have somewhere to go. A
  // param that already has a slot must not get a second one: the absorb
  // would overwrite the value captured for it (`fix 1 all nve` lost `all`).
  const lastPd = def.params[def.params.length - 1];
  if (isTextParam(lastPd) && !(lastPd.key in fixed) && !paramKeys.includes(lastPd.key) && tokens.length > 0) {
    tokens.push(slotOf(def.params.length - 1));
    isSlot.push(true);
    paramKeys.push(lastPd.key);
    affix.push(null);
  }

  // A final string/text slot eats the rest of the line (multi-word values
  // such as `processors * * *` or custom special_bonds weights).
  let absorbIndex = -1;
  const li = tokens.length - 1;
  if (li >= 0 && isSlot[li] && paramKeys[li] && !affix[li]) {
    const pd = def.params.find(p => p.key === paramKeys[li]);
    if (isTextParam(pd)) absorbIndex = li;
  }

  return { tokens, isSlot, paramKeys, affix, absorbIndex, fixed };
};

/** Option combinations of the enum and flag params, defaults first. */
const optionCombos = (def: CommandDef): Record<string, string>[] => {
  const vary = def.params
    .filter(pd => (pd.type === 'enum' && pd.options?.length) || pd.type === 'flag')
    .map(pd => {
      const values = pd.type === 'flag' ? ['no', 'yes'] : pd.options!.map(o => o.value);
      const d = pd.default ?? values[0];
      return { key: pd.key, values: [d, ...values.filter(v => v !== d)] };
    });
  const size = vary.reduce((n, v) => n * v.values.length, 1);
  const used = size <= MAX_COMBOS ? vary : vary.slice(0, 2);
  let combos: Record<string, string>[] = [{}];
  for (const v of used) {
    combos = combos.flatMap(c => v.values.map(val => ({ ...c, [v.key]: val })));
  }
  return combos.slice(0, MAX_COMBOS);
};

const buildPattern = (def: CommandDef): Pattern | null => {
  const cached = patternCache.get(def.id);
  if (cached !== undefined) return cached;

  const variants: PatternVariant[] = [];
  const seen = new Set<string>();
  const add = (v: PatternVariant | null) => {
    if (!v) return;
    const key = v.tokens.join('\u0001') + '\u0002' + v.absorbIndex;
    if (seen.has(key)) return;
    seen.add(key);
    variants.push(v);
  };
  add(buildVariant(def, true, {}));
  add(buildVariant(def, false, {}));
  for (const combo of optionCombos(def)) {
    add(buildVariant(def, true, combo));
    add(buildVariant(def, false, combo));
  }
  if (variants.length === 0) {
    patternCache.set(def.id, null);
    return null;
  }

  const enumKeys = new Set<string>();
  def.params.forEach(pd => { if (pd.type === 'enum') enumKeys.add(pd.key); });

  const pattern: Pattern = { def, variants, enumKeys };
  patternCache.set(def.id, pattern);
  return pattern;
};

export interface LineMatch {
  def: CommandDef;
  params: Record<string, string>;
  score: number;
}

/** True when the def rebuilt with `params` gives exactly these tokens. */
const rebuildsTo = (def: CommandDef, params: Record<string, string>, tokens: string[]): boolean => {
  let built: string[];
  try {
    built = def.build({ ...defaultParams(def), ...params }).filter(l => l.trim() !== '');
  } catch {
    return false;
  }
  return built.length === 1 && tokenizeLine(built[0]).join('\u0001') === tokens.join('\u0001');
};

/** Match one statement's tokens against the catalog. */
export const matchLine = (tokens: string[]): LineMatch | null => {
  if (tokens.length === 0) return null;
  const keyword = tokens[0];

  let best: (LineMatch & { filled: number }) | null = null;
  for (const def of ALL_COMMANDS) {
    if (def.command !== keyword) continue;
    const pat = buildPattern(def);
    if (!pat) continue;

    for (const variant of pat.variants) {
      const required = variant.absorbIndex >= 0 ? variant.absorbIndex : variant.tokens.length;
      if (tokens.length < required) continue;
      if (variant.absorbIndex < 0 && tokens.length !== variant.tokens.length) continue;

      let ok = true;
      let score = variant.tokens.filter((t, i) => !variant.isSlot[i]).length + Object.keys(variant.fixed).length;
      const values: Record<string, string> = { ...variant.fixed };

      for (let t = 0; t < required && ok; t++) {
        if (variant.isSlot[t]) {
          const key = variant.paramKeys[t];
          let tok = tokens[t];
          const af = variant.affix[t];
          if (af) {
            // a slot written inside quotes etc.: the input must carry the same
            // surrounding text, and only the inside is the value
            if (!tok.startsWith(af[0]) || !tok.endsWith(af[1]) || tok.length < af[0].length + af[1].length) { ok = false; break; }
            tok = tok.slice(af[0].length, tok.length - af[1].length);
          }
          if (key) {
            values[key] = tok;
            if (pat.enumKeys.has(key)) {
              const pd = def.params.find(p => p.key === key);
              if (pd?.options?.some(o => o.value === tok)) score += 2;
            }
          }
        } else if (variant.tokens[t] !== tokens[t]) {
          ok = false;
        }
      }
      if (!ok) continue;

      if (variant.absorbIndex >= 0) {
        const key = variant.paramKeys[variant.absorbIndex];
        const rest = tokens.slice(variant.absorbIndex).join(' ');
        if (key) {
          const earlier = variant.paramKeys.indexOf(key);
          if (earlier >= 0 && earlier < variant.absorbIndex) {
            // the key already has its own slot: an absorb must not overwrite it
            if (rest !== '' && rest !== values[key]) continue;
          } else {
            values[key] = rest;   // may be '' — an explicitly empty tail
          }
        }
      }

      // Accept only matches that regenerate the statement exactly.
      if (!rebuildsTo(def, values, tokens)) continue;

      const filled = Object.values(values).filter(v => v !== '').length;
      if (!best || score > best.score || (score === best.score && filled > best.filled)) {
        best = { def, params: values, score, filled };
      }
    }
  }
  return best ? { def: best.def, params: best.params, score: best.score } : null;
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

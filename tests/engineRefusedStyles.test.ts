import { describe, expect, it } from 'vitest';
import { REFUSED_STYLES } from '../src/engine/refusedStyles';
import { registeredNamesOf, styleNames } from '../src/engine/styles';
import type { StyleKind } from '../src/engine/lazyStyles';

const LIST_OF: Record<Exclude<StyleKind, 'HOOKS' | 'COMMANDS'>, string> = {
  PAIRS: 'pair_style', BONDS: 'bond_style', ANGLES: 'angle_style', DIHEDRALS: 'dihedral_style',
  IMPROPERS: 'improper_style', KSPACES: 'kspace_style', FIXES: 'fix', COMPUTES: 'compute',
};

describe('refusal-only styles', () => {
  it('are still registered (so using them gives their specific error) but not listed as supported', () => {
    const names = styleNames();
    for (const [kind, list] of Object.entries(REFUSED_STYLES) as [Exclude<StyleKind, 'HOOKS' | 'COMMANDS'>, readonly string[]][]) {
      for (const n of list) {
        expect(registeredNamesOf(kind), `${kind} ${n} registered`).toContain(n);
        expect(names[LIST_OF[kind]], `${kind} ${n} not listed`).not.toContain(n);
      }
    }
  });
});

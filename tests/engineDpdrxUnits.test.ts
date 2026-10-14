import { describe, expect, it } from 'vitest';
import { unorderedSetOrder } from '../src/engine-gpl/dpdrx/unordered_order';
import { powint } from '../src/engine-gpl/dpdrx/fix_rx';
import { RxTableReader } from '../src/engine-gpl/dpdrx/rx_table_reader';

/* Unit checks of the DPD-REACT rx port (src/engine-gpl/dpdrx): the species order native's unordered_set gives, the integer
 * power of fix rx's sparse rates, and the keyword-section reader of the rx tables. The oracle cases cover the physics. */

describe('unordered_set iteration order (libstdc++, GCC 11)', () => {
  it('orders the example species as std::unordered_set<std::string> does', () => {
    // inserted in the order the example's reaction file first names them
    expect(unorderedSetOrder(['rdx', 'hcn', 'no2', 'no', 'h2o', 'n2', 'h2', 'co', 'co2']).join(' ')).toBe('co2 co n2 no h2o no2 hcn h2 rdx');
  });

  it('ignores repeated names', () => {
    expect(unorderedSetOrder(['a', 'b', 'a'])).toEqual(unorderedSetOrder(['a', 'b']));
  });
});

describe('powint (math_special.h)', () => {
  it('computes integer powers, and the reciprocal for negative exponents', () => {
    expect(powint(2, 10)).toBe(1024);
    expect(powint(2, -2)).toBe(0.25);
    expect(powint(0, 3)).toBe(0);
    expect(powint(5, 0)).toBe(1);
  });
});

describe('rx table reader (rx_table_file_reader.cpp)', () => {
  const text = '# comment\n\nKEYWORD\nN 2 rdx h2\n1 0.0 1.5 2.5\n2 10.0 3.5 4.5\n';

  it('reads N, the species words and the entries', () => {
    const r = new RxTableReader(text, 'KEYWORD', 'table.eos', 'eos/table/rx');
    expect(r.N).toBe(2);
    expect(r.params).toEqual(['rdx', 'h2']);
    const rows: number[][] = [];
    r.readTable((i, v) => {
      v.int();
      rows.push([v.double(), v.double(), v.double()]);
      expect(i).toBeGreaterThanOrEqual(0);
    });
    expect(rows).toEqual([[0, 1.5, 2.5], [10, 3.5, 4.5]]);
  });

  it('names a missing keyword', () => {
    expect(() => new RxTableReader(text, 'MISSING', 'table.eos', 'eos/table/rx')).toThrow(/Did not find keyword 'MISSING' in 'table.eos'/);
  });
});

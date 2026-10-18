import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * The extensive flag of global arrays (Compute.extarray, Fix.extarray): thermo_modify norm yes divides an
 * extensive array element by the atom count (src/thermo.cpp at b332f28), and fix vector, fix ave/time and
 * fix ave/moments refuse inputs whose flags differ, as native does. Native parity of the normalized values is
 * tests/oracle/w238ext_norm.
 */

/** Runs an input and returns the first error message and the thermo rows. */
async function run(script: string) {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  try {
    await session.execute(script);
  } catch {
    // the failure is reported as an error event below
  }
  const err = events.find((e) => e.kind === 'error');
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as ThermoRow);
  return { error: err && 'message' in err ? String(err.message) : undefined, rows };
}

/** 32-atom lj crystal (norm yes by default in lj units) with an extensive (ke, temp) and an intensive (pressure) source. */
const BASE = `units lj
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 4928 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
compute ke all ke
compute t all temp
compute p all pressure t
variable tt equal c_t
fix 1 all nve
`;

describe('extensive global arrays', () => {
  it('refuses array outputs whose inputs mix extensive and intensive values (native messages)', async () => {
    const cases: [string, RegExp][] = [
      ['fix a all ave/time 2 5 10 c_t c_p mode vector', /Fix ave\/time cannot set output array intensive\/extensive from these inputs/],
      ['fix a all ave/moments 2 5 10 c_ke v_tt mean', /Fix ave\/moments cannot set output array intensive\/extensive from these inputs/],
      ['fix a all vector 5 c_ke v_tt', /Fix vector cannot set output array intensive\/extensive from these inputs/],
    ];
    for (const [line, msg] of cases) expect((await run(`${BASE}${line}\nrun 0\n`)).error, line).toMatch(msg);
  });

  it('divides an extensive fix vector element by N with norm yes, and leaves it with norm no', async () => {
    const script = (norm: string) => `${BASE}fix v all vector 1 c_ke c_ke
run 1
thermo_style custom step c_ke f_v[1][1]
thermo_modify norm ${norm}
run 0
`;
    const yes = (await run(script('yes'))).rows.at(-1)!;
    const no = (await run(script('no'))).rows.at(-1)!;
    expect(no['f_v[1][1]'] as number).toBeCloseTo((yes['f_v[1][1]'] as number) * 32, 12);
    expect(no.c_ke as number).toBeCloseTo((yes.c_ke as number) * 32, 12);
  });
});

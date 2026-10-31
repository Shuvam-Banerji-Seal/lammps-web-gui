import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * write_data noinit (src/write_data.cpp at b332f28: "noinit is a hidden arg, only used by -r command-line switch"):
 * no system init and no remap of the atoms into the box; with pair ij the mixed coefficients are computed with the
 * warning "Computing missing pair coeffs from mixing". The fixtures are native's files for the same input (the first
 * line, which names the LAMMPS version and the timestep, removed).
 */

const INPUT = `units lj
lattice fcc 0.8442
region b block 0 2 0 2 0 2
create_box 2 b
create_atoms 1 box
mass * 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
pair_coeff 2 2 1.5 1.1
set atom 1 x 4.0
set atom 2 y -0.5
write_data a_init.data
set atom 1 x 4.0
set atom 2 y -0.5
write_data a_noinit.data noinit
write_data a_noinit_ij.data noinit pair ij
`;

const fixture = (f: string) => readFileSync(join(__dirname, 'fixtures', 'write_data_noinit', f), 'utf8');
const body = (text: string | undefined) => (text ?? '').split('\n').slice(1).join('\n');

describe('write_data noinit', () => {
  it('writes the atoms where they are and computes mixed pair coefficients for pair ij, as native', async () => {
    const files = new Map<string, string>();
    const logs: string[] = [];
    const s = new Session({
      emit: (e: EngineEvent) => { if (e.kind === 'log') logs.push(e.text); },
      writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
    });
    await s.execute(INPUT);
    // Measured with native LAMMPS: x 4.0 (box length 3.359) stays 4 and y -0.5 stays -0.5, images unchanged
    expect(body(files.get('a_noinit.data'))).toBe(fixture('noinit.data'));
    expect(body(files.get('a_noinit_ij.data'))).toBe(fixture('noinit_ij.data'));
    expect(logs.filter((l) => l.startsWith('WARNING'))).toEqual(['WARNING: Computing missing pair coeffs from mixing']);
    // without noinit the atoms are remapped into the box
    expect(files.get('a_init.data')).toContain('\n1 1 0.6408076172349855 0 0 1 0 0\n');
  });
});

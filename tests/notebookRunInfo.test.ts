import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { thermoUnit, unitsCaption } from '../src/lammps/thermoUnits';

const events = async (script: string): Promise<EngineEvent[]> => {
  const out: EngineEvent[] = [];
  await new Session({ emit: (e) => out.push(e) }).execute(script);
  return out;
};

const BOX = (units: string) => `units ${units}
lattice fcc ${units === 'lj' ? '0.8442' : '5.3'}
region b block 0 2 0 2 0 2
create_box 1 b
create_atoms 1 box
mass 1 ${units === 'lj' ? '1.0' : '39.95'}
pair_style lj/cut ${units === 'lj' ? '2.5' : '8.5'}
pair_coeff 1 1 ${units === 'lj' ? '1.0 1.0' : '0.238 3.405'}
fix 1 all nve
`;

describe('notebook run information from the engine', () => {
  it('each run announces its first and last step (progress bar), after its thermo header', async () => {
    const ev = await events(`${BOX('lj')}run 10\nrun 15\n`);
    const runs = ev.filter((e) => e.kind === 'run');
    expect(runs).toEqual([{ kind: 'run', from: 0, to: 10 }, { kind: 'run', from: 10, to: 25 }]);
    const i = ev.findIndex((e) => e.kind === 'run');
    expect(ev[i - 1].kind).toBe('thermo-header');
  });

  it('the thermo header carries the units style', async () => {
    const real = (await events(`${BOX('real')}run 0\n`)).find((e) => e.kind === 'thermo-header');
    expect(real && 'units' in real ? real.units : null).toBe('real');
  });

  it('units of thermo columns follow docs.lammps.org/units.html', () => {
    expect(thermoUnit('real', 'press')).toBe('atmospheres');
    expect(thermoUnit('metal', 'pe')).toBe('eV');
    expect(thermoUnit('metal', 'temp')).toBe('Kelvin');
    expect(thermoUnit('real', 'step')).toBeNull();
    expect(thermoUnit('lj', 'press')).toBeNull();
    expect(unitsCaption('lj')).toMatch(/unitless/);
    expect(unitsCaption('metal')).toBe('units metal: energy eV, temperature Kelvin, pressure bars, distance Angstroms, time picoseconds');
  });
});

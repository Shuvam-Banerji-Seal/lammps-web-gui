import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { Session } from '../src/engine/interpreter';

/*
 * The restart output of prd and tad (prd.html: "a frequency of 10 means write a restart file every 10th uncorrelated
 * event"; tad.html: "The restart file is generated after an event is executed"). The file names are those native LAMMPS
 * writes for the same inputs (measured with native LAMMPS, 3 partitions): prd.120.rst and prd.180.rst with pa.rst and
 * pb.rst (every restart file is written when one is due; the event at the last step gets none), and tad.40.rst and
 * tad.60.rst. Their dumps are compared with native in tests/oracle/w308prd_dump.in and w309tad_dump.in.
 */

const inputOf = (name: string) => readFileSync(`tests/oracle/${name}.in`, 'utf8').split('\n').filter((l) => !l.startsWith('#')).join('\n');

const written = async (text: string): Promise<string[]> => {
  const names: string[] = [];
  const session = new Session({ emit: () => undefined, writeFile: (name: string) => { names.push(name); } }, undefined, 0, 3);
  session.addFile('Si.sw', readFileSync('third_party/lammps/potentials/Si.sw', 'utf8'));
  await session.execute(text);
  return names;
};

describe('prd and tad restart files', () => {
  it('prd writes every restart file after each N-th uncorrelated event and its correlated time', async () => {
    const names = await written(inputOf('w195prd_si').replace(/^prd 200/m, 'restart 1 prd.*.rst\nrestart 2 pa.rst pb.rst\nprd 200'));
    expect(names.filter((n) => n.endsWith('.rst'))).toEqual(['prd.120.rst', 'pa.rst', 'prd.180.rst', 'pb.rst']);
  }, 120_000);

  it('tad writes a restart file after each N-th executed event', async () => {
    const names = await written(inputOf('w195tad_si').replace(/^tad 200/m, 'restart 1 tad.*.rst\ntad 200'));
    expect(names.filter((n) => n.endsWith('.rst'))).toEqual(['tad.40.rst', 'tad.60.rst']);
  }, 120_000);
});

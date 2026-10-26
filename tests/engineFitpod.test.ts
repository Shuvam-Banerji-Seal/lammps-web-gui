import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * fitpod (src/engine-gpl/pod/fitpod.ts, ported from src/ML-POD/fitpod_command.cpp): its argument and file checks, and
 * the descriptor files of compute_pod_descriptors. Measured with native LAMMPS: each input below gives the message
 * expected (the first line of the engine's message; a missing file adds the engine's hint on a second line). The fit
 * itself is compared with native output by the oracle cases fitpod_ta, fitpod_ta4 and fitpod_ta_coeff.
 */

const TA = join(__dirname, '..', 'third_party', 'lammps', 'examples', 'PACKAGES', 'pod', 'Ta');

const session = (files: Record<string, string>, written?: Map<string, string>): Session => {
  const s = new Session({ emit: () => {}, writeFile: (n: string, t: string, ap: boolean) => { written?.set(n, (ap ? written.get(n) ?? '' : '') + t); } } as never);
  for (const [n, t] of Object.entries(files)) s.addFile(n, t);
  return s;
};

const param = readFileSync(join(TA, 'Ta_param.pod'), 'utf8');

const fails = async (files: Record<string, string>, cmd: string, message: string): Promise<void> => {
  const err = await session(files).execute(`${cmd}\n`).then(() => null, (e: Error) => e.message);
  expect(err?.split('\n')[0]).toBe(`line 1: ${message}`);
};

describe('fitpod: native checks', () => {
  it('arguments and files', async () => {
    await fails({}, 'fitpod', 'Illegal fitpod command: missing argument(s)');
    await fails({}, 'fitpod Ta_param.pod', 'Illegal fitpod command: missing argument(s)');
    await fails({}, 'fitpod missing.pod nodata.pod', 'Cannot open POD coefficient file missing.pod: ');
    await fails({ 'Ta_param.pod': param }, 'fitpod Ta_param.pod missing_data.pod', 'Cannot open training data file missing_data.pod: ');
    await fails({ 'Ta_param.pod': param, 'd.pod': 'path_to_training_data_set "X"\n' }, 'fitpod Ta_param.pod d.pod missing_coeff.pod', 'Cannot open model coefficient file missing_coeff.pod: ');
  });

  it('the data file', async () => {
    await fails({ 'Ta_param.pod': param, 'd.pod': 'file_format extxyz\npath_to_training_data_set "NOPE"\n' }, 'fitpod Ta_param.pod d.pod',
      'Cannot fit potential without data files. The data paths may not be valid. Please check the data paths in the POD data file.');
    await fails({ 'Ta_param.pod': param, 'd.pod': 'file_format extxyz extra\n' }, 'fitpod Ta_param.pod d.pod', 'Improper POD data file.');
    await fails({ 'Ta_param.pod': param, 'd.pod': 'path_to_training_data_set "X"\n' }, 'fitpod Ta_param.pod d.pod', 'data set is not found');
  });

  it('compute_pod_descriptors 2 writes each configuration\'s base and global descriptors, with derivatives', async () => {
    const files: Record<string, string> = { 'Ta_param.pod': param, 'd.pod': 'path_to_training_data_set "XYZ"\ncompute_pod_descriptors 2\n' };
    files['XYZ/Volume_BCC.xyz'] = readFileSync(join(TA, 'XYZ', 'Volume_BCC.xyz'), 'utf8');
    const written = new Map<string, string>();
    await session(files, written).execute('fitpod Ta_param.pod d.pod\n');
    const doubles = (t: string) => new Float64Array(Uint8Array.from(t, (c) => c.charCodeAt(0)).buffer);
    // 21 configurations of 2 atoms; 32 global descriptors (one-body, 6 two-body, 25 three-body), Mdesc 31
    expect([...written.keys()].filter((n) => n.startsWith('XYZ/globaldescriptors_config')).length).toBe(21);
    const g = doubles(written.get('XYZ/globaldescriptors_config1.bin')!);
    expect([g[0], g[1], g.length]).toEqual([2, 32, 2 + 32 + 3 * 2 * 32]);
    expect(g[2]).toBe(2); // the one-body descriptor counts the atoms
    const b = doubles(written.get('XYZ/basedescriptors_config1.bin')!);
    expect([b[0], b[1], b.length]).toEqual([2, 31, 2 + 2 * 31]);
  });
});


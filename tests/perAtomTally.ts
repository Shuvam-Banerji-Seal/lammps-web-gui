import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * The per-atom tally check of an oracle case: after the case's input, the sum of compute pe/atom must equal thermo pe
 * (A), and the sum of compute stress/atom NULL virial must equal -(compute pressure NULL virial) times the volume, per
 * component (B). A style that tallies only the global energy or virial fails it while every thermo value is right. Native
 * LAMMPS passes it for the cases the tests list (each family's test names its cases); tail corrections and fixes whose
 * energy pe/atom does not include break (A) in native too, so such cases are not listed.
 */

const CASES = join(__dirname, 'oracle');
const POTENTIALS = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const EXAMPLES = join(__dirname, '..', 'third_party', 'lammps', 'examples');

const directive = (text: string, key: string): string[] => {
  const m = text.match(new RegExp(`^# ${key}:(.*)$`, 'm'));
  return m ? m[1].trim().split(/\s+/).filter(Boolean) : [];
};

const readInput = (path: string): string => {
  const buf = readFileSync(path);
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(buf);
  } catch {
    return buf.toString('latin1');
  }
};

const CHECK = `
compute zzpa all pe/atom
compute zzsa all stress/atom NULL virial
compute zzpv all pressure NULL virial
compute zzrp all reduce sum c_zzpa
compute zzrs all reduce sum c_zzsa[1] c_zzsa[2] c_zzsa[3] c_zzsa[4] c_zzsa[5] c_zzsa[6]
thermo_style custom step pe c_zzrp vol c_zzpv[1] c_zzpv[2] c_zzpv[3] c_zzpv[4] c_zzpv[5] c_zzpv[6] c_zzrs[1] c_zzrs[2] c_zzrs[3] c_zzrs[4] c_zzrs[5] c_zzrs[6]
thermo_modify format float %.15g norm no
run 0
`;

/** The relative mismatches of checks (A) and (B) after running the oracle case `name`. */
export const perAtomTally = async (name: string): Promise<{ relA: number; relB: number }> => {
  const text = readFileSync(join(CASES, `${name}.in`), 'utf8');
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const f of directive(text, 'oracle-inputs')) session.addFile(f, readInput(join(CASES, f)));
  for (const f of directive(text, 'oracle-potentials')) session.addFile(f, readFileSync(join(POTENTIALS, f), 'utf8'));
  for (const f of directive(text, 'oracle-examples')) session.addFile(f.split('/').pop()!, readInput(join(EXAMPLES, f)));
  await session.execute(`${text}\n${CHECK}`);
  const err = events.find((e) => e.kind === 'error');
  if (err && 'message' in err) throw new Error(`${name}: ${err.message}`);
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo');
  const r = rows[rows.length - 1].row as Record<string, number>;
  const pe = r.pe, sumPe = r.c_zzrp, vol = r.vol;
  const relA = Math.abs(sumPe - pe) / Math.max(Math.abs(pe), 1e-300);
  const want = [1, 2, 3, 4, 5, 6].map((k) => -r[`c_zzpv[${k}]`] * vol);
  const got = [1, 2, 3, 4, 5, 6].map((k) => r[`c_zzrs[${k}]`]);
  let scale = 0;
  for (const w of want) scale = Math.max(scale, Math.abs(w));
  let relB = 0;
  for (let k = 0; k < 6; k++) relB = Math.max(relB, Math.abs(got[k] - want[k]) / Math.max(scale, 1e-300));
  return { relA, relB };
};

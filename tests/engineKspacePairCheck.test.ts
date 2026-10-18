import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * KSpace::pair_check (src/kspace.cpp at b332f28): a solver style is refused when the pair style does not carry the
 * flags it needs, or when the dispersion or TIP4P flags differ. Each case below was run with the native binary
 * (black box, the same small charged lattice in every case: units real, lattice sc 3.0, a 3x3x3 box of 27 atoms,
 * half of them type 2 with charge -1, displaced randomly). The accepted cases finish the run 0; the refused ones stop
 * with the message given in the comment. The engine must agree with each.
 */

const run = async (text: string): Promise<string | null> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  try {
    await session.execute(text);
  } catch (e) {
    return e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  return err && 'message' in err ? String(err.message) : null;
};

const preamble = (atom: string) => `units real
atom_style ${atom}
lattice sc 3.0
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.5 4321
set type 1 charge 1.0
set type 2 charge -1.0
mass * 20.0
displace_atoms all random 0.2 0.2 0.2 777
`;

const DIPOLE_SETUP = 'set group all charge 0.0\nset group all diameter 1.0\nset group all dipole/random 777 1.0\n';
const PP = 'kspace_modify gewald 0.3 mesh 10 10 10';
const PPD = 'kspace_modify gewald 0.3 mesh 10 10 10 gewald/disp 0.3 mesh/disp 10 10 10';
const EW = 'kspace_modify gewald 0.3';

interface Case {
  /** Pair style line, pair_coeff lines and the kspace line; the modifier line follows. */
  atom: 'charge' | 'dipole';
  pair: string;
  coeff: string;
  kspace: string;
  modify: string;
  /** Native message ("KSpace style ... is incompatible ..."), or null when native accepts the run. */
  native: string | null;
}

const cases: Array<[string, Case]> = [
  // Measured with native LAMMPS: ERROR: KSpace style esp is incompatible with Pair style lj/long/coul/long (esp has no
  // dispersion term; native KSpace::pair_check from esp.cpp init)
  ['lj/long/coul/long long long + esp', { atom: 'charge', pair: 'lj/long/coul/long long long 7.0', coeff: 'pair_coeff * * 0.1 2.5', kspace: 'esp 1e-5', modify: '', native: 'KSpace style esp is incompatible with Pair style lj/long/coul/long' }],
  // Measured with native LAMMPS: lj/cut/coul/long 8.0 with ewald 1e-5 runs (Total wall time) and so does pppm 1e-5
  ['lj/cut/coul/long + ewald', { atom: 'charge', pair: 'lj/cut/coul/long 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'ewald 1e-5', modify: EW, native: null }],
  ['lj/cut/coul/long + pppm', { atom: 'charge', pair: 'lj/cut/coul/long 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'pppm 1e-5', modify: PP, native: null }],
  // Measured with native LAMMPS: ERROR: KSpace style msm is incompatible with Pair style lj/cut/coul/long (src/kspace.cpp:215)
  ['lj/cut/coul/long + msm', { atom: 'charge', pair: 'lj/cut/coul/long 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'msm 1e-5', modify: '', native: 'KSpace style msm is incompatible with Pair style lj/cut/coul/long' }],
  // Measured with native LAMMPS: ERROR: KSpace style pppm is incompatible with Pair style lj/cut/coul/msm
  ['lj/cut/coul/msm + pppm', { atom: 'charge', pair: 'lj/cut/coul/msm 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'pppm 1e-5', modify: PP, native: 'KSpace style pppm is incompatible with Pair style lj/cut/coul/msm' }],
  // Measured with native LAMMPS: ERROR: KSpace style ewald is incompatible with Pair style lj/cut/coul/msm
  ['lj/cut/coul/msm + ewald', { atom: 'charge', pair: 'lj/cut/coul/msm 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'ewald 1e-5', modify: EW, native: 'KSpace style ewald is incompatible with Pair style lj/cut/coul/msm' }],
  // Measured with native LAMMPS: ERROR: KSpace style pppm is incompatible with Pair style lj/cut/coul/cut (the task's cut-off case)
  ['lj/cut/coul/cut + pppm', { atom: 'charge', pair: 'lj/cut/coul/cut 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'pppm 1e-5', modify: PP, native: 'KSpace style pppm is incompatible with Pair style lj/cut/coul/cut' }],
  // Measured with native LAMMPS: ERROR: KSpace style ewald is incompatible with Pair style lj/cut/coul/debye
  ['lj/cut/coul/debye + ewald', { atom: 'charge', pair: 'lj/cut/coul/debye 0.2 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'ewald 1e-5', modify: EW, native: 'KSpace style ewald is incompatible with Pair style lj/cut/coul/debye' }],
  // Measured with native LAMMPS: ERROR: KSpace style pppm is incompatible with Pair style coul/dsf
  ['coul/dsf + pppm', { atom: 'charge', pair: 'coul/dsf 0.2 8.0', coeff: 'pair_coeff * *', kspace: 'pppm 1e-5', modify: PP, native: 'KSpace style pppm is incompatible with Pair style coul/dsf' }],
  // Measured with native LAMMPS: ERROR: KSpace style ewald is incompatible with Pair style lj/cut
  ['lj/cut + ewald', { atom: 'charge', pair: 'lj/cut 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'ewald 1e-5', modify: EW, native: 'KSpace style ewald is incompatible with Pair style lj/cut' }],
  // Measured with native LAMMPS: ERROR: KSpace style pppm is incompatible with Pair style lj/cut
  ['lj/cut + pppm', { atom: 'charge', pair: 'lj/cut 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'pppm 1e-5', modify: PP, native: 'KSpace style pppm is incompatible with Pair style lj/cut' }],
  // Measured with native LAMMPS: runs with ewald 1e-5 (no error)
  ['coul/long + ewald', { atom: 'charge', pair: 'coul/long 8.0', coeff: 'pair_coeff * *', kspace: 'ewald 1e-5', modify: EW, native: null }],
  // Measured with native LAMMPS: ERROR: KSpace style pppm/disp is incompatible with Pair style coul/long
  ['coul/long + pppm/disp', { atom: 'charge', pair: 'coul/long 8.0', coeff: 'pair_coeff * *', kspace: 'pppm/disp 1e-5', modify: PPD, native: 'KSpace style pppm/disp is incompatible with Pair style coul/long' }],
  // Measured with native LAMMPS: runs with pppm/disp 1e-5 (gewald/disp and mesh/disp set)
  ['lj/long/coul/long long long + pppm/disp', { atom: 'charge', pair: 'lj/long/coul/long long long 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'pppm/disp 1e-5', modify: PPD, native: null }],
  // Measured with native LAMMPS: runs with ewald/disp 1e-5
  ['lj/long/coul/long long long + ewald/disp', { atom: 'charge', pair: 'lj/long/coul/long long long 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'ewald/disp 1e-5', modify: EW, native: null }],
  // Measured with native LAMMPS: ERROR: KSpace style ewald is incompatible with Pair style lj/long/coul/long
  ['lj/long/coul/long long long + ewald', { atom: 'charge', pair: 'lj/long/coul/long long long 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'ewald 1e-5', modify: EW, native: 'KSpace style ewald is incompatible with Pair style lj/long/coul/long' }],
  // Measured with native LAMMPS: ERROR: KSpace style pppm/disp is incompatible with Pair style lj/long/coul/long (flag_lj cut)
  ['lj/long/coul/long cut long + pppm/disp', { atom: 'charge', pair: 'lj/long/coul/long cut long 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'pppm/disp 1e-5', modify: PPD, native: 'KSpace style pppm/disp is incompatible with Pair style lj/long/coul/long' }],
  // Measured with native LAMMPS: runs with ewald/disp 1e-5 on lj/cut/coul/long (no dispersion or dipole term)
  ['lj/cut/coul/long + ewald/disp', { atom: 'charge', pair: 'lj/cut/coul/long 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'ewald/disp 1e-5', modify: EW, native: null }],
  // Measured with native LAMMPS: runs with pppm/disp 1e-5 on buck/long/coul/long long long
  // Measured with native LAMMPS: runs with pppm 1e-5 on buck/coul/long
  ['buck/coul/long + pppm', { atom: 'charge', pair: 'buck/coul/long 8.0', coeff: 'pair_coeff * * 1000.0 0.3 10.0', kspace: 'pppm 1e-5', modify: PP, native: null }],
  // Measured with native LAMMPS: ERROR: KSpace style ewald/dipole is incompatible with Pair style lj/cut/coul/long (dipole atoms)
  ['lj/cut/coul/long + ewald/dipole', { atom: 'dipole', pair: 'lj/cut/coul/long 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'ewald/dipole 1e-5', modify: EW, native: 'KSpace style ewald/dipole is incompatible with Pair style lj/cut/coul/long' }],
  // Measured with native LAMMPS: runs with ewald/dipole 1e-5 on lj/cut/dipole/long
  ['lj/cut/dipole/long + ewald/dipole', { atom: 'dipole', pair: 'lj/cut/dipole/long 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'ewald/dipole 1e-5', modify: EW, native: null }],
  // Measured with native LAMMPS: runs with pppm/dipole 1e-5 on lj/cut/dipole/long
  ['lj/cut/dipole/long + pppm/dipole', { atom: 'dipole', pair: 'lj/cut/dipole/long 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'pppm/dipole 1e-5', modify: PP, native: null }],
  // Measured with native LAMMPS: ERROR: KSpace style pppm/dipole is incompatible with Pair style lj/cut/coul/long
  ['lj/cut/coul/long + pppm/dipole', { atom: 'dipole', pair: 'lj/cut/coul/long 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'pppm/dipole 1e-5', modify: PP, native: 'KSpace style pppm/dipole is incompatible with Pair style lj/cut/coul/long' }],
  // Measured with native LAMMPS: runs with ewald 1e-5 on pair_style lepton/coul 8.0 ewald
  ['lepton/coul ewald keyword + ewald', { atom: 'charge', pair: 'lepton/coul 8.0 ewald', coeff: 'pair_coeff * * "qi*qj/r"', kspace: 'ewald 1e-5', modify: EW, native: null }],
  // Measured with native LAMMPS: ERROR: KSpace style ewald is incompatible with Pair style lepton/coul (no keyword)
  ['lepton/coul without keyword + ewald', { atom: 'charge', pair: 'lepton/coul 8.0', coeff: 'pair_coeff * * "qi*qj/r"', kspace: 'ewald 1e-5', modify: EW, native: 'KSpace style ewald is incompatible with Pair style lepton/coul' }],
  // Measured with native LAMMPS: runs with pppm 1e-5 on pair_style lepton/coul 8.0 pppm
  ['lepton/coul pppm keyword + pppm', { atom: 'charge', pair: 'lepton/coul 8.0 pppm', coeff: 'pair_coeff * * "qi*qj/r"', kspace: 'pppm 1e-5', modify: PP, native: null }],
  // Measured with native LAMMPS: ERROR: KSpace style ewald is incompatible with Pair style lepton/coul (pppm keyword only)
  ['lepton/coul pppm keyword + ewald', { atom: 'charge', pair: 'lepton/coul 8.0 pppm', coeff: 'pair_coeff * * "qi*qj/r"', kspace: 'ewald 1e-5', modify: EW, native: 'KSpace style ewald is incompatible with Pair style lepton/coul' }],
  // Measured with native LAMMPS: runs with ewald 1e-5 on hybrid/overlay (sub-style coul/long has ewald and pppm)
  ['hybrid/overlay with coul/long + ewald', { atom: 'charge', pair: 'hybrid/overlay lj/cut 8.0 coul/long 8.0', coeff: 'pair_coeff * * lj/cut 0.1 3.0\npair_coeff * * coul/long', kspace: 'ewald 1e-5', modify: EW, native: null }],
  // Measured with native LAMMPS: ERROR: KSpace style ewald is incompatible with Pair style hybrid/overlay (no long-range sub-style)
  ['hybrid/overlay without long sub-style + ewald', { atom: 'charge', pair: 'hybrid/overlay lj/cut 8.0 coul/cut 8.0', coeff: 'pair_coeff * * lj/cut 0.1 3.0\npair_coeff * * coul/cut', kspace: 'ewald 1e-5', modify: EW, native: 'KSpace style ewald is incompatible with Pair style hybrid/overlay' }],
  // Measured with native LAMMPS: runs with ewald 1e-5 on zero/coul
  ['zero/coul + ewald', { atom: 'charge', pair: 'zero/coul 8.0', coeff: 'pair_coeff * *', kspace: 'ewald 1e-5', modify: EW, native: null }],
  // Measured with native LAMMPS: ERROR: KSpace style pppm/disp is incompatible with Pair style zero/coul
  ['zero/coul + pppm/disp', { atom: 'charge', pair: 'zero/coul 8.0', coeff: 'pair_coeff * *', kspace: 'pppm/disp 1e-5', modify: PPD, native: 'KSpace style pppm/disp is incompatible with Pair style zero/coul' }],
  // Measured with native LAMMPS: KSpace style ewald is incompatible with Pair style zero (src/kspace.cpp:215)
  ['zero + ewald', { atom: 'charge', pair: 'zero 8.0', coeff: 'pair_coeff * *', kspace: 'ewald 1e-5', modify: EW, native: 'KSpace style ewald is incompatible with Pair style zero' }],
  // Measured with native LAMMPS: runs with kspace_style zero 1e-5 (kspace_zero.cpp has no pair_check)
  ['coul/long + zero', { atom: 'charge', pair: 'coul/long 8.0', coeff: 'pair_coeff * *', kspace: 'zero 1e-5', modify: '', native: null }],
  // Measured with native LAMMPS: ERROR: KSpace style is incompatible with Pair style (kspace_zero.cpp:74, no cut_coul)
  ['lj/cut + zero', { atom: 'charge', pair: 'lj/cut 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'zero 1e-5', modify: '', native: 'KSpace style is incompatible with Pair style' }],
  // Measured with native LAMMPS: ERROR: KSpace style pppm/tip4p is incompatible with Pair style lj/cut/coul/long
  ['lj/cut/coul/long + pppm/tip4p', { atom: 'charge', pair: 'lj/cut/coul/long 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'pppm/tip4p 1e-5', modify: PP, native: 'KSpace style pppm/tip4p is incompatible with Pair style lj/cut/coul/long' }],
  // Measured with native LAMMPS: ERROR: KSpace style pppm/disp is incompatible with Pair style lj/cut/coul/long
  ['lj/cut/coul/long + pppm/disp', { atom: 'charge', pair: 'lj/cut/coul/long 8.0', coeff: 'pair_coeff * * 0.1 3.0', kspace: 'pppm/disp 1e-5', modify: PPD, native: 'KSpace style pppm/disp is incompatible with Pair style lj/cut/coul/long' }],
  // Measured with native LAMMPS: ERROR: KSpace style msm is incompatible with Pair style coul/long
  ['coul/long + msm', { atom: 'charge', pair: 'coul/long 8.0', coeff: 'pair_coeff * *', kspace: 'msm 1e-5', modify: '', native: 'KSpace style msm is incompatible with Pair style coul/long' }],
  // Measured with native LAMMPS: runs with ewald/disp 1e-5 on lj/long/coul/long long long (kspace ewald/disp dispersion)
];

// Measured with native LAMMPS: Kspace style requires atom attribute q (src/KSPACE/ewald.cpp:106, before pair_check)
describe('KSpace::pair_check charge attribute', () => {
  it('ewald on atomic atoms is refused with the charge message before the pair check', async () => {
    const body = `units real
atom_style atomic
lattice sc 3.0
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 box
mass * 20.0
pair_style lj/cut 8.0
pair_coeff * * 0.1 3.0
kspace_style ewald 1e-5
kspace_modify gewald 0.3
run 0
`;
    expect(await run(body)).toContain('Kspace style requires atom attribute q');
  });
});

describe('KSpace::pair_check (native refusals and acceptances, measured with native LAMMPS)', () => {
  for (const [label, c] of cases) {
    it(`${label}`, async () => {
      const body = preamble(c.atom === 'dipole' ? 'hybrid charge dipole sphere' : c.atom)
        + (c.atom === 'dipole' ? DIPOLE_SETUP : '')
        + `pair_style ${c.pair}\n${c.coeff}\nkspace_style ${c.kspace}\n${c.modify ? c.modify + '\n' : ''}run 0\n`;
      const err = await run(body);
      if (c.native === null) expect(err).toBeNull();
      else expect(err).toContain(c.native);
    });
  }
});

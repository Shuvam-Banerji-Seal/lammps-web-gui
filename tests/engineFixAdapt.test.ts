import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix adapt (docs.lammps.org/fix_adapt.html): argument errors, and that the
 * energy follows an adapted pair parameter at fixed positions (run 0 keeps the
 * coordinates). Oracle parity with native LAMMPS is in tests/oracle/w9adapt_*.in.
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  let error: unknown = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e;
  }
  const thermo = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row);
  return { thermo, error, files, message: error instanceof Error ? error.message : String(error ?? '') };
};

const SYS1 = `
units           lj
atom_style      atomic
boundary        p p p
lattice         sc 1.0
region          box block 0 2 0 2 0 2
create_box      1 box
create_atoms    1 box
mass            1 1.0
thermo_style    custom step evdwl pe
thermo_modify   format float %.15g
`;

const SYS2 = SYS1.replace('create_box      1 box', 'create_box      2 box')
  .replace('mass            1 1.0', 'mass * 1.0\nset group all type 1\ngroup odd id 1:8:2\nset group odd type 2');

describe('fix adapt: energy follows the adapted parameter', () => {
  it('lj/cut epsilon doubled doubles evdwl at fixed positions', async () => {
    const r = await runScript(`${SYS1}
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
run 0
variable e equal 2.0
fix 1 all adapt 0 pair lj/cut epsilon 1 1 v_e
run 0
`);
    expect(r.error).toBeNull();
    expect(r.thermo.length).toBe(2);
    const e1 = r.thermo[0].evdwl, e2 = r.thermo[1].evdwl;
    expect(Math.abs(e1)).toBeGreaterThan(0);
    expect(e2).toBeCloseTo(2 * e1, 12);
  });

  it('lj/cut sigma changes the energy (sigma 0.9 vs 1.0 at the same positions)', async () => {
    const r = await runScript(`${SYS1}
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
run 0
variable s equal 0.9
fix 1 all adapt 0 pair lj/cut sigma 1 1 v_s
run 0
`);
    expect(r.error).toBeNull();
    expect(r.thermo[1].evdwl).not.toBeCloseTo(r.thermo[0].evdwl, 6);
  });

  it('soft a doubled doubles the energy (soft is linear in A)', async () => {
    const r = await runScript(`${SYS1}
pair_style soft 2.5
pair_coeff * * 10.0
run 0
variable a equal 20.0
fix 1 all adapt 0 pair soft a * * v_a
run 0
`);
    expect(r.error).toBeNull();
    expect(r.thermo[1].evdwl).toBeCloseTo(2 * r.thermo[0].evdwl, 10);
  });

  it('morse d0 doubled doubles the energy; D0 is not a name native extract() knows', async () => {
    const script = (name: string) => `${SYS1}
pair_style morse 2.5
pair_coeff * * 1.0 1.5 1.0
run 0
variable d equal 2.0
fix 1 all adapt 0 pair morse ${name} * * v_d
run 0
`;
    const r = await runScript(script('d0'));
    expect(r.error).toBeNull();
    expect(r.thermo[1].evdwl).toBeCloseTo(2 * r.thermo[0].evdwl, 10);
    // Measured with native LAMMPS: ERROR: Fix adapt pair style morse param D0 not supported (src/fix_adapt.cpp:440)
    expect((await runScript(script('D0'))).message).toContain('Fix adapt pair style morse param D0 not supported');
  });

  it('scale yes multiplies the value the run started with', async () => {
    const r = await runScript(`${SYS1}
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
run 0
variable e equal 3.0
fix 1 all adapt 0 pair lj/cut epsilon 1 1 v_e scale yes
run 0
`);
    expect(r.error).toBeNull();
    expect(r.thermo[1].evdwl).toBeCloseTo(3 * r.thermo[0].evdwl, 12);
  });

  it('reset yes restores the original value at the end of the run; reset no keeps it', async () => {
    const body = (reset: string) => `${SYS1}
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
run 0
variable e equal 2.0
fix 1 all adapt 0 pair lj/cut epsilon 1 1 v_e reset ${reset}
run 0
unfix 1
run 0
`;
    const yes = await runScript(body('yes'));
    const no = await runScript(body('no'));
    expect(yes.error).toBeNull();
    expect(no.error).toBeNull();
    const e1 = yes.thermo[0].evdwl;
    expect(yes.thermo[2].evdwl).toBeCloseTo(e1, 12);
    expect(no.thermo[2].evdwl).toBeCloseTo(2 * e1, 12);
  });

  it('a mixed (mixing-derived) epsilon(1,2) follows the diagonal epsilon(1,1)', async () => {
    // pair_coeff 1 1 2.0 with 1 2 mixed (geometric) equals the adapted run
    const adapted = await runScript(`${SYS2}
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
pair_coeff 2 2 0.5 1.1
variable e equal 2.0
fix 1 all adapt 0 pair lj/cut epsilon 1 1 v_e
run 0
`);
    const direct = await runScript(`${SYS2}
pair_style lj/cut 2.5
pair_coeff 1 1 2.0 1.0
pair_coeff 2 2 0.5 1.1
run 0
`);
    expect(adapted.error).toBeNull();
    expect(direct.error).toBeNull();
    expect(adapted.thermo[0].evdwl).toBeCloseTo(direct.thermo[0].evdwl, 12);
  });

  it('atom charge scaled by a variable changes ecoul as q squared', async () => {
    const r = await runScript(`
units           lj
atom_style      charge
boundary        p p p
lattice         sc 1.0
region          box block 0 2 0 2 0 2
create_box      1 box
create_atoms    1 box
mass            1 1.0
set             group all charge 1.0
pair_style      coul/cut 2.5
pair_coeff      * *
thermo_style    custom step ecoul pe
thermo_modify   format float %.15g
run 0
variable q equal 2.0
fix 1 all adapt 0 atom charge v_q
run 0
`);
    expect(r.error).toBeNull();
    expect(r.thermo[1].ecoul).toBeCloseTo(4 * r.thermo[0].ecoul, 10);
  });

  it('atom diameter with mass yes: mass scales with d^3 (sphere style, dynamic radii)', async () => {
    const r = await runScript(`
units           lj
atom_style      sphere 1
boundary        p p p
lattice         sc 1.0
region          box block 0 2 0 2 0 2
create_box      1 box
create_atoms    1 box
set             group all diameter 1.0 density 1.0
variable d equal 2.0
fix 1 all adapt 0 atom diameter v_d
run 0
write_dump all custom dmp.txt id radius mass modify format float %.12g sort id
`);
    expect(r.error).toBeNull();
    const lines = (r.files.get('dmp.txt') ?? '').trim().split('\n');
    const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
    const rows = lines.slice(k + 1).map((l) => l.trim().split(/\s+/).map(Number));
    expect(rows.length).toBe(8);
    for (const [, radius, mass] of rows) {
      expect(radius).toBeCloseTo(1.0, 10);
      // density 1.0 (per-atom) and volume 4/3 pi r^3: mass = 4/3 pi (d/2)^3 at d = 2
      expect(mass).toBeCloseTo((4 / 3) * Math.PI, 8);
    }
  });

  it('atom diameter with mass no keeps the mass', async () => {
    const r = await runScript(`
units           lj
atom_style      sphere 1
boundary        p p p
lattice         sc 1.0
region          box block 0 2 0 2 0 2
create_box      1 box
create_atoms    1 box
set             group all diameter 1.0 density 1.0
variable d equal 2.0
fix 1 all adapt 0 atom diameter v_d mass no
run 0
write_dump all custom dmp.txt id radius mass modify format float %.12g sort id
`);
    expect(r.error).toBeNull();
    const lines = (r.files.get('dmp.txt') ?? '').trim().split('\n');
    const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
    const first = lines[k + 1].trim().split(/\s+/).map(Number);
    expect(first[1]).toBeCloseTo(1.0, 10);
    expect(first[2]).toBeCloseTo((4 / 3) * Math.PI * 0.125, 8);
  });
});

describe('fix adapt: argument and option errors', () => {
  const pairSetup = `${SYS2}
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
pair_coeff 2 2 0.5 1.1
variable e equal 2.0
`;

  it('rejects a negative every value as native does', async () => {
    const r = await runScript(`${SYS1}\nvariable e equal 2.0\nfix 1 all adapt -1 pair lj/cut epsilon 1 1 v_e\n`);
    expect(r.message).toContain('Illegal fix adapt every value -1');
  });

  it('at least one attribute is required', async () => {
    const r = await runScript(`${SYS1}\nfix 1 all adapt 1 scale no\n`);
    expect(r.message).toMatch(/no attribute given/);
  });

  it('unknown keyword is refused with the native message', async () => {
    const r = await runScript(`${SYS1}\nfix 1 all adapt 1 frob v_x\n`);
    expect(r.message).toContain('Unknown fix adapt keyword frob');
  });

  it('kspace attribute: native texts without a variable or a kspace style (numbers in oracle w382adapt_kspace)', async () => {
    const nv = await runScript(`${pairSetup}\nfix 1 all adapt 1 kspace e\n`);
    expect(nv.message).toMatch(/Argument must be variable not e/);
    const nk = await runScript(`${pairSetup}\nvariable e equal 0.5\nfix 1 all adapt 1 kspace v_e\nrun 0\n`);
    expect(nk.message).toMatch(/Fix adapt expected a kspace style but none was defined/);
  });

  it('atom diameter needs atom_style sphere 1, as native AtomVecSphere::init', async () => {
    const box = 'units lj\natom_style sphere\nlattice fcc 0.8442\nregion b block 0 2 0 2 0 2\ncreate_box 1 b\ncreate_atoms 1 box\npair_style lj/cut 2.5\npair_coeff * * 1 1\nvariable d equal 1.2\n';
    const r = await runScript(`${box}fix 1 all adapt 1 atom diameter v_d\nrun 0\n`);
    expect(r.message).toMatch(/Fix adapt changes atom radii but atom_style sphere is not dynamic/);
    const ok = await runScript(`${box.replace('atom_style sphere', 'atom_style sphere 1')}fix 1 all adapt 1 atom diameter v_d\nrun 0\n`);
    expect(ok.message ?? '').toBe('');
  });

  // bonded attributes: native's texts (src/fix_adapt.cpp at b332f28); numerical parity is in w345-w347adapt_bonded*
  const MOL = `
units           lj
atom_style      molecular
lattice         sc 1.0
region          box block 0 2 0 2 0 2
create_box      1 box bond/types 1 angle/types 1 dihedral/types 1 improper/types 1 extra/bond/per/atom 2
create_atoms    1 box
mass            1 1.0
pair_style      lj/cut 1.1
pair_coeff      * * 1.0 1.0
bond_style      harmonic
bond_coeff      1 10.0 1.0
angle_style     harmonic
angle_coeff     1 10.0 90.0
dihedral_style  charmm
dihedral_coeff  1 1.0 2 180 0.0
improper_style  harmonic
improper_coeff  1 1.0 0.0
variable        e equal 2.0
`;
  it.each([
    ['bond morse d0 1 v_e', /Fix adapt bond style morse does not exist/],
    ['bond harmonic r1 1 v_e', /Fix adapt bond style parameter r1 not supported/],
    ['angle harmonic theta 1 v_e', /Fix adapt angle style parameter theta not supported/],
    // native writes doubles through the integer multiplicity array: refused, as the doc table lists only k
    ['dihedral charmm n 1 v_e', /Fix adapt dihedral style parameter n not supported/],
    ['improper harmonic k 1 2.0', /Argument must be variable not 2.0/],
  ])('bonded attribute errors: %s', async (attr, msg) => {
    const r = await runScript(`${MOL}fix 1 all adapt 1 ${attr}\nrun 0\n`);
    expect(r.message).toMatch(msg);
  });

  it('a bonded hybrid style is refused by name', async () => {
    // every sub-style used: an unused one stops first at the bond style's init (Bond hybrid sub-style X is not used)
    const r = await runScript(`${MOL.replace('bond_style      harmonic\nbond_coeff      1 10.0 1.0', 'bond_style      hybrid harmonic\nbond_coeff      1 harmonic 10.0 1.0')}fix 1 all adapt 1 bond harmonic k 1 v_e\nrun 0\n`);
    expect(r.message).toMatch(/bond with bond_style hybrid is not supported/);
  });

  it('the variable must exist when the run starts (native FixAdapt::init), not when the fix is defined', async () => {
    const r = await runScript(`${pairSetup}\nfix 1 all adapt 1 pair lj/cut epsilon 1 1 v_nope\nrun 0\n`);
    expect(r.message).toMatch(/Variable name nope for fix adapt does not exist/);
    const later = await runScript(`${pairSetup}\nfix 1 all adapt 1 pair lj/cut epsilon 1 1 v_eps\nvariable eps equal 1.0\nrun 0\n`);
    expect(later.message).toBe('');
  });

  it('the value must be v_name', async () => {
    const r = await runScript(`${pairSetup}\nfix 1 all adapt 1 pair lj/cut epsilon 1 1 2.0\n`);
    expect(r.message).toMatch(/Argument must be variable not 2\.0/);
  });

  it('atom-style variables are rejected (equal-style only)', async () => {
    const r = await runScript(`${pairSetup}\nvariable av atom x/10\nfix 1 all adapt 1 pair lj/cut epsilon 1 1 v_av\nrun 0\n`);
    expect(r.message).toMatch(/Variable av for fix adapt is invalid style/);
  });

  it('a parameter extract() does not name is refused at the run (lj/cut has no cut)', async () => {
    // Measured with native LAMMPS: ERROR: Fix adapt pair style lj/cut param cut not supported (src/fix_adapt.cpp:440)
    const r = await runScript(`${pairSetup}\nfix 1 all adapt 1 pair lj/cut cut 1 1 v_e\nrun 0\n`);
    expect(r.message).toContain('Fix adapt pair style lj/cut param cut not supported');
  });

  it('a pair style other than the defined one is not found', async () => {
    // Measured with native LAMMPS: ERROR: Fix adapt pair style eam not found (src/fix_adapt.cpp:436), and the same for soft
    for (const w of ['eam epsilon', 'soft a']) {
      const r = await runScript(`${pairSetup}\nfix 1 all adapt 1 pair ${w} 1 1 v_e\nrun 0\n`);
      expect(r.message).toContain(`Fix adapt pair style ${w.split(' ')[0]} not found`);
    }
  });

  it('style:N on a plain pair style names that style (pair_match ignores N for the defined style)', async () => {
    // Measured with native LAMMPS: pe 0.359199256203132 before, -0.794635511109422 with lj/cut:2 epsilon 1 1 v_e
    const r = await runScript(`${pairSetup}\nrun 0\nfix 1 all adapt 1 pair lj/cut:2 epsilon 1 1 v_e\nrun 0\n`);
    expect(r.error).toBeNull();
    expect(r.thermo[0].pe).toBeCloseTo(0.359199256203132, 12);
    expect(r.thermo[1].pe).toBeCloseTo(-0.794635511109422, 12);
  });

  it('a type pair set only by mixing takes the value, which the re-mixing at reinit replaces (no effect)', async () => {
    // Measured with native LAMMPS: pe 0.359199256203132 both before and with fix adapt ... epsilon 1 2 v_e
    const r = await runScript(`${pairSetup}\nrun 0\nfix 1 all adapt 1 pair lj/cut epsilon 1 2 v_e\nrun 0\n`);
    expect(r.error).toBeNull();
    expect(r.thermo[1].pe).toBe(r.thermo[0].pe);
  });

  it('atom diameter needs atom_style sphere', async () => {
    const r = await runScript(`${SYS1}\nvariable d equal 2.0\nfix 1 all adapt 0 atom diameter v_d\nrun 0\n`);
    expect(r.message).toMatch(/atom diameter needs atom_style sphere/);
  });

  it('atom charge needs atom_style charge or full', async () => {
    const r = await runScript(`${SYS1}\nvariable q equal 2.0\nfix 1 all adapt 0 atom charge v_q\nrun 0\n`);
    expect(r.message).toMatch(/atom charge needs atom_style charge or full/);
  });

  it('diameter/disc is refused', async () => {
    const r = await runScript(`${SYS1}\nvariable d equal 2.0\nfix 1 all adapt 0 atom diameter/disc v_d\n`);
    expect(r.message).toMatch(/diameter\/disc is not supported/);
  });

  it('scale and reset take yes or no', async () => {
    const r = await runScript(`${pairSetup}\nfix 1 all adapt 1 pair lj/cut epsilon 1 1 v_e reset maybe\n`);
    expect(r.message).toContain("Expected boolean parameter instead of 'maybe' in input script or data file");
  });
});

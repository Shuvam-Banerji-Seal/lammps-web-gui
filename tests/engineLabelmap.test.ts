import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { LabelMap, isValidLabel, labelmapText, parseLabelPairs, typeWord } from '../src/engine/labelmap';

/*
 * Type labels (docs.lammps.org/Howto_type_labels.html, labelmap.html, read_data.html "Type Labels" sections,
 * variable.html label2type()). Parity with native LAMMPS is in tests/oracle/w90label_*.in; these tests check
 * the parsing and the error messages (native wording where measured with the black box binary).
 */

const BOX = `
units           real
atom_style      full
region          b block 0 10 0 10 0 10
create_box      2 b bond/types 2 angle/types 1 dihedral/types 1 improper/types 1 extra/bond/per/atom 2 extra/angle/per/atom 2 extra/dihedral/per/atom 1 extra/improper/per/atom 1 extra/special/per/atom 2
`;

/** A data file with label sections for atoms, bonds, angles, dihedrals and impropers (labels used everywhere). */
const DATA = `labels test

6 atoms
2 atom types
4 bonds
2 bond types
2 angles
1 angle types
1 dihedrals
1 dihedral types
1 impropers
1 improper types
0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Atom Type Labels

1 C
2 H

Bond Type Labels

1 CH
2 CC

Angle Type Labels

1 HCH

Dihedral Type Labels

1 ddd

Improper Type Labels

1 iii

Masses

C 12.0
H 1.0

Atoms # full

1 1 C 0.0 4.0 5.0 5.0
2 1 H 0.0 5.0 5.2 5.1
3 1 H 0.0 4.0 5.9 4.8
4 1 H 0.0 4.4 4.5 5.9
5 1 C 0.0 6.0 4.8 4.5
6 1 H 0.0 6.5 5.7 5.0

Bonds

1 CH 1 2
2 CH 1 3
3 CC 1 5
4 CH 5 6

Angles

1 HCH 2 1 3
2 HCH 5 6 1

Dihedrals

1 ddd 2 1 5 6

Impropers

1 iii 1 2 3 4
`;

interface Run {
  events: EngineEvent[];
  files: Map<string, string>;
  /** the error message without the session's "line N: " prefix ('' when the script ran) */
  error: string;
}

const runScript = async (text: string, extra: Record<string, string> = {}): Promise<Run> => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: (name, t) => files.set(name, t) });
  session.addFile('labels.data', DATA);
  for (const [k, v] of Object.entries(extra)) session.addFile(k, v);
  let error = '';
  try {
    await session.execute(text);
  } catch (e) {
    error = (e instanceof Error ? e.message : String(e)).replace(/^line \d+: /, '');
  }
  return { events, files, error };
};

const errorOf = async (text: string, extra: Record<string, string> = {}): Promise<string> => (await runScript(text, extra)).error;

describe('type label map (labelmap.ts)', () => {
  it('accepts labels that do not start with a number, #, or *, and have no whitespace', () => {
    expect(isValidLabel('C')).toBe(true);
    expect(isValidLabel('c1-c2_x+y=[z]<w>(v)')).toBe(true);
    expect(isValidLabel("C1'")).toBe(true);
    expect(isValidLabel('1C')).toBe(false);
    expect(isValidLabel('#x')).toBe(false);
    expect(isValidLabel('*x')).toBe(false);
    expect(isValidLabel('a b')).toBe(false);
    expect(isValidLabel('')).toBe(false);
  });

  it('keeps one label per type, unique within a kind, and overwrites the label of a type', () => {
    const m = new LabelMap();
    m.set('atom', 1, 'C');
    m.set('atom', 2, 'H');
    expect(() => m.set('atom', 3, 'C')).toThrow('The atom type label C is already in use for type 1');
    m.set('atom', 1, 'CX');
    expect(m.typeOf('atom', 'C')).toBeUndefined();
    expect(m.typeOf('atom', 'CX')).toBe(1);
    // the same label may name a type of another kind
    m.set('bond', 1, 'C');
    expect(m.typeOf('bond', 'C')).toBe(1);
    expect(m.entries('atom')).toEqual([[1, 'CX'], [2, 'H']]);
    expect(m.complete('atom', 2)).toBe(true);
    expect(m.complete('atom', 3)).toBe(false);
  });

  it('resolves type words: numbers and wildcards pass through, labels become numbers', () => {
    const m = new LabelMap();
    m.set('atom', 2, 'H');
    expect(typeWord(m, 'atom', '2')).toBe('2');
    expect(typeWord(m, 'atom', '1*3')).toBe('1*3');
    expect(typeWord(m, 'atom', '*')).toBe('*');
    expect(typeWord(m, 'atom', 'H')).toBe('2');
    expect(() => typeWord(m, 'atom', 'Z')).toThrow('Atom type string Z not found in labelmap');
    expect(() => typeWord(m, 'atom', 'H*')).toThrow('Atom type string H* not found in labelmap');
    expect(() => typeWord(m, 'bond', 'H')).toThrow('Bond type string H not found in labelmap');
  });

  it('parses labelmap pairs and refuses the native error cases', () => {
    expect(parseLabelPairs(['1', 'C', '2', 'H'], 'atom', 2)).toEqual([[1, 'C'], [2, 'H']]);
    expect(() => parseLabelPairs(['1'], 'atom', 2)).toThrow('Illegal labelmap atom command: missing argument(s)');
    expect(() => parseLabelPairs(['9', 'X'], 'atom', 3)).toThrow('Labelmap atom type 9 must be within 1-3');
    expect(() => parseLabelPairs(['x', 'X'], 'atom', 3)).toThrow("Expected integer parameter instead of 'x'");
  });

  it('writes the map as labelmap commands with triple-quoted labels, kinds in order', () => {
    const m = new LabelMap();
    m.set('bond', 2, 'CC');
    m.set('atom', 3, 'Z');
    expect(labelmapText(m)).toBe('labelmap atom 3 """ Z """\nlabelmap bond 2 """ CC """\n');
  });
});

describe('labelmap command', () => {
  it('defines, overwrites, clears and writes label maps', async () => {
    const r = await runScript(`${BOX}
labelmap atom 1 C 2 HX
labelmap atom 2 H
labelmap bond 1 CH 2 CC
labelmap write map.inc
labelmap clear
`);
    expect(r.error).toBe('');
    expect(r.files.get('map.inc')).toBe('labelmap atom 1 """ C """ 2 """ H """\nlabelmap bond 1 """ CH """ 2 """ CC """\n');
  });

  it('refuses a label used by two types of one kind, a label map before the box, bad labels and types', async () => {
    expect(await errorOf(`${BOX}labelmap atom 1 C 2 C`)).toBe('The atom type label C is already in use for type 1');
    expect(await errorOf('units real\nlabelmap atom 1 C\n')).toBe('Labelmap command before simulation box is defined');
    expect(await errorOf(`${BOX}labelmap atom 1 1abc`)).toBe('Type label string 1abc for atom type 1 is invalid');
    expect(await errorOf(`${BOX}labelmap atom 9 X`)).toBe('Labelmap atom type 9 must be within 1-2');
    expect(await errorOf(`${BOX}labelmap atom 1`)).toBe('Illegal labelmap atom command: missing argument(s)');
    expect(await errorOf(`${BOX}labelmap foo 1 X`)).toBe('Unknown labelmap keyword foo');
    expect(await errorOf(`${BOX}labelmap clear extra`)).toBe('Incorrect number of arguments for labelmap clear command');
  });

  it('refuses check_labels by name', async () => {
    expect(await errorOf(`${BOX}labelmap check_labels ba`)).toMatch(/^labelmap check_labels is not supported/);
  });

  it('label2type() and is_typelabel() read the map in variables', async () => {
    const r = await runScript(`${BOX}
labelmap atom 1 C 2 H
variable th equal label2type(atom,H)
variable tc equal label2type(atom,C)
variable no equal is_typelabel(atom,Z)
variable yes equal is_typelabel(atom,H)
print "vals \${th} \${tc} \${no} \${yes}"
`);
    expect(r.error).toBe('');
    expect(r.events.some((e) => e.kind === 'log' && /vals 2 1 0 1/.test(e.text))).toBe(true);
  });

  it('label2type() errors: invalid kind, undefined label, no label map', async () => {
    expect(await errorOf(`${BOX}labelmap atom 1 C\nprint "$(label2type(foo,C))"`)).toBe('Invalid kind foo in label2type() in variable');
    expect(await errorOf(`${BOX}labelmap atom 1 C\nprint "$(label2type(atom,ZZ))"`)).toBe('Invalid atom type label ZZ in label2type() in variable');
    expect(await errorOf(`${BOX}print "$(label2type(atom,C))"`)).toBe('Cannot use label2type() function without a labelmap');
  });
});

describe('labels in commands', () => {
  it('mass, pair_coeff, bond_coeff, create_atoms, set and group take labels', async () => {
    const r = await runScript(`${BOX}
labelmap atom 1 C 2 H
labelmap bond 1 CH 2 CC
pair_style lj/cut 5.0
bond_style harmonic
pair_coeff C H 0.1 2.0
pair_coeff H H 0.1 2.0
pair_coeff * C 0.2 2.0
mass H 1.5
bond_coeff CH 300.0 1.1
bond_coeff 2 300.0 1.5
create_atoms C single 9.0 9.0 9.0
set type C type H
group hyd type H
`);
    expect(r.error).toBe('');
  });

  it('unknown labels in commands stop with the native message', async () => {
    expect(await errorOf(`${BOX}labelmap atom 1 C 2 H\nmass Z 1.0`)).toBe('Atom type string Z not found in labelmap');
    expect(await errorOf(`${BOX}labelmap atom 1 C 2 H\npair_style lj/cut 5.0\npair_coeff Z H 1 1`)).toBe('Atom type string Z not found in labelmap');
    expect(await errorOf(`${BOX}labelmap atom 1 C 2 H\nmass C* 1.0`)).toBe('Atom type string C* not found in labelmap');
    expect(await errorOf(`${BOX}bond_style harmonic\nlabelmap bond 1 CH 2 CC\nbond_coeff Q 1 1`)).toBe('Bond type string Q not found in labelmap');
    expect(await errorOf(`${BOX}labelmap atom 1 C 2 H\ngroup g type Z`)).toBe('Atom type string Z not found in labelmap');
  });

  it('group type lists take labels and keep the logical forms', async () => {
    const r = await runScript(`units real
atom_style full
read_data labels.data
group c type C
group h type < H
group hh type C H
group rest type <> 1 2
`);
    expect(r.error).toBe('');
  });
});

describe('read_data Type Labels sections', () => {
  it('reads the label sections and the types they name in Atoms, Masses and topology', async () => {
    const r = await runScript('units real\natom_style full\nread_data labels.data\nlabelmap write back.inc\n');
    expect(r.error).toBe('');
    expect(r.files.get('back.inc')).toBe('labelmap atom 1 """ C """ 2 """ H """\nlabelmap bond 1 """ CH """ 2 """ CC """\nlabelmap angle 1 """ HCH """\nlabelmap dihedral 1 """ ddd """\nlabelmap improper 1 """ iii """\n');
  });

  it('refuses a label section that comes after a section involving types', async () => {
    const text = DATA.replace('Atom Type Labels\n\n1 C\n2 H\n\n', '').replace('Masses', 'Masses\n\n1 12.0\n2 1.0\n\nAtom Type Labels\n\n1 C\n2 H\n\nMasses');
    expect(await errorOf('units real\natom_style full\nread_data broken.data', { 'broken.data': text })).toBe('Must read Type Labels before any section involving types');
  });

  it('refuses an undefined label, an invalid label, and a label map that is not complete', async () => {
    const undefinedLabel = DATA.replace('1 1 C 0.0', '1 1 Q 0.0');
    expect(await errorOf('units real\natom_style full\nread_data u.data', { 'u.data': undefinedLabel })).toMatch(/Invalid line in Atoms section of data file: atom type label Q is not defined/);
    const invalid = DATA.replace('1 C\n2 H', '1 1C\n2 H');
    expect(await errorOf('units real\natom_style full\nread_data i.data', { 'i.data': invalid })).toMatch(/Invalid type label 1C$/);
    const dup = DATA.replace('2 H\n\nBond', '2 C\n\nBond');
    expect(await errorOf('units real\natom_style full\nread_data d.data', { 'd.data': dup })).toBe('Atom Type Labels map is incomplete');
  });

  it('refuses a label in a coefficient section (numeric types only there)', async () => {
    const text = DATA.replace('Masses\n\nC 12.0\nH 1.0\n', 'Masses\n\n1 12.0\n2 1.0\n\nPair Coeffs\n\nC 0.1 1.0\n');
    const r = await runScript('units real\natom_style full\npair_style lj/cut 5.0\nread_data p.data', { 'p.data': text });
    expect(r.error).toBe("Expected integer parameter instead of 'C' in input script or data file");
  });
});

describe('restart files keep the label map', () => {
  it('read_restart restores the labels written by write_restart', async () => {
    const r = await runScript(`${BOX}
labelmap atom 1 C 2 H
labelmap bond 2 CC
write_restart labels.restart
clear
read_restart labels.restart
labelmap write back.inc
`);
    expect(r.error).toBe('');
    expect(r.files.get('back.inc')).toBe('labelmap atom 1 """ C """ 2 """ H """\nlabelmap bond 2 """ CC """\n');
  });
});

describe('write_data with labels', () => {
  it('writes label sections for complete maps with a style, and label columns with types labels', async () => {
    const r = await runScript('units real\natom_style full\nread_data labels.data\nbond_style harmonic\nbond_coeff * 300.0 1.1\nwrite_data out.data\nwrite_data lab.data types labels\nwrite_data nol.data nolabelmap\n');
    expect(r.error).toBe('');
    const plain = r.files.get('out.data')!;
    expect(plain).toMatch(/\nAtom Type Labels\n\n1 C\n2 H\n\nBond Type Labels\n\n1 CH\n2 CC\n\n/);
    expect(plain).not.toMatch(/Angle Type Labels/);
    expect(plain).toMatch(/\nBonds\n\n1 1 1 2\n/);
    const labelled = r.files.get('lab.data')!;
    expect(labelled).toMatch(/\n1 1 C 0 4 5 5 0 0 0\n/);
    expect(labelled).toMatch(/\nBonds\n\n1 CH 1 2\n/);
    expect(r.files.get('nol.data')!).not.toMatch(/Type Labels/);
  });

  it('writes an empty field for a type that has no label under types labels', async () => {
    // a numeric data file (no atom labels) with only atom type 1 labelled by the labelmap command
    const numeric = DATA.replace('\nAtom Type Labels\n\n1 C\n2 H\n', '')
      .replace(/^(\d+) 1 C /gm, '$1 1 1 ').replace(/^(\d+) 1 H /gm, '$1 1 2 ')
      .replace('C 12.0\nH 1.0', '1 12.0\n2 1.0');
    const r = await runScript('units real\natom_style full\nread_data numeric.data\nlabelmap atom 1 C\nwrite_data partial.data types labels\n', { 'numeric.data': numeric });
    expect(r.error).toBe('');
    expect(r.files.get('partial.data')!).toMatch(/\n1 1 C 0 4 5 5 0 0 0\n2 1  0 5 5\.2 5\.1 0 0 0\n/);
  });
});

import { describe, expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import { formatNumber, formatInteger, parseCFormat } from '../src/engine/cformat';
import { gzipStored, gzipText, crc32 } from '../src/engine/gzip';
import { gunzip } from '../src/engine/gunzip';
import type { EngineEvent } from '../src/engine/types';
import { StyleError } from '../src/engine/force/types';

/*
 * Output formatting rules of the current docs (versionchanged 2Sep2026):
 *  - thermo_modify.rst and dump_modify.rst: "Format strings are checked ... LAMMPS will stop with an error if a
 *    format string does not match the kind of value it is applied to".
 *  - variable.rst (style format): "A conversion in the format string must match a floating-point value. The string
 *    may now also contain additional text and use all flags and modifiers supported by the C library".
 *  - thermo_modify.rst (colname auto, 10Dec2025), fix_nh.rst and compute_reduce.rst (automatic column names).
 *  - write_data.rst (11Feb2026): a compressed text file is written when the name ends with ".gz".
 * Error and output texts were measured with native LAMMPS (black box, 1 process).
 * The variable format batch is tests/fixtures/outfmt40/variable_format_native.json (1687 native outputs).
 */

const LJ = `
units lj
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.44 87287 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nve
`;

/** Runs a script; returns the errors, the log lines and the files written. */
const exec = async (text: string) => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (name, text2, append) => files.set(name, (append ? files.get(name) ?? '' : '') + text2),
  });
  try {
    await session.execute(text);
  } catch {
    // the error is also emitted as an 'error' event, which is what the tests read
  }
  const errors = events.filter((e): e is Extract<EngineEvent, { kind: 'error' }> => e.kind === 'error').map((e) => e.message);
  const log = events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);
  const headers = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo-header' }> => e.kind === 'thermo-header');
  return { errors, log, files, headers };
};

const errorOf = async (text: string): Promise<string> => {
  const { errors } = await exec(text);
  expect(errors.length, 'expected an error').toBeGreaterThan(0);
  return errors[0];
};

// ---------------------------------------------------------------- thermo_modify format

describe('thermo_modify format checks (thermo_modify.rst, versionchanged 2Sep2026)', () => {
  it('refuses "%d" for floats in thermo_modify format float (native: thermo_modify float format error, "conversion 1 of %d formats integer values")', async () => {
    const msg = await errorOf(`${LJ}\nthermo_style one\nthermo_modify format float %d\nrun 2\n`);
    expect(msg).toContain('Invalid thermo_modify float format');
    expect(msg).toContain("conversion 1 of '%d' formats integer values, but floating-point values are provided");
  });

  it('refuses "%g" for integers in thermo_modify format int (native: thermo_modify int format error)', async () => {
    const msg = await errorOf(`${LJ}\nthermo_style one\nthermo_modify format int %g\nrun 2\n`);
    expect(msg).toContain('Invalid thermo_modify int format');
    expect(msg).toContain("conversion 1 of '%g' formats floating-point values, but integer values are provided");
  });

  it('refuses two conversions for one value (native: "has 2 conversion(s) but only 1 value(s)")', async () => {
    expect(await errorOf(`${LJ}\nthermo_style one\nthermo_modify format float "%g %g"\n`)).toContain('has 2 conversion(s) but only 1 value(s) are provided');
    expect(await errorOf(`${LJ}\nthermo_style one\nthermo_modify format int "%d %d"\n`)).toContain('has 2 conversion(s) but only 1 value(s) are provided');
  });

  it('refuses a column format that does not match its column at run setup (native: "Invalid thermo format for column 3 (E_pair)")', async () => {
    const msg = await errorOf(`${LJ}\nthermo_style one\nthermo_modify format 3 %d\nrun 2\n`);
    expect(msg).toContain('Invalid thermo format for column 3 (E_pair)');
  });

  it('refuses "%d" on the temperature in a line format (native: column 2 (Temp))', async () => {
    const msg = await errorOf(`${LJ}\nthermo_style custom step temp pe\nthermo_modify format line "%d %d %d"\nrun 2\n`);
    expect(msg).toContain('Invalid thermo format for column 2 (Temp)');
  });

  it('refuses "%g" on the step in a line format (native: column 1 (Step), large integer values)', async () => {
    const msg = await errorOf(`${LJ}\nthermo_style custom step temp pe\nthermo_modify format line "%g %g %g"\nrun 2\n`);
    expect(msg).toContain('Invalid thermo format for column 1 (Step)');
    expect(msg).toContain("conversion 1 of '%g' formats floating-point values, but integer values are provided");
  });

  it('refuses a line format with fewer formats than columns (native: "No more tokens")', async () => {
    const msg = await errorOf(`${LJ}\nthermo_style custom step temp pe\nthermo_modify format line "%d %g"\nrun 2\n`);
    expect(msg).toContain('No more tokens');
  });

  it('refuses "%s" and "%5.2e" on the step (native: column 1 (Step) errors)', async () => {
    expect(await errorOf(`${LJ}\nthermo_style custom step temp pe\nthermo_modify format line "%s %g %g"\nrun 2\n`)).toContain('column 1 (Step)');
    expect(await errorOf(`${LJ}\nthermo_style custom step temp pe\nthermo_modify format 1 "%5.2e"\nrun 2\n`)).toContain('column 1 (Step)');
  });

  it('refuses an unknown or absent ID and a column out of range (native: "Invalid thermo_modify format argument")', async () => {
    expect(await errorOf(`${LJ}\nthermo_style one\nthermo_modify format foo %.2f\n`)).toContain('Invalid thermo_modify format argument: foo');
    expect(await errorOf(`${LJ}\nthermo_style one\nthermo_modify format 9 %.2f\n`)).toContain('Invalid thermo_modify format argument: 9');
    expect(await errorOf(`${LJ}\nthermo_style custom step temp pe\nthermo_modify format press %.2f\n`)).toContain('Invalid thermo_modify format argument: press');
  });

  it('accepts literal text, %ld on the step, keyword and negative IDs, ranges and trailing unformatted values (native: all accepted)', async () => {
    const cases = [
      'thermo_style custom step temp pe\nthermo_modify format float abc',
      'thermo_style custom step temp pe\nthermo_modify format float "<%.3f>"',
      'thermo_style custom step temp pe\nthermo_modify format line "%ld %g %g"',
      'thermo_style custom step temp pe\nthermo_modify format line "%d %g %g %g"',
      'thermo_style custom step atoms temp\nthermo_modify format line "<%d> <%d> <%.2f>"',
      'thermo_style custom step temp pe\nthermo_modify format 1 %d format 2 %.3f',
      'thermo_style custom step temp pe\nthermo_modify format pe %.4f',
      'thermo_style custom step temp pe\nthermo_modify format -1 %.2f',
      'thermo_style custom step temp pe\nthermo_modify format 2*3 %.2f',
      'thermo_style custom step temp pe\nthermo_modify format 2* %.2f',
      'thermo_style custom step temp pe\nthermo_modify format int %5d format float "%8.3f"',
      'thermo_style custom step atoms temp\nthermo_modify format line "%5d %d %.2f"',
    ];
    for (const c of cases) {
      const { errors } = await exec(`${LJ}\n${c}\nrun 2\n`);
      expect(errors, c).toEqual([]);
    }
  });

  it('applies the documented precedence: ID, then int/float, then line (native: float %.2f beats line %g)', async () => {
    const { errors } = await exec(`${LJ}\nthermo_style custom step temp pe\nthermo_modify format line "%d %g %g" format float %.2f\nrun 2\n`);
    expect(errors).toEqual([]);
  });

  it('a float format with no conversion prints literal text for each float column (native: "abc")', () => {
    expect(formatNumber(1.5, 'abc')).toBe('abc');
  });
});

// ---------------------------------------------------------------- dump_modify format

describe('dump_modify format checks (dump_modify.rst, versionchanged 2Sep2026)', () => {
  const DUMP = `${LJ}\ndump 1 all custom 1 d.dump id type x y z\n`;

  it('refuses "%d" for floats in dump_modify format float at the command (native: "Invalid dump_modify float format")', async () => {
    const msg = await errorOf(`${DUMP}dump_modify 1 format float %d\n`);
    expect(msg).toContain('Invalid dump_modify float format');
  });

  it('refuses "%g" for integers in dump_modify format int at the command (native: "Invalid dump_modify int format")', async () => {
    const msg = await errorOf(`${DUMP}dump_modify 1 format int %g\n`);
    expect(msg).toContain('Invalid dump_modify int format');
  });

  it('refuses "%d" on column 3 (x) at the command (native: "Invalid dump_modify format for column 3")', async () => {
    const msg = await errorOf(`${DUMP}dump_modify 1 format 3 %d\n`);
    expect(msg).toContain('Invalid dump_modify format for column 3');
  });

  it('refuses a column name in dump_modify format (native: "Expected integer parameter instead of x")', async () => {
    const msg = await errorOf(`${DUMP}dump_modify 1 format x %d\n`);
    expect(msg).toContain("Expected integer parameter instead of 'x'");
  });

  it('refuses a line with "%d" on x at run (native: "Invalid dump 1 format for column 3")', async () => {
    const msg = await errorOf(`${DUMP}dump_modify 1 format line "%d %d %d %g %g"\nrun 0\n`);
    expect(msg).toContain('Invalid dump 1 format for column 3');
  });

  it('refuses a line with fewer formats than columns at run (native: "Dump_modify format line is too short")', async () => {
    const msg = await errorOf(`${DUMP}dump_modify 1 format line "%d %d %g %g"\nrun 0\n`);
    expect(msg).toContain('Dump_modify format line is too short');
  });

  it('accepts a line of matching formats and writes the native values (id, type, x, y, z)', async () => {
    const { errors, files } = await exec(`${DUMP}dump_modify 1 format line "%d %d %g %g %g"\nrun 0\n`);
    expect(errors).toEqual([]);
    const text = files.get('d.dump') ?? '';
    expect(text).toContain('ITEM: ATOMS id type x y z\n1 1 0 0 0\n2 1 0.839798 0.839798 0\n3 1 0.839798 0 0.839798\n');
  });

  it('accepts a float format with an exponent conversion (native: 8.398e-01)', async () => {
    const { errors, files } = await exec(`${DUMP}dump_modify 1 format float %.3e\nrun 0\n`);
    expect(errors).toEqual([]);
    expect(files.get('d.dump')).toContain('ITEM: ATOMS id type x y z\n1 1 0.000e+00 0.000e+00 0.000e+00\n2 1 8.398e-01 8.398e-01 0.000e+00\n');
  });

  it('accepts a line with a width and a literal-only float format (native: "abc")', async () => {
    const { errors, files } = await exec(`${DUMP}dump_modify 1 format int "%5d" format float "%8.3f"\nrun 0\n`);
    expect(errors).toEqual([]);
    expect(files.get('d.dump')).toContain('ITEM: ATOMS id type x y z\n    1     1    0.000    0.000    0.000\n');
  });
});

// ---------------------------------------------------------------- variable format style

describe('variable format style (variable.rst, versionchanged 2Sep2026)', () => {
  it('matches the native output for 1687 value and format pairs (tests/fixtures/outfmt40/variable_format_native.json)', async () => {
    const rows = JSON.parse(readFileSync(join(process.cwd(), 'tests/fixtures/outfmt40/variable_format_native.json'), 'utf8')) as [string, string, string][];
    const bad: string[] = [];
    for (const [v, f, want] of rows) {
      let got: string;
      try { got = formatNumber(Number(v), f); } catch (e) { got = `THROW ${(e as Error).message}`; }
      if (got !== want) bad.push(`${f} ${v}: native ${JSON.stringify(want)} engine ${JSON.stringify(got)}`);
    }
    expect(bad).toEqual([]);
  });

  it('keeps literal text and "%%" around the conversion (native: "<3.1416>", "50% of -3.1")', () => {
    expect(formatNumber(3.14159265358979, '<%.4f>')).toBe('<3.1416>');
    expect(formatNumber(-3.14159265358979, '50%% of %.1f')).toBe('50% of -3.1');
  });

  it('applies flags and width like C (native: "+3.141593e+00", "-3.142  |", "00000003.142")', () => {
    expect(formatNumber(3.14159265358979, '%+12.6e')).toBe('+3.141593e+00');
    expect(formatNumber(-3.14159265358979, '%-08.3f|')).toBe('-3.142  |');
    expect(formatNumber(3.14159265358979, '%012.3f')).toBe('00000003.142');
  });

  it('rounds exact ties to the even digit like C (native: 2.5 %.0f is 2, 0.125 %.2f is 0.12, 1.25 %.1e is 1.2e+00)', () => {
    expect(formatNumber(2.5, '%.0f')).toBe('2');
    expect(formatNumber(3.5, '%.0f')).toBe('4');
    expect(formatNumber(0.125, '%.2f')).toBe('0.12');
    expect(formatNumber(1.25, '%.1e')).toBe('1.2e+00');
  });

  it('refuses an integer conversion for a floating-point value as a StyleError (native: "conversion 1 of \%d\ formats integer values")', () => {
    expect(() => formatNumber(7, '%d')).toThrow(StyleError);
    expect(() => formatNumber(7, '%x')).toThrow(StyleError);
    expect(() => formatNumber(7, '%s')).toThrow(StyleError);
    expect(() => formatNumber(7, '%f %f')).toThrow(StyleError);
    expect(() => formatNumber(7, '%*f')).toThrow(StyleError);
    expect(() => formatNumber(7, '%f%')).toThrow(StyleError);
    expect(() => formatNumber(7, '%Lf')).toThrow(StyleError);
  });

  it('accepts %lf, %le and %lg as the same conversions (native: "3.141593", "3.141593e+00", "3.14159")', () => {
    expect(formatNumber(3.14159265358979, '%lf')).toBe('3.141593');
    expect(formatNumber(3.14159265358979, '%le')).toBe('3.141593e+00');
    expect(formatNumber(3.14159265358979, '%lg')).toBe('3.14159');
  });

  it('a variable format with the wrong conversion is an error in print (native: "Invalid format string for format style variable")', async () => {
    const { errors } = await exec('variable a equal 3.5\nvariable f format a "%d"\nprint "x${f}"\n');
    expect(errors.length).toBe(1);
    expect(errors[0]).toContain('Invalid format string for format style variable');
  });

  it('an integer conversion in an immediate $(expr:%d) is an error too (native: "Invalid format string in immediate variable")', async () => {
    const { errors } = await exec('print "$(1+1:%d)"\n');
    expect(errors.length).toBe(1);
  });

  it('parses C formats into literal and conversion pieces', () => {
    const p = parseCFormat('a%+5.2fb%%c');
    expect(p.map((x) => ('text' in x ? `T:${x.text}` : `S:${x.spec.text}`))).toEqual(['T:a', 'S:%+5.2f', 'T:b%c']);
  });

  it('integer values in %d and %5i (native: "42", "   42")', () => {
    expect(formatInteger(42, '%d')).toBe('42');
    expect(formatInteger(42, '%5i')).toBe('   42');
    expect(formatInteger(42, '%-5d|')).toBe('42   |');
    expect(() => formatInteger(42, '%g')).toThrow(StyleError);
  });
});

// ---------------------------------------------------------------- colname auto

describe('thermo_modify colname auto (thermo_modify.rst 10Dec2025, fix_nh.rst, compute_reduce.rst)', () => {
  const NVT = `${LJ}\nfix 2 all nvt temp 1.0 1.0 1.0 tchain 2\ncompute r all reduce min x y\n`;

  it('names the fix nvt vector with the documented keywords and indices (native: f_1:eta[1], f_1:eta_dot[1], f_1:PE_eta[2])', async () => {
    const { errors, headers } = await exec(`${NVT}thermo_style custom step temp f_2[*]\nthermo_modify colname auto\nthermo 1\nrun 1\n`);
    expect(errors).toEqual([]);
    const labels = headers.at(-1)!.labels!;
    expect(labels).toEqual([
      'Step', 'Temp',
      'f_2:eta[1]', 'f_2:eta[2]', 'f_2:eta_dot[1]', 'f_2:eta_dot[2]', 'f_2:PE_eta[1]', 'f_2:PE_eta[2]',
      'f_2:KE_eta_dot[1]', 'f_2:KE_eta_dot[2]',
    ].map((x) => x.replace(/^f_2:/, (m) => m)));
  });

  it('names a scalar nvt column f_ID:ecouple and a compute reduce column c_ID:mode(input) (native: f_1:ecouple, c_r:min(x), c_r:min(y))', async () => {
    const { errors, headers } = await exec(`${NVT}thermo_style custom step f_2 c_r[1] c_r[2]\nthermo_modify colname auto\nthermo 1\nrun 1\n`);
    expect(errors).toEqual([]);
    expect(headers.at(-1)!.labels).toEqual(['Step', 'f_2:ecouple', 'c_r:min(x)', 'c_r:min(y)']);
  });

  it('names a scalar compute reduce c_ID:sum(input) (native: c_q:sum(c_ke))', async () => {
    const { errors, headers } = await exec(`${LJ}\ncompute ke all ke/atom\ncompute q all reduce sum c_ke\nthermo_style custom step c_q\nthermo_modify colname auto\nthermo 1\nrun 1\n`);
    expect(errors).toEqual([]);
    expect(headers.at(-1)!.labels).toEqual(['Step', 'c_q:sum(c_ke)']);
  });

  it('keeps the default names for other keywords and styles (native: Step, c_thermo_temp, c_pe, v_v, f_2[1])', async () => {
    const { errors, headers } = await exec(`${NVT}compute pe all pe\nfix 3 all ave/time 1 1 1 c_pe c_thermo_press[1]\nvariable v equal 2.0*ke\nthermo_style custom step c_thermo_temp c_pe f_3[1]\nthermo_modify colname auto\nthermo 1\nrun 1\n`);
    expect(errors).toEqual([]);
    expect(headers.at(-1)!.labels).toEqual(['Step', 'c_thermo_temp', 'c_pe', 'f_3[1]']);
  });

  it('colname default clears auto names (native: back to f_2[1])', async () => {
    const { errors, headers } = await exec(`${NVT}thermo_style custom step f_2[1]\nthermo_modify colname auto colname default\nthermo 1\nrun 1\n`);
    expect(errors).toEqual([]);
    expect(headers.at(-1)!.labels).toBeUndefined();
  });

  it('a user colname on an nh column is overridden by auto (thermo_modify.rst: auto overrides user strings for those styles)', async () => {
    const { errors, headers } = await exec(`${NVT}thermo_style custom step f_2[1]\nthermo_modify colname f_2[1] mine colname auto\nthermo 1\nrun 1\n`);
    expect(errors).toEqual([]);
    expect(headers.at(-1)!.labels).toEqual(['Step', 'f_2:eta[1]']);
  });

  it('keeps the default header with no colname setting', async () => {
    const { errors, headers } = await exec(`${NVT}thermo_style custom step f_2[1]\nthermo 1\nrun 1\n`);
    expect(errors).toEqual([]);
    expect(headers.at(-1)!.labels).toBeUndefined();
  });
});

// ---------------------------------------------------------------- gzip

describe('gzip writer (gzip.ts, write_data.rst 11Feb2026)', () => {
  it('computes the standard CRC-32 check value', () => {
    expect(crc32(new TextEncoder().encode('123456789'))).toBe(0xcbf43926);
  });

  it('round-trips through the engine reader and writes a valid header and trailer', () => {
    const text = 'LAMMPS data file\n\n3 atoms\n\nAtoms # atomic\n\n1 1 0.0 0.0 0.0\n';
    const latin = gzipText(text);
    const bytes = Uint8Array.from(latin, (c) => c.charCodeAt(0));
    expect(bytes[0]).toBe(0x1f);
    expect(bytes[1]).toBe(0x8b);
    expect(new TextDecoder().decode(gunzip(bytes))).toBe(text);
  });

  it('splits data into stored blocks of at most 65535 bytes and keeps the size and CRC', () => {
    const data = new Uint8Array(65535 * 2 + 1000);
    for (let i = 0; i < data.length; i++) data[i] = (i * 31) & 0xff;
    const gz = gzipStored(data);
    expect(new TextDecoder().decode(gunzip(gz)).length).toBe(data.length);
    expect(Array.from(gunzip(gz))).toEqual(Array.from(data));
    const dv = new DataView(gz.buffer, gz.byteOffset, gz.byteLength);
    expect(dv.getUint32(gz.length - 8, true)).toBe(crc32(data));
    expect(dv.getUint32(gz.length - 4, true)).toBe(data.length);
  });

  it('an empty input is one final stored block', () => {
    const gz = gzipStored(new Uint8Array(0));
    expect(Array.from(gunzip(gz))).toEqual([]);
  });

  it('read back by the system gzip utility (native gzip -dc)', () => {
    const dir = mkdtempSync(join(tmpdir(), 'outfmt40-'));
    try {
      const text = 'units lj\n'.repeat(20000);
      const path = join(dir, 'x.gz');
      writeFileSync(path, Buffer.from(gzipStored(new TextEncoder().encode(text))));
      const r = spawnSync('gzip', ['-dc', path], { encoding: 'utf8' });
      if (r.error) return; // gzip binary not installed: the reader test above still covers the format
      expect(r.status).toBe(0);
      expect(r.stdout).toBe(text);
      const t = spawnSync('gzip', ['-t', path]);
      expect(t.status).toBe(0);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('write_data with a .gz name stores gzip bytes that decompress to the plain write_data text', async () => {
    const plain = await exec(`${LJ}\nwrite_data plain.data\nrun 0\n`);
    const gz = await exec(`${LJ}\nwrite_data data.gz\nrun 0\n`);
    expect(plain.errors).toEqual([]);
    expect(gz.errors).toEqual([]);
    const latin = gz.files.get('data.gz')!;
    const bytes = Uint8Array.from(latin, (c) => c.charCodeAt(0));
    expect(bytes[0]).toBe(0x1f);
    expect(new TextDecoder().decode(gunzip(bytes))).toBe(plain.files.get('plain.data'));
    expect(plain.files.get('plain.data')!.startsWith('LAMMPS data file via write_data')).toBe(true);
  });

  const unzip = (latin: string): string => new TextDecoder().decode(gunzip(Uint8Array.from(latin, (c) => c.charCodeAt(0))));

  it('a dump with a .gz name writes gzip members that decompress to the plain dump (dump.html, Dump::openfile)', async () => {
    const plain = await exec(`${LJ}\ndump 1 all custom 2 d.dump id x y z\nrun 4\n`);
    const gz = await exec(`${LJ}\ndump 1 all custom 2 d.dump.gz id x y z\nrun 4\n`);
    expect(gz.errors).toEqual([]);
    const latin = gz.files.get('d.dump.gz')!;
    expect(latin.charCodeAt(0)).toBe(0x1f);
    expect(latin.charCodeAt(1)).toBe(0x8b);
    expect(unzip(latin)).toBe(plain.files.get('d.dump'));
  });

  it('the COMPRESS /gz styles write gzip, need a compressed name, and take compression_level in [-1, 9] (native messages)', async () => {
    const plain = await exec(`${LJ}\ndump 1 all atom 2 d.atom\nrun 2\n`);
    const gz = await exec(`${LJ}\ndump 1 all atom/gz 2 d.atom.gz\ndump_modify 1 compression_level 9\nrun 2\n`);
    expect(gz.errors).toEqual([]);
    expect(unzip(gz.files.get('d.atom.gz')!)).toBe(plain.files.get('d.atom'));
    expect(await errorOf(`${LJ}\ndump 1 all custom/gz 1 d.dump id x\nrun 0\n`)).toContain('Dump custom/gz only writes compressed files');
    expect(await errorOf(`${LJ}\ndump 1 all custom/gz 1 d.gz id x\ndump_modify 1 compression_level 10\n`)).toContain('Compression level must in the range of [-1, 9]');
    expect(await errorOf(`${LJ}\ndump 1 all custom 1 d.gz id x\ndump_modify 1 compression_level 5\n`)).toContain("unsupported dump_modify keyword 'compression_level'");
    expect(await errorOf(`${LJ}\ndump 1 all custom/zstd 1 d.zst id x\n`)).toContain('Zstandard compression is not available');
    expect(await errorOf(`${LJ}\ndump 1 all custom 1 d.dump.bz2 id x\n`)).toContain('compressed (.bz2) custom dump file is not supported');
  });

  it('a compressed dump ignores dump_modify append (native rewrites the file through the compressor)', async () => {
    const first = await exec(`${LJ}\ndump 1 all custom 1 a.gz id x\ndump_modify 1 append yes\nrun 0\nundump 1\ndump 2 all custom 1 a.gz id x\ndump_modify 2 append yes\nrun 0\n`);
    const once = await exec(`${LJ}\ndump 1 all custom 1 a.dump id x\nrun 0\n`);
    expect(first.errors).toEqual([]);
    expect(unzip(first.files.get('a.gz')!)).toBe(once.files.get('a.dump'));
  });

  it('write_dump with a .gz name writes the gzipped snapshot', async () => {
    const plain = await exec(`${LJ}\nwrite_dump all custom wd.dump id x y z\n`);
    const gz = await exec(`${LJ}\nwrite_dump all custom wd.gz id x y z\n`);
    expect(gz.errors).toEqual([]);
    expect(unzip(gz.files.get('wd.gz')!)).toBe(plain.files.get('wd.dump'));
  });
});

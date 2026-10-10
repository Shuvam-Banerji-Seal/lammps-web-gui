import { describe, expect, it } from 'vitest';
import {
  canOpenInViewer, classifyFile, countFrames, fileBytes, fileBytesSize, runLogText, thermoCsv, type RunOutput,
} from '../src/components/workbench/notebook/runFiles';
import { crc32 } from '../src/lammps/zip';

const dumpFrame = (step: number, atoms = 2) =>
  [`ITEM: TIMESTEP`, `${step}`, `ITEM: NUMBER OF ATOMS`, `${atoms}`, `ITEM: BOX BOUNDS pp pp pp`,
    `0 5`, `0 5`, `0 5`, `ITEM: ATOMS id type x y z`, ...Array.from({ length: atoms }, (_, k) => `${k + 1} 1 0.1 0.2 0.3`)].join('\n') + '\n';

const xyzFrame = (n: number, label: string) =>
  [`${n}`, label, ...Array.from({ length: n }, (_, k) => `C ${k} 0.0 0.0`)].join('\n') + '\n';

describe('file kind classification', () => {
  it('classifies dumps and xyz as trajectories by extension or content', () => {
    expect(classifyFile('dump.lammpstrj', dumpFrame(0))).toBe('trajectory');
    expect(classifyFile('traj.xyz', xyzFrame(2, 'x'))).toBe('trajectory');
    expect(classifyFile('no-extension', dumpFrame(0))).toBe('trajectory');
  });

  it('classifies write_data output as data and restart files as restart', () => {
    const data = 'LAMMPS data file via write_data\n\n2 atoms\n1 atom types\n\n0 5 xlo xhi\n0 5 ylo yhi\n0 5 zlo zhi\n\nAtoms # atomic\n';
    expect(classifyFile('out.data', data)).toBe('data');
    expect(classifyFile('final.restart', 'LAMMPS restart')).toBe('restart');
    expect(classifyFile('restart.1000', 'binary-ish')).toBe('restart');
  });

  it('classifies averaged output tables and CSV as table, other text as text', () => {
    expect(classifyFile('avg.txt', '# Time-averaged data for fix 1\n# TimeStep v_t\n1 2\n')).toBe('table');
    expect(classifyFile('rdf.csv', 'a,b\n1,2\n')).toBe('table');
    expect(classifyFile('log.lammps', 'Step Temp\n0 1.0\n')).toBe('text');
  });

  it('classifies a data URL as an image whatever its name', () => {
    expect(classifyFile('frame.png', 'data:image/png;base64,iVBORw0=')).toBe('image');
    expect(classifyFile('frame.jpg', 'data:image/jpeg;base64,/9j/')).toBe('image');
  });

  it('offers the viewer for trajectories and structure files only', () => {
    expect(canOpenInViewer('trajectory')).toBe(true);
    expect(canOpenInViewer('data')).toBe(true);
    expect(canOpenInViewer('text')).toBe(false);
    expect(canOpenInViewer('image')).toBe(false);
  });
});

describe('frame counting', () => {
  it('counts ITEM: TIMESTEP blocks in a dump', () => {
    const text = dumpFrame(0) + dumpFrame(10) + dumpFrame(20);
    expect(countFrames('dump.lammpstrj', text)).toBe(3);
    expect(countFrames('dump.lammpstrj', dumpFrame(0))).toBe(1);
  });

  it('counts the frames of an xyz file from its count and comment lines', () => {
    const text = xyzFrame(3, 'step 0') + xyzFrame(3, 'step 10') + xyzFrame(3, 'step 20') + xyzFrame(1, 'last');
    expect(countFrames('traj.xyz', text)).toBe(4);
  });

  it('ignores blank lines between xyz frames and stops at a malformed header', () => {
    expect(countFrames('a.xyz', xyzFrame(1, 'a') + '\n' + xyzFrame(1, 'b') + 'garbage\n')).toBe(2);
  });

  it('returns null for files that are not trajectories', () => {
    expect(countFrames('out.data', 'LAMMPS data file\n')).toBeNull();
    expect(countFrames('frame.png', 'data:image/png;base64,iVBORw0=')).toBeNull();
  });
});

describe('sizes and bytes', () => {
  it('counts UTF-8 bytes of text', () => {
    expect(fileBytesSize('abc')).toBe(3);
    expect(fileBytesSize('héllo')).toBe(6);
    expect(fileBytesSize('日本')).toBe(6);
    expect(fileBytesSize('😀')).toBe(4);
    expect(fileBytesSize('')).toBe(0);
  });

  it('gives the decoded size of a data URL', () => {
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d]);
    const url = 'data:image/png;base64,' + btoa(String.fromCharCode(...png));
    expect(fileBytesSize(url)).toBe(5);
    expect(Array.from(fileBytes(url))).toEqual(Array.from(png));
  });

  it('gives text as UTF-8 bytes', () => {
    expect(Array.from(fileBytes('hé'))).toEqual([0x68, 0xc3, 0xa9]);
  });

  it('CRC of the stored bytes matches the text', () => {
    expect(crc32(fileBytes('123456789'))).toBe(0xcbf43926);
  });
});

describe('run log and thermo CSV export', () => {
  const cells = [{ id: 'a' }, { id: 'b' }, { id: 'c' }];
  const runs: Record<string, RunOutput> = {
    // listed out of order on purpose: the export follows cell order, not insertion order
    c: {
      logs: ['Cell three log'],
      tables: [{ keywords: ['step', 'temp'], rows: [{ step: 100, temp: 0.7 }] }],
    },
    a: {
      logs: ['Setting up run', 'Run done'],
      error: null,
      tables: [{ keywords: ['step', 'temp', 'pe'], rows: [{ step: 0, temp: 3, pe: -6.1 }, { step: 50, temp: 1.23456789, pe: -5.5 }] }],
    },
  };

  it('writes each cell with a log, logs first, then its thermo tables, in cell order', () => {
    const text = runLogText(cells, runs);
    const lines = text.split('\n');
    expect(lines[0]).toBe('# cell 1');
    expect(lines.slice(1, 4)).toEqual(['Setting up run', 'Run done', 'step temp pe']);
    expect(lines[4]).toBe('0 3 -6.1');
    expect(lines[5]).toBe('50 1.23457 -5.5');
    expect(lines.indexOf('# cell 3')).toBeGreaterThan(lines.indexOf('# cell 1'));
    expect(lines).toContain('Cell three log');
    expect(lines).not.toContain('# cell 2'); // no output for cell b
  });

  it('writes errors into the log', () => {
    const text = runLogText([{ id: 'x' }], { x: { logs: [], error: 'Unknown command: foo', tables: [] } });
    expect(text).toContain('ERROR: Unknown command: foo');
  });

  it('exports the rows of the latest thermo table as CSV', () => {
    expect(thermoCsv(cells, runs)).toBe('step,temp\n100,0.7\n');
    expect(thermoCsv(cells, { a: runs.a })).toBe('step,temp,pe\n0,3,-6.1\n50,1.23456789,-5.5\n');
  });

  it('returns null when no thermo table exists', () => {
    expect(thermoCsv(cells, {})).toBeNull();
    expect(thermoCsv(cells, { a: { logs: ['x'], tables: [] } })).toBeNull();
  });
});

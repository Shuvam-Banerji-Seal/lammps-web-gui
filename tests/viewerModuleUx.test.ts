import { describe, expect, it, vi } from 'vitest';

// ViewerModule pulls in the WebGL canvas through MoleculeCanvas; the pure
// helpers under test need none of it, so keep the import graph light.
vi.mock('../src/components/MoleculeCanvas', () => ({ default: () => null }));

import {
  SIDEBAR_MAX_WIDTH,
  SIDEBAR_MIN_WIDTH,
  chooseUploadFormat,
  clampSidebarWidth,
  nextTabIndex,
  stepSidebarWidth,
} from '../src/components/workbench/ViewerModule';

describe('clampSidebarWidth (sidebar resize clamp)', () => {
  it('keeps in-range values unchanged', () => {
    expect(clampSidebarWidth(384)).toBe(384);
    expect(clampSidebarWidth(SIDEBAR_MIN_WIDTH)).toBe(SIDEBAR_MIN_WIDTH);
    expect(clampSidebarWidth(SIDEBAR_MAX_WIDTH)).toBe(SIDEBAR_MAX_WIDTH);
  });

  it('clamps below the 280px minimum', () => {
    expect(clampSidebarWidth(0)).toBe(280);
    expect(clampSidebarWidth(-100)).toBe(280);
    expect(clampSidebarWidth(279)).toBe(280);
  });

  it('clamps above the 560px maximum', () => {
    expect(clampSidebarWidth(1200)).toBe(560);
    expect(clampSidebarWidth(561)).toBe(560);
  });

  it('rounds fractional pointer positions', () => {
    expect(clampSidebarWidth(300.4)).toBe(300);
    expect(clampSidebarWidth(300.6)).toBe(301);
  });
});

describe('stepSidebarWidth (keyboard resize, 16px steps)', () => {
  it('widens by 16px on ArrowRight within the clamp', () => {
    expect(stepSidebarWidth(384, 'ArrowRight')).toBe(400);
  });

  it('narrows by 16px on ArrowLeft within the clamp', () => {
    expect(stepSidebarWidth(384, 'ArrowLeft')).toBe(368);
  });

  it('stops at the maximum on ArrowRight', () => {
    expect(stepSidebarWidth(560, 'ArrowRight')).toBe(560);
    expect(stepSidebarWidth(552, 'ArrowRight')).toBe(560);
  });

  it('stops at the minimum on ArrowLeft', () => {
    expect(stepSidebarWidth(280, 'ArrowLeft')).toBe(280);
    expect(stepSidebarWidth(288, 'ArrowLeft')).toBe(280);
  });

  it('ignores unrelated keys', () => {
    expect(stepSidebarWidth(384, 'Home')).toBe(384);
    expect(stepSidebarWidth(384, 'ArrowDown')).toBe(384);
  });
});

describe('nextTabIndex (sidebar tablist roving index)', () => {
  const count = 5;

  it('moves cyclically forward on ArrowRight', () => {
    expect(nextTabIndex(0, count, 'ArrowRight')).toBe(1);
    expect(nextTabIndex(2, count, 'ArrowRight')).toBe(3);
    expect(nextTabIndex(4, count, 'ArrowRight')).toBe(0);
  });

  it('moves cyclically backward on ArrowLeft', () => {
    expect(nextTabIndex(2, count, 'ArrowLeft')).toBe(1);
    expect(nextTabIndex(1, count, 'ArrowLeft')).toBe(0);
    expect(nextTabIndex(0, count, 'ArrowLeft')).toBe(4);
  });

  it('Home jumps to the first and End to the last tab', () => {
    expect(nextTabIndex(3, count, 'Home')).toBe(0);
    expect(nextTabIndex(0, count, 'Home')).toBe(0);
    expect(nextTabIndex(1, count, 'End')).toBe(4);
    expect(nextTabIndex(4, count, 'End')).toBe(4);
  });

  it('leaves the index alone for other keys', () => {
    expect(nextTabIndex(2, count, 'ArrowDown')).toBe(2);
    expect(nextTabIndex(2, count, 'Escape')).toBe(2);
  });

  it('normalises a negative or overflowing current index', () => {
    expect(nextTabIndex(-1, count, 'ArrowRight')).toBe(0);
    expect(nextTabIndex(7, count, 'ArrowRight')).toBe(3);
  });

  it('returns -1 for an empty tab list', () => {
    expect(nextTabIndex(0, 0, 'ArrowRight')).toBe(-1);
  });
});

describe('chooseUploadFormat (upload format detection)', () => {
  const xyz = '3\nwater\nO 0.0 0.0 0.0\nH 0.9572 0.0 0.0\nH -0.24 0.927 0.0\n';

  it('sniffs a generic .txt upload as XYZ from its content', () => {
    expect(chooseUploadFormat('molecule.txt', xyz)).toBe('xyz');
  });

  it('sniffs unknown extensions from their content', () => {
    expect(chooseUploadFormat('molecule.str', xyz)).toBe('xyz');
    expect(chooseUploadFormat('structure', xyz)).toBe('xyz');
  });

  it('keeps known extensions authoritative regardless of content', () => {
    expect(chooseUploadFormat('protein.pdb', xyz)).toBe('pdb');
    expect(chooseUploadFormat('run.data', xyz)).toBe('lammps');
    expect(chooseUploadFormat('run.lmp', xyz)).toBe('lammps');
    expect(chooseUploadFormat('traj.lammpstrj', xyz)).toBe('lammpsdump');
    expect(chooseUploadFormat('cell.cif', xyz)).toBe('cif');
    expect(chooseUploadFormat('frames.dump', xyz)).toBe('lammpsdump');
  });

  it('detects a LAMMPS dump hidden behind a generic name', () => {
    const dump = 'ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\n';
    expect(chooseUploadFormat('traj.txt', dump)).toBe('lammpsdump');
  });

  it('falls back to LAMMPS when nothing matches', () => {
    expect(chooseUploadFormat('mystery.bin', 'hello world\n')).toBe('lammps');
    expect(chooseUploadFormat('empty.txt', '')).toBe('lammps');
  });
});
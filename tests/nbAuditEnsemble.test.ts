/**
 * Regression (notebook audit): a run inherits the fixes that earlier cells defined. A `run` cell with no fix of
 * its own, after `fix 1 all nve` in an earlier cell, is NVE (the statistics hid the energy drift as "unknown").
 * A fix stays defined until `unfix`, and a markdown cell's text is not a command.
 */
import { describe, expect, it } from 'vitest';
import { ensembleOf, ensembleOfRun } from '../src/components/workbench/notebook/chartData';

describe('ensembleOfRun: the fixes a run inherits from earlier cells', () => {
  it('a run after an nve fix in an earlier cell is NVE', () => {
    expect(ensembleOfRun(['units lj\nfix 1 all nve', 'run 250'])).toBe('NVE');
  });

  it('a run with no fixes anywhere is unknown', () => {
    expect(ensembleOfRun(['units lj', 'run 250'])).toBe('unknown');
  });

  it('a later fix replaces an earlier one with the same id', () => {
    expect(ensembleOfRun(['fix 1 all nvt temp 1 1 1', 'fix 1 all nve', 'run 10'])).toBe('NVE');
  });

  it('unfix removes the fix, so the run after it is not that ensemble', () => {
    expect(ensembleOfRun(['fix 1 all nvt temp 1 1 1', 'unfix 1', 'fix 2 all nve', 'run 10'])).toBe('NVE');
    expect(ensembleOfRun(['fix 1 all npt temp 1 1 1 iso 1 1 1', 'unfix 1', 'run 10'])).toBe('unknown');
  });

  it('a thermostat fix in the history makes the run NVT even after an nve fix', () => {
    expect(ensembleOfRun(['fix 1 all nve', 'fix 2 all langevin 1 1 1 42', 'run 10'])).toBe('NVT');
  });

  it('NPT outranks NVT and NVE, as before', () => {
    expect(ensembleOfRun(['fix 1 all nvt temp 1 1 1', 'fix 2 all npt temp 1 1 1 iso 1 1 1', 'run 10'])).toBe('NPT');
  });

  it('a single cell gives the same answer as ensembleOf', () => {
    const s = 'fix 1 all nvt temp 300 300 0.1\nrun 100';
    expect(ensembleOfRun([s])).toBe(ensembleOf(s));
    expect(ensembleOf(s)).toBe('NVT');
  });
});

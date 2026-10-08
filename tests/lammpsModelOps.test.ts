import { describe, it, expect } from 'vitest';
import {
  addBranch, takeBranchAtFork, promoteBranch, removeBranch, updateBranch,
  findLane, laneSteps, pathIndexToLane, insertStepAtPathIndex,
  moveStepToPathIndex, moveStepInModel, removeStepFromModel,
  duplicateStepInModel, updateParamInModel, appendToLane, branchesByFork,
} from '../src/lammps/model';
import { resolvePath } from '../src/lammps/generator';
import { COMMAND_BY_ID, defaultParams, ScriptModel, ScriptStep } from '../src/lammps/catalog';

let n = 0;
const step = (defId: string, o: Record<string, string> = {}): ScriptStep => ({
  uid: `${defId}-${++n}`,
  defId,
  params: { ...defaultParams(COMMAND_BY_ID[defId]), ...o },
  enabled: true,
});

const base = (): ScriptModel => ({
  title: 'T',
  steps: [step('units'), step('fix_nve'), step('run', { steps: '100' })],
});

describe('branch lifecycle', () => {
  it('forking seeds the branch with a copy of the trunk tail', () => {
    const m = base();
    const { model, branch } = addBranch(m, m.steps[0].uid);
    expect(branch.steps).toHaveLength(2);
    expect(branch.steps.map(s => s.defId)).toEqual(['fix_nve', 'run']);
    // copies, not aliases — editing the branch must not touch the trunk
    expect(branch.steps[0].uid).not.toBe(m.steps[1].uid);
    expect(model.activeBranchIds).toEqual([branch.id]);
  });

  it('forking with seed:empty starts blank', () => {
    const m = base();
    const { branch } = addBranch(m, m.steps[0].uid, { seed: 'empty' });
    expect(branch.steps).toEqual([]);
  });

  it('a second fork at the same point replaces the taken one', () => {
    const m = base();
    const first = addBranch(m, m.steps[0].uid);
    const second = addBranch(first.model, m.steps[0].uid);
    expect(second.model.activeBranchIds).toEqual([second.branch.id]);
    expect(second.model.branches).toHaveLength(2);
  });

  it('labels siblings Concept A, B, C…', () => {
    const m = base();
    const a = addBranch(m, m.steps[0].uid);
    const b = addBranch(a.model, m.steps[0].uid);
    expect([a.branch.label, b.branch.label]).toEqual(['Concept A', 'Concept B']);
  });

  it('takeBranchAtFork(null) returns to the main line', () => {
    const m = base();
    const { model, branch } = addBranch(m, m.steps[0].uid);
    const back = takeBranchAtFork(model, branch.forkAfter, null);
    expect(back.activeBranchIds).toEqual([]);
    expect(resolvePath(back)).toHaveLength(3);
  });

  it('promoting a branch rewrites the trunk and clears that fork', () => {
    const m = base();
    const { model, branch } = addBranch(m, m.steps[0].uid, { seed: 'empty' });
    const withStep = { ...model, branches: model.branches!.map(b =>
      b.id === branch.id ? { ...b, steps: [step('fix_npt')] } : b) };
    const promoted = promoteBranch(withStep, branch.id);
    expect(promoted.steps.map(s => s.defId)).toEqual(['units', 'fix_npt']);
    expect(promoted.branches).toHaveLength(0);
    expect(promoted.activeBranchIds).toEqual([]);
  });

  it('promoting a rejoining branch keeps the trunk tail', () => {
    const m = base();
    const { model, branch } = addBranch(m, m.steps[0].uid, { seed: 'empty', rejoin: true });
    const withStep = updateBranch(model, branch.id, { steps: [step('thermo')] });
    const promoted = promoteBranch(withStep, branch.id);
    expect(promoted.steps.map(s => s.defId)).toEqual(['units', 'thermo', 'fix_nve', 'run']);
  });

  it('removing a trunk fork anchor re-anchors its branches to the step before', () => {
    const m = base();
    const { model, branch } = addBranch(m, m.steps[1].uid, { seed: 'empty' });
    const pruned = removeStepFromModel(model, m.steps[1].uid);
    expect(pruned.branches!.find(b => b.id === branch.id)!.forkAfter).toBe(m.steps[0].uid);
  });

  it('removing the FIRST trunk step re-anchors its branches to the start', () => {
    const m = base();
    const { model, branch } = addBranch(m, m.steps[0].uid, { seed: 'empty' });
    const pruned = removeStepFromModel(model, m.steps[0].uid);
    expect(pruned.branches!.find(b => b.id === branch.id)!.forkAfter).toBeNull();
  });

  it('removeBranch drops it from the active set too', () => {
    const m = base();
    const { model, branch } = addBranch(m, m.steps[0].uid);
    const gone = removeBranch(model, branch.id);
    expect(gone.branches).toHaveLength(0);
    expect(gone.activeBranchIds).toEqual([]);
  });

  it('branchesByFork groups branches by their anchor', () => {
    const m = base();
    const a = addBranch(m, m.steps[0].uid);
    const b = addBranch(a.model, m.steps[0].uid);
    expect(branchesByFork(b.model).get(m.steps[0].uid)).toHaveLength(2);
  });
});

describe('lane-aware step edits', () => {
  const forked = () => {
    const m = base();
    const { model, branch } = addBranch(m, m.steps[0].uid);
    return { m, model, branch };
  };

  it('findLane locates trunk and branch steps', () => {
    const { m, model, branch } = forked();
    expect(findLane(model, m.steps[0].uid)).toBeNull();
    expect(findLane(model, branch.steps[0].uid)).toBe(branch.id);
    expect(findLane(model, 'nope')).toBeUndefined();
  });

  it('editing a branch step leaves the trunk copy alone', () => {
    const { m, model, branch } = forked();
    const edited = updateParamInModel(model, branch.steps[1].uid, 'steps', '9999');
    expect(laneSteps(edited, branch.id)[1].params.steps).toBe('9999');
    expect(edited.steps[2].params.steps).toBe('100');
    void m;
  });

  it('pathIndexToLane maps the end of the path into the branch', () => {
    const { model, branch } = forked();
    const path = resolvePath(model);
    expect(pathIndexToLane(model, path.length)).toEqual({
      lane: branch.id, laneIndex: 2,
    });
    expect(pathIndexToLane(model, 0)).toEqual({ lane: null, laneIndex: 0 });
  });

  it('inserting at a path index inside the branch lands in that branch', () => {
    const { model, branch } = forked();
    const fresh = step('thermo');
    const next = insertStepAtPathIndex(model, fresh, resolvePath(model).length);
    expect(laneSteps(next, branch.id).map(s => s.uid)).toContain(fresh.uid);
    expect(next.steps.map(s => s.uid)).not.toContain(fresh.uid);
  });

  it('dragging a step out of a branch into the trunk moves lanes', () => {
    const { model, branch } = forked();
    const moved = moveStepToPathIndex(model, branch.steps[1].uid, 0);
    expect(findLane(moved, branch.steps[1].uid)).toBeNull();
    expect(laneSteps(moved, branch.id)).toHaveLength(1);
    expect(moved.steps[0].uid).toBe(branch.steps[1].uid);
  });

  it('reordering inside a lane keeps the lane', () => {
    const { model, branch } = forked();
    const moved = moveStepInModel(model, branch.steps[1].uid, -1);
    expect(laneSteps(moved, branch.id).map(s => s.defId)).toEqual(['run', 'fix_nve']);
  });

  it('duplicating a branch step stays in the branch', () => {
    const { model, branch } = forked();
    const dup = duplicateStepInModel(model, branch.steps[0].uid);
    expect(laneSteps(dup, branch.id)).toHaveLength(3);
    expect(dup.steps).toHaveLength(3);
  });

  it('appendToLane groups by section inside a branch', () => {
    const { model, branch } = forked();
    const next = appendToLane(model, branch.id, step('fix_langevin'));
    // fix_langevin is `control`, same as fix_nve/run → lands after the last one
    expect(laneSteps(next, branch.id).map(s => s.defId)).toEqual(
      ['fix_nve', 'run', 'fix_langevin'],
    );
  });

  it('unknown uids are no-ops rather than throwing', () => {
    const m = base();
    expect(removeStepFromModel(m, 'ghost')).toBe(m);
    expect(moveStepInModel(m, 'ghost', 1)).toBe(m);
    expect(duplicateStepInModel(m, 'ghost')).toBe(m);
    expect(moveStepToPathIndex(m, 'ghost', 0)).toBe(m);
  });
});

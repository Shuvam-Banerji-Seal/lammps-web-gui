import { describe, it, expect } from 'vitest';
import {
  addBranch, findStepInModel, canForkAfter, updateStepInModel,
} from '../src/lammps/model';
import { resolvePath } from '../src/lammps/generator';
import { COMMAND_BY_ID, defaultParams, ScriptModel, ScriptStep } from '../src/lammps/catalog';

// Mirrors the ScriptBuilder's lane-aware UX contracts: selection, toggling,
// connection-menu indexing and fork anchoring must all work on the RESOLVED
// path, not just the trunk.
let n = 0;
const step = (defId: string, o: Record<string, string> = {}): ScriptStep => ({
  uid: `sbux-${defId}-${++n}`,
  defId,
  params: { ...defaultParams(COMMAND_BY_ID[defId]), ...o },
  enabled: true,
});

const base = (): ScriptModel => ({
  title: 'T',
  steps: [step('units'), step('fix_nve'), step('run', { steps: '100' })],
});

describe('lane-aware selection (selectedStep / card eye toggle)', () => {
  it('the selection helper finds a step that lives inside a branch', () => {
    const m = base();
    const { model, branch } = addBranch(m, m.steps[0].uid);
    const branchStep = branch.steps[0];
    // The trunk-only lookup misses this; findStepInModel returns the branch step.
    expect(findStepInModel(model, branchStep.uid)).toBe(branchStep);
    expect(model.steps.some(s => s.uid === branchStep.uid)).toBe(false);
    // Trunk steps and unknown uids still behave.
    expect(findStepInModel(model, m.steps[1].uid)?.uid).toBe(m.steps[1].uid);
    expect(findStepInModel(model, 'missing')).toBeUndefined();
  });

  it('toggling a branch step flips its enabled flag inside model.branches', () => {
    const m = base();
    const { model, branch } = addBranch(m, m.steps[0].uid);
    const uid = branch.steps[1].uid;
    const next = updateStepInModel(model, uid, { enabled: !findStepInModel(model, uid)?.enabled });
    expect(next.branches!.find(b => b.id === branch.id)!.steps[1].enabled).toBe(false);
    // The trunk is untouched by a branch-lane toggle.
    expect(next.steps.every(s => s.enabled)).toBe(true);
  });
});

describe('connection-menu path indexing', () => {
  it('edge menu index i acts on pathSteps[i], not trunk steps[i], when a concept is taken', () => {
    const m = base();
    const { model, branch } = addBranch(m, m.steps[0].uid, { seed: 'empty' });
    const withStep: ScriptModel = {
      ...model,
      branches: model.branches!.map(b =>
        b.id === branch.id ? { ...b, steps: [step('fix_npt')] } : b),
    };
    const pathSteps = resolvePath(withStep).map(r => r.step);
    // With Concept A taken after `units`, path position 1 is the branch's
    // step; the trunk's fix_nve is cut off. The menu item labelled for
    // position 1 must target THIS uid — the one displayed on the card.
    expect(pathSteps[1].defId).toBe('fix_npt');
    expect(pathSteps[1].uid).not.toBe(withStep.steps[1].uid);
    expect(findStepInModel(withStep, pathSteps[1].uid)?.defId).toBe('fix_npt');
  });
});

describe('fork anchoring is trunk-only', () => {
  it('a branch uid is refused; the start and trunk uids are accepted', () => {
    const m = base();
    const { model, branch } = addBranch(m, m.steps[0].uid);
    expect(canForkAfter(model, null)).toBe(true);
    expect(canForkAfter(model, m.steps[0].uid)).toBe(true);
    expect(canForkAfter(model, branch.steps[0].uid)).toBe(false);
    expect(canForkAfter(model, 'unknown-uid')).toBe(false);
  });

  it('refusing the branch uid keeps the model untouched (no unreachable concept)', () => {
    const m = base();
    const { model, branch } = addBranch(m, m.steps[0].uid);
    const branchUid = branch.steps[0].uid;
    if (!canForkAfter(model, branchUid)) {
      // the guarded forkHere path: no addBranch call happens
      expect(model.branches).toHaveLength(1);
      expect(model.branches![0].forkAfter).toBe(m.steps[0].uid);
    } else {
      throw new Error('branch uid must not be forkable');
    }
  });
});

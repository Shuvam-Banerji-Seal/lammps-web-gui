import { describe, it, expect } from 'vitest';
import {
  generateScript,
  deriveFlowchart,
  resolvePath,
  skippedTrunkUids,
  forkPoints,
  takenBranchAt,
} from '../src/lammps/generator';
import {
  COMMAND_BY_ID,
  defaultParams,
  ScriptBranch,
  ScriptModel,
  ScriptStep,
} from '../src/lammps/catalog';

let n = 0;
const step = (defId: string, overrides: Record<string, string> = {}): ScriptStep => {
  const def = COMMAND_BY_ID[defId];
  return {
    uid: `${defId}-${++n}`,
    defId,
    params: { ...defaultParams(def), ...overrides },
    enabled: true,
  };
};

/** trunk: units → fix nve → run 1000 */
const trunk = () => {
  const s = [
    step('units', { style: 'lj' }),
    step('fix_nve'),
    step('run', { steps: '1000' }),
  ];
  return s;
};

const branch = (over: Partial<ScriptBranch> & { id: string }): ScriptBranch => ({
  label: 'Variant',
  forkAfter: null,
  steps: [],
  rejoin: false,
  ...over,
});

describe('concept branching', () => {
  it('follows the trunk when no branch is taken', () => {
    const steps = trunk();
    const m: ScriptModel = {
      title: 'T',
      steps,
      branches: [branch({ id: 'b1', forkAfter: steps[1].uid, steps: [step('fix_npt')] })],
      activeBranchIds: [],
    };
    expect(resolvePath(m).map(r => r.step.uid)).toEqual(steps.map(s => s.uid));
    expect(generateScript(m).text).toContain('run 1000');
    expect(generateScript(m).text).not.toContain('npt');
  });

  it('a divergent branch REPLACES the trunk tail', () => {
    const steps = trunk();
    const alt = step('run', { steps: '9999' });
    const m: ScriptModel = {
      title: 'T',
      steps,
      branches: [branch({ id: 'b1', label: 'Long run', forkAfter: steps[1].uid, steps: [alt] })],
      activeBranchIds: ['b1'],
    };
    const path = resolvePath(m).map(r => r.step.uid);
    expect(path).toEqual([steps[0].uid, steps[1].uid, alt.uid]);
    const text = generateScript(m).text;
    expect(text).toContain('run 9999');
    expect(text).not.toContain('run 1000');
    expect(text).toContain('# Branch: Long run');
  });

  it('a rejoining branch is a detour — the trunk resumes after it', () => {
    const steps = trunk();
    const extra = step('thermo', { n: '25' });
    const m: ScriptModel = {
      title: 'T',
      steps,
      branches: [
        branch({ id: 'b1', label: 'Verbose', forkAfter: steps[1].uid, steps: [extra], rejoin: true }),
      ],
      activeBranchIds: ['b1'],
    };
    expect(resolvePath(m).map(r => r.step.uid)).toEqual([
      steps[0].uid, steps[1].uid, extra.uid, steps[2].uid,
    ]);
    const text = generateScript(m).text;
    expect(text).toContain('thermo 25');
    expect(text).toContain('run 1000');
    expect(text).toContain('rejoins main line');
  });

  it('forking before the first step replaces the whole script', () => {
    const steps = trunk();
    const alt = step('units', { style: 'metal' });
    const m: ScriptModel = {
      title: 'T',
      steps,
      branches: [branch({ id: 'b1', forkAfter: null, steps: [alt] })],
      activeBranchIds: ['b1'],
    };
    expect(resolvePath(m).map(r => r.step.uid)).toEqual([alt.uid]);
    expect(generateScript(m).text).toContain('units metal');
    expect(generateScript(m).text).not.toContain('units lj');
  });

  it('only one branch per fork point is honoured', () => {
    const steps = trunk();
    const a = step('fix_nvt');
    const b = step('fix_npt');
    const m: ScriptModel = {
      title: 'T',
      steps,
      branches: [
        branch({ id: 'a', label: 'NVT', forkAfter: steps[0].uid, steps: [a] }),
        branch({ id: 'b', label: 'NPT', forkAfter: steps[0].uid, steps: [b] }),
      ],
      activeBranchIds: ['a', 'b'],
    };
    const uids = resolvePath(m).map(r => r.step.uid);
    expect(uids).toContain(a.uid);
    expect(uids).not.toContain(b.uid);
    expect(takenBranchAt(m, steps[0].uid)?.id).toBe('a');
  });

  it('two sequential fork points compose when the first rejoins', () => {
    const steps = trunk();
    const d1 = step('thermo', { n: '10' });
    const d2 = step('write_data_out', { file: 'final.data' });
    const m: ScriptModel = {
      title: 'T',
      steps,
      branches: [
        branch({ id: 'a', forkAfter: steps[0].uid, steps: [d1], rejoin: true }),
        branch({ id: 'b', forkAfter: steps[2].uid, steps: [d2] }),
      ],
      activeBranchIds: ['a', 'b'],
    };
    expect(resolvePath(m).map(r => r.step.uid)).toEqual([
      steps[0].uid, d1.uid, steps[1].uid, steps[2].uid, d2.uid,
    ]);
    expect(forkPoints(m)).toEqual([steps[0].uid, steps[2].uid]);
  });

  it('reports which trunk steps a divergent branch cut off', () => {
    const steps = trunk();
    const m: ScriptModel = {
      title: 'T',
      steps,
      branches: [branch({ id: 'b1', forkAfter: steps[0].uid, steps: [step('fix_npt')] })],
      activeBranchIds: ['b1'],
    };
    expect([...skippedTrunkUids(m)].sort()).toEqual([steps[1].uid, steps[2].uid].sort());
  });

  it('tags flowchart nodes with their branch and marks the fork edge', () => {
    const steps = trunk();
    const alt = step('fix_npt');
    const m: ScriptModel = {
      title: 'T',
      steps,
      branches: [branch({ id: 'b1', label: 'NPT', forkAfter: steps[1].uid, steps: [alt] })],
      activeBranchIds: ['b1'],
    };
    const g = deriveFlowchart(m);
    const node = g.nodes.find(nd => nd.uid === alt.uid)!;
    expect(node.branchId).toBe('b1');
    expect(node.branchLabel).toBe('NPT');
    expect(g.edges.find(e => e.to === alt.uid)?.fork).toBe(true);
    expect(g.forks).toEqual([{ afterUid: steps[1].uid, takenBranchId: 'b1', total: 1 }]);
  });

  it('a branch with no steps and no rejoin truncates the script', () => {
    const steps = trunk();
    const m: ScriptModel = {
      title: 'T',
      steps,
      branches: [branch({ id: 'b1', forkAfter: steps[0].uid, steps: [] })],
      activeBranchIds: ['b1'],
    };
    expect(resolvePath(m).map(r => r.step.uid)).toEqual([steps[0].uid]);
  });

  it('an active id that matches no branch is ignored', () => {
    const steps = trunk();
    const m: ScriptModel = { title: 'T', steps, branches: [], activeBranchIds: ['ghost'] };
    expect(resolvePath(m)).toHaveLength(3);
  });
});

/**
 * Persistence hardening: a stored workspace can outlive the step a branch
 * forked after. The revive path must not resurrect a branch that can never be
 * spliced in, and must not leave two branches taken at one fork point.
 */
describe('branch invariants that revive must preserve', () => {
  it('a branch anchored to a missing step can never be taken', () => {
    const steps = trunk();
    const m: ScriptModel = {
      title: 'T',
      steps,
      branches: [branch({ id: 'orphan', forkAfter: 'step-that-was-deleted', steps: [step('fix_npt')] })],
      activeBranchIds: ['orphan'],
    };
    // resolvePath simply never reaches it — the whole trunk is emitted.
    expect(resolvePath(m).map(r => r.step.uid)).toEqual(steps.map(s => s.uid));
  });

  it('only the first active branch at a fork point is honoured', () => {
    const steps = trunk();
    const a = step('fix_nvt');
    const b = step('fix_npt');
    const m: ScriptModel = {
      title: 'T',
      steps,
      branches: [
        branch({ id: 'a', forkAfter: steps[0].uid, steps: [a] }),
        branch({ id: 'b', forkAfter: steps[0].uid, steps: [b] }),
      ],
      activeBranchIds: ['b', 'a'],   // both taken at one fork — invalid state
    };
    const uids = resolvePath(m).map(r => r.step.uid);
    // exactly one of them appears, never both
    expect([uids.includes(a.uid), uids.includes(b.uid)].filter(Boolean)).toHaveLength(1);
  });
});

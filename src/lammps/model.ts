/**
 * Lane-aware edits for a branching ScriptModel.
 *
 * A "lane" is a list of steps: the trunk (`null`) or one ScriptBranch (its id).
 * The flowchart shows the RESOLVED path — trunk steps with the taken branch
 * spliced in — so every index the UI hands us is a path index. These helpers
 * translate path indices to (lane, index-in-lane) and apply the edit there, so
 * dragging a card, inserting at a connector or deleting a step all behave the
 * same whether the user is working on the trunk or inside a branch.
 */

import {
  COMMAND_BY_ID,
  ScriptBranch,
  ScriptModel,
  ScriptStep,
  newBranchId,
} from './catalog';
import { branchesOf, resolvePath, takenBranchAt } from './generator';

/** `null` = the trunk; otherwise a ScriptBranch id. */
export type LaneId = string | null;

export const laneSteps = (model: ScriptModel, lane: LaneId): ScriptStep[] =>
  lane === null
    ? model.steps
    : branchesOf(model).find(b => b.id === lane)?.steps ?? [];

export const laneLabel = (model: ScriptModel, lane: LaneId): string =>
  lane === null
    ? 'Main line'
    : branchesOf(model).find(b => b.id === lane)?.label ?? 'Branch';

/** Replace one lane's steps, leaving everything else untouched. */
export const setLaneSteps = (
  model: ScriptModel,
  lane: LaneId,
  steps: ScriptStep[],
): ScriptModel =>
  lane === null
    ? { ...model, steps }
    : {
        ...model,
        branches: branchesOf(model).map(b => (b.id === lane ? { ...b, steps } : b)),
      };

/** Which lane owns a step uid (undefined when the uid is unknown). */
export const findLane = (model: ScriptModel, uid: string): LaneId | undefined => {
  if (model.steps.some(s => s.uid === uid)) return null;
  const b = branchesOf(model).find(br => br.steps.some(s => s.uid === uid));
  return b ? b.id : undefined;
};

/**
 * Map a position in the resolved path onto the lane that owns it.
 *
 * `index` is an INSERTION point: 0 = before the first card, `path.length` =
 * after the last one. Appending lands in the lane of the final card, so adding
 * a step at the end of a branch stays in that branch.
 */
export const pathIndexToLane = (
  model: ScriptModel,
  index: number,
): { lane: LaneId; laneIndex: number } => {
  const path = resolvePath(model);
  if (path.length === 0) return { lane: null, laneIndex: 0 };

  const clamped = Math.max(0, Math.min(index, path.length));
  if (clamped === path.length) {
    const last = path[path.length - 1];
    return { lane: last.branchId, laneIndex: laneSteps(model, last.branchId).length };
  }
  const target = path[clamped];
  const lane = target.branchId;
  return {
    lane,
    laneIndex: Math.max(0, laneSteps(model, lane).findIndex(s => s.uid === target.step.uid)),
  };
};

export const insertStepAtPathIndex = (
  model: ScriptModel,
  step: ScriptStep,
  index: number,
): ScriptModel => {
  const { lane, laneIndex } = pathIndexToLane(model, index);
  const steps = [...laneSteps(model, lane)];
  steps.splice(Math.max(0, Math.min(laneIndex, steps.length)), 0, step);
  return setLaneSteps(model, lane, steps);
};

/** Append to a specific lane (used by the palette's section-aware add). */
export const appendToLane = (
  model: ScriptModel,
  lane: LaneId,
  step: ScriptStep,
): ScriptModel => {
  const def = COMMAND_BY_ID[step.defId];
  const steps = [...laneSteps(model, lane)];
  // Keep the palette's habit of grouping: land after the last step of the
  // same section when there is one, else at the end.
  let at = steps.length;
  for (let i = steps.length - 1; i >= 0; i--) {
    if (COMMAND_BY_ID[steps[i].defId]?.section === def?.section) {
      at = i + 1;
      break;
    }
  }
  steps.splice(at, 0, step);
  return setLaneSteps(model, lane, steps);
};

export const updateStepInModel = (
  model: ScriptModel,
  uid: string,
  patch: Partial<ScriptStep>,
): ScriptModel => {
  const lane = findLane(model, uid);
  if (lane === undefined) return model;
  return setLaneSteps(
    model,
    lane,
    laneSteps(model, lane).map(s => (s.uid === uid ? { ...s, ...patch } : s)),
  );
};

export const updateParamInModel = (
  model: ScriptModel,
  uid: string,
  key: string,
  value: string,
): ScriptModel => {
  const lane = findLane(model, uid);
  if (lane === undefined) return model;
  return setLaneSteps(
    model,
    lane,
    laneSteps(model, lane).map(s =>
      s.uid === uid ? { ...s, params: { ...s.params, [key]: value } } : s,
    ),
  );
};

/**
 * Remove a step. Removing a trunk step that a branch forks after re-anchors
 * those branches onto the previous trunk step so they are never orphaned.
 */
export const removeStepFromModel = (model: ScriptModel, uid: string): ScriptModel => {
  const lane = findLane(model, uid);
  if (lane === undefined) return model;

  if (lane !== null) {
    return setLaneSteps(model, lane, laneSteps(model, lane).filter(s => s.uid !== uid));
  }

  const idx = model.steps.findIndex(s => s.uid === uid);
  const previousUid = idx > 0 ? model.steps[idx - 1].uid : null;
  return {
    ...model,
    steps: model.steps.filter(s => s.uid !== uid),
    branches: branchesOf(model).map(b =>
      b.forkAfter === uid ? { ...b, forkAfter: previousUid } : b,
    ),
  };
};

export const duplicateStepInModel = (model: ScriptModel, uid: string): ScriptModel => {
  const lane = findLane(model, uid);
  if (lane === undefined) return model;
  const steps = [...laneSteps(model, lane)];
  const i = steps.findIndex(s => s.uid === uid);
  if (i < 0) return model;
  steps.splice(i + 1, 0, { ...steps[i], uid: freshUid(), params: { ...steps[i].params } });
  return setLaneSteps(model, lane, steps);
};

/** Swap with the neighbour INSIDE the owning lane. */
export const moveStepInModel = (
  model: ScriptModel,
  uid: string,
  dir: -1 | 1,
): ScriptModel => {
  const lane = findLane(model, uid);
  if (lane === undefined) return model;
  const steps = [...laneSteps(model, lane)];
  const i = steps.findIndex(s => s.uid === uid);
  const j = i + dir;
  if (i < 0 || j < 0 || j >= steps.length) return model;
  [steps[i], steps[j]] = [steps[j], steps[i]];
  return setLaneSteps(model, lane, steps);
};

/**
 * Drag-drop: move `uid` so it lands at `pathIndex` in the resolved path.
 * Crossing lanes is allowed — that is how a step is promoted out of a branch
 * or pulled into one.
 */
export const moveStepToPathIndex = (
  model: ScriptModel,
  uid: string,
  pathIndex: number,
): ScriptModel => {
  const path = resolvePath(model);
  const from = path.findIndex(r => r.step.uid === uid);
  if (from < 0) return model;
  const clamped = Math.max(0, Math.min(pathIndex, path.length));
  if (clamped === from || clamped === from + 1) return model;

  const target = pathIndexToLane(model, clamped);
  const sourceLane = path[from].branchId;
  const step = path[from].step;

  if (sourceLane === target.lane) {
    const steps = [...laneSteps(model, sourceLane)];
    const i = steps.findIndex(s => s.uid === uid);
    const [dragged] = steps.splice(i, 1);
    const at = target.laneIndex > i ? target.laneIndex - 1 : target.laneIndex;
    steps.splice(Math.max(0, Math.min(at, steps.length)), 0, dragged);
    return setLaneSteps(model, sourceLane, steps);
  }

  const stripped = setLaneSteps(
    model,
    sourceLane,
    laneSteps(model, sourceLane).filter(s => s.uid !== uid),
  );
  const destination = [...laneSteps(stripped, target.lane)];
  destination.splice(Math.max(0, Math.min(target.laneIndex, destination.length)), 0, step);
  return setLaneSteps(stripped, target.lane, destination);
};

/* ------------------------------------------------------------------ */
/* Branch lifecycle                                                    */
/* ------------------------------------------------------------------ */

let uidSeq = 0;
/** Fresh uid for cloned steps — distinct from the builder's own counter. */
export const freshUid = (): string => `step-c${Date.now().toString(36)}-${++uidSeq}`;

const cloneSteps = (steps: ScriptStep[]): ScriptStep[] =>
  steps.map(s => ({ ...s, uid: freshUid(), params: { ...s.params } }));

export interface AddBranchOptions {
  label?: string;
  /**
   * `tail` (default) seeds the branch with a COPY of the trunk steps after the
   * fork, so the user edits a variant of what they already have; `empty`
   * starts from nothing.
   */
  seed?: 'tail' | 'empty';
  rejoin?: boolean;
  note?: string;
  /** Take the new branch immediately (default true). */
  take?: boolean;
}

/**
 * Fork the pipeline after `forkAfter` (null = before everything) and return
 * the new model plus the branch that was created.
 */
export const addBranch = (
  model: ScriptModel,
  forkAfter: string | null,
  opts: AddBranchOptions = {},
): { model: ScriptModel; branch: ScriptBranch } => {
  const seed = opts.seed ?? 'tail';
  const idx = forkAfter === null ? -1 : model.steps.findIndex(s => s.uid === forkAfter);
  const tail = seed === 'tail' ? cloneSteps(model.steps.slice(idx + 1)) : [];

  const siblings = branchesOf(model).filter(b => b.forkAfter === forkAfter).length;
  const branch: ScriptBranch = {
    id: newBranchId(),
    label: opts.label ?? `Concept ${String.fromCharCode(65 + siblings)}`,
    forkAfter,
    steps: tail,
    rejoin: opts.rejoin ?? false,
    note: opts.note,
  };

  const active = (model.activeBranchIds ?? []).filter(
    id => branchesOf(model).find(b => b.id === id)?.forkAfter !== forkAfter,
  );
  return {
    model: {
      ...model,
      branches: [...branchesOf(model), branch],
      activeBranchIds: opts.take === false ? active : [...active, branch.id],
    },
    branch,
  };
};

/** Take a branch at its fork point, or `null` to go back to the main line. */
export const takeBranchAtFork = (
  model: ScriptModel,
  forkAfter: string | null,
  branchId: string | null,
): ScriptModel => {
  const active = (model.activeBranchIds ?? []).filter(
    id => branchesOf(model).find(b => b.id === id)?.forkAfter !== forkAfter,
  );
  return {
    ...model,
    activeBranchIds: branchId === null ? active : [...active, branchId],
  };
};

export const updateBranch = (
  model: ScriptModel,
  branchId: string,
  patch: Partial<Omit<ScriptBranch, 'id'>>,
): ScriptModel => ({
  ...model,
  branches: branchesOf(model).map(b => (b.id === branchId ? { ...b, ...patch } : b)),
});

export const removeBranch = (model: ScriptModel, branchId: string): ScriptModel => ({
  ...model,
  branches: branchesOf(model).filter(b => b.id !== branchId),
  activeBranchIds: (model.activeBranchIds ?? []).filter(id => id !== branchId),
});

/**
 * Promote the taken branch into the trunk and drop every branch at that fork —
 * "this concept won, make it the main line".
 */
export const promoteBranch = (model: ScriptModel, branchId: string): ScriptModel => {
  const branch = branchesOf(model).find(b => b.id === branchId);
  if (!branch) return model;
  const idx =
    branch.forkAfter === null
      ? -1
      : model.steps.findIndex(s => s.uid === branch.forkAfter);
  const head = model.steps.slice(0, idx + 1);
  const tail = branch.rejoin ? model.steps.slice(idx + 1) : [];
  return {
    ...model,
    steps: [...head, ...branch.steps, ...tail],
    branches: branchesOf(model).filter(b => b.forkAfter !== branch.forkAfter),
    activeBranchIds: (model.activeBranchIds ?? []).filter(
      id => branchesOf(model).find(b => b.id === id)?.forkAfter !== branch.forkAfter,
    ),
  };
};

/** Branches grouped by fork point, in trunk order. */
export const branchesByFork = (model: ScriptModel): Map<string | null, ScriptBranch[]> => {
  const map = new Map<string | null, ScriptBranch[]>();
  for (const b of branchesOf(model)) {
    const list = map.get(b.forkAfter) ?? [];
    list.push(b);
    map.set(b.forkAfter, list);
  }
  return map;
};

export { takenBranchAt };

import type { Ctx, Handler } from './args';
import { int } from './args';
import { StyleError } from '../force/types';
import { RunCancelled } from '../errors';
import { emitFrame, ensureThermoComputes } from './run';
import { initRun, setupRun } from '../run/verlet';
import {
  findSnapshot, parseDumpFile, parseDumpSpec, readDumpSnapshot, selectSnapshots, type DumpFile, type DumpReadSpec,
} from '../output/read_dump';

/*
 * read_dump, rerun (Haiku wave 9).
 *
 * read_dump file Nstep fields... keywords: docs.lammps.org/read_dump.html
 * (see output/read_dump.ts). rerun file1 file2 ... keywords dump fields...:
 * docs.lammps.org/rerun.html: "Perform a pseudo simulation run where atom
 * information is read one snapshot at a time from a dump file(s), and energies
 * and forces are computed on the shapshot to produce thermodynamic or other
 * output." Each selected snapshot is one pseudo run of zero steps: read the
 * atoms, set the timestep, compute forces and energies, then output. No
 * integration: "Fixes that perform time integration, such as fix nve or fix npt
 * are not invoked, since no time integration is performed."
 *
 * The run loop mirrors run.ts runSteps(n = 0) but is written here because
 * runSteps is private to run.ts; the thermo rule is the one measured with native
 * LAMMPS: a row is printed on multiples of the thermo interval, and also on the
 * first and last snapshot of the rerun (thermo.html: "at the beginning and end.").
 */

const KEYWORDS = new Set(['first', 'last', 'every', 'skip', 'start', 'stop', 'post', 'dump']);

interface RerunArgs {
  files: string[];
  first: number;
  last: number;
  every: number;
  skip: number;
  start: number | null;
  stop: number | null;
  spec: DumpReadSpec;
}

/** rerun file1 ... keywords ... dump fields ... (rerun.html syntax, dump last). */
export const parseRerunArgs = (a: string[]): RerunArgs => {
  let k = 0;
  const files: string[] = [];
  while (k < a.length && !KEYWORDS.has(a[k])) files.push(a[k++]);
  if (files.length === 0) throw new StyleError('usage: rerun file1 [file2 ...] [first N] [last N] [every N] [skip N] [start N] [stop N] [post yes|no] dump field1 ...');
  const out = { first: 0, last: Number.MAX_SAFE_INTEGER, every: 0, skip: 1, start: null as number | null, stop: null as number | null };
  let dumpWords: string[] | null = null;
  while (k < a.length) {
    const w = a[k];
    if (w === 'dump') { dumpWords = a.slice(k + 1); break; }
    if (!KEYWORDS.has(w)) throw new StyleError(`rerun: unknown keyword '${w}' (keywords: first last every skip start stop post dump)`);
    if (w === 'post') {
      if (a[k + 1] !== 'yes' && a[k + 1] !== 'no') throw new StyleError('rerun post must be yes or no');
      k += 2;
      continue;
    }
    if (k + 1 >= a.length) throw new StyleError(`rerun: keyword ${w} needs a value`);
    const v = int(a[k + 1], `rerun ${w}`);
    switch (w) {
      case 'first': out.first = v; break;
      case 'last': out.last = v; break;
      case 'every':
        if (v < 0) throw new StyleError(`Invalid every value: ${v} < 0`);
        out.every = v; break;
      case 'skip':
        if (v <= 0) throw new StyleError(`Invalid skip value: ${v} <= 0`);
        out.skip = v; break;
      case 'start': out.start = v; break;
      case 'stop': out.stop = v; break;
    }
    k += 2;
  }
  if (dumpWords === null) throw new StyleError('rerun: missing keyword dump (rerun.html: "keyword dump must appear and be last")');
  if (dumpWords.length === 0) throw new StyleError('rerun: keyword dump needs field arguments, e.g. dump x y z');
  if (out.first > out.last) throw new StyleError(`Invalid rerun settings: first must come before last (${out.first} > ${out.last})`);
  const spec = parseDumpSpec(dumpWords, 'rerun dump');
  if (spec.timestepGiven) throw new StyleError('rerun dump: keyword timestep is not used; every snapshot sets the timestep (rerun.html)');
  return { files, ...out, spec };
};

/** rerun ... — see parseRerunArgs; loads the files, selects snapshots and runs each as a zero-step run. */
const rerun: Handler = async (ctx: Ctx, a) => {
  const args = parseRerunArgs(a);
  const { sys, session } = ctx;
  const files: DumpFile[] = args.files.map((name) => parseDumpFile(name, sys.readFile(name)));
  const selected = selectSnapshots(files, { first: args.first, last: args.last, every: args.every, skip: args.skip });
  const s = sys.state;
  if (selected.length === 0) {
    sys.log('rerun: no snapshot selected');
    return;
  }
  ensureThermoComputes(sys);
  const th = sys.thermo;
  const first = selected[0].snap.step;
  const last = selected[selected.length - 1].snap.step;
  sys.io.emit({ kind: 'thermo-header', keywords: [...th.keywords], labels: th.labels() });
  const t0 = performance.now();
  let k = 0;
  for (const { file, snap } of selected) {
    if (args.stop !== null && snap.step > args.stop) {
      throw new StyleError(`Read rerun dump file timestep ${snap.step} > specified stop ${args.stop}`);
    }
    readDumpSnapshot(sys, file, snap, args.spec);
    ensureThermoComputes(sys);
    initRun(sys);
    sys.run = {
      inRun: true, firstStep: first, lastStep: last, beginStep: args.start ?? first, endStep: args.stop ?? last,
      t0, ranOnce: true,
    };
    setupRun(sys);
    if (th.due(s.step, first, last)) sys.io.emit({ kind: 'thermo', row: th.row() });
    let dumped = false;
    for (const d of sys.dumps) {
      if (d.due(s.step, k === 0)) { d.write(); dumped = true; }
    }
    if (dumped || session.frameEvery > 0) emitFrame(sys);
    k++;
    if (session.isCancelled) throw new RunCancelled();
  }
  sys.run.inRun = false;
  emitFrame(sys);
  const seconds = (performance.now() - t0) / 1000;
  sys.log(`Rerun of ${k} snapshot(s) from ${args.files.join(', ')} with ${s.n} atoms (CPU · fp64, no time integration)`);
  sys.io.emit({ kind: 'done', steps: 0, seconds, backend: session.backendLabel });
};

/** read_dump file Nstep field1 ... keywords — docs.lammps.org/read_dump.html. */
const readDump: Handler = ({ sys }, a) => {
  if (a.length < 3) throw new StyleError('usage: read_dump file Nstep field1 field2 ... keyword values ...');
  const [file, nstep] = a;
  const step = int(nstep, 'read_dump Nstep');
  const spec = parseDumpSpec(a.slice(2), 'read_dump');
  const dump = parseDumpFile(file, sys.readFile(file));
  readDumpSnapshot(sys, dump, findSnapshot(dump, step), spec);
};

export const RERUN_COMMANDS: Record<string, Handler> = {
  read_dump: readDump,
  rerun,
};

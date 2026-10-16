import type { RunApi } from './args';
import { emitThermoRow, writeDumps, emitFrame, ensureThermoComputes, markRunSetup } from './run';
import { startRestarts, writeRestarts } from './restart';
import { initRun, runVerlet } from '../run/verlet';
import { minimize } from '../run/min';
import { runNeb } from './neb';
import { FixNeb } from '../fix/neb';

/** The run-loop helpers handed to every command handler (Ctx.run), see RunApi. */
export const RUN_API: RunApi = { emitThermoRow, writeDumps, emitFrame, ensureThermoComputes, markRunSetup, startRestarts, writeRestarts, initRun, runVerlet, minimize, runNeb,
  makeNebFix: (sys, id, group, args) => new FixNeb(sys, id, group, args) };

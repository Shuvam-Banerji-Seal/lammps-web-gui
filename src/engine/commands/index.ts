import type { Handler } from './args';
import { SETUP_COMMANDS } from './setup';
import { FORCEFIELD_COMMANDS } from './forcefield';
import { RUN_COMMANDS } from './run';
import { MISC_COMMANDS } from './misc';

export type { Ctx, Handler } from './args';

/** Every command the browser engine implements (control flow lives in session.ts). */
export const COMMANDS: Record<string, Handler> = {
  ...SETUP_COMMANDS, ...FORCEFIELD_COMMANDS, ...RUN_COMMANDS, ...MISC_COMMANDS,
};

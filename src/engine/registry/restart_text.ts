import { readRestartText, writeRestartText } from '../output/restart';

/** The restart file writer and reader of the core (output/restart.ts), as the commands call them. */
export interface RestartText {
  write: (sys: Parameters<typeof writeRestartText>[0]) => string;
  read: (sys: Parameters<typeof readRestartText>[0], text: string, name: string) => void;
}

const text: RestartText = { write: writeRestartText, read: readRestartText };

/**
 * The restart file writer and reader (output/restart.ts), loaded when a command names restart, write_restart or
 * read_restart (lazyStyles.ts group restart_text), so they stay out of the engine worker's first chunk. The periodic
 * restart files of a run are written only after a restart command, which loads the group.
 */
export const HOOKS: Record<string, RestartText> = { restart: text, write_restart: text, read_restart: text };

/** Thrown when a run is cancelled; the session stays usable. */
export class RunCancelled extends Error {
  constructor() { super('run cancelled'); this.name = 'RunCancelled'; }
}

/** quit: stop reading input (not an error) — docs.lammps.org/quit.html. */
export class QuitSignal extends Error {
  constructor(readonly status: number) { super('quit'); this.name = 'QuitSignal'; }
}

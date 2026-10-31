import { formatNumber } from './cformat';

/*
 * The run's timing breakdown (timer.html: "timer normal" collects "timing information for the most important
 * sections", the default; "timer off" and "timer loop" do not). As native LAMMPS does (src/timer.cpp and the
 * Timer::stamp calls of src/verlet.cpp at b332f28): each stamp adds the wall time since the previous stamp to one
 * section, and the time not stamped to a section is "Other". src/finish.cpp prints the table after the loop time.
 */

export const PAIR = 0, BOND = 1, KSPACE = 2, NEIGH = 3, COMM = 4, OUTPUT = 5, MODIFY = 6;
const LABELS = ['Pair', 'Bond', 'Kspace', 'Neigh', 'Comm', 'Output', 'Modify'];

export class RunTimer {
  /** timer.html styles: off, loop, normal (default), full. */
  level: 'off' | 'loop' | 'normal' | 'full' = 'normal';
  /** Seconds per section of the current or last run. */
  readonly t = new Float64Array(7);
  private last = 0;
  private on = false;
  private t0 = 0;
  /** Seconds of the last run's loop (from start() to stop()). */
  loop = 0;

  /** The loop starts (after the run's setup, as native's timer->init() and barrier_start()). */
  start(): void {
    this.t.fill(0);
    this.on = this.level === 'normal' || this.level === 'full';
    this.t0 = this.last = performance.now();
  }

  stop(): void {
    this.on = false;
    this.loop = (performance.now() - this.t0) / 1000;
  }

  /** timer->stamp(): the time since the last stamp belongs to no section. */
  mark(): void {
    if (this.on) this.last = performance.now();
  }

  /** timer->stamp(section): the time since the last stamp belongs to the section. */
  stamp(section: number): void {
    if (!this.on) return;
    const now = performance.now();
    this.t[section] += (now - this.last) / 1000;
    this.last = now;
  }

  /**
   * The lines of native's "MPI task timing breakdown" for one process (src/finish.cpp mpi_timings: min = avg = max,
   * %varavg 0): Pair, Bond (molecular systems), Kspace (with a kspace style), Neigh, Comm, Output, Modify, Other.
   * Empty when the level collects no sections.
   */
  breakdown(molecular: boolean, kspace: boolean): string[] {
    if (this.level !== 'normal' && this.level !== 'full') return [];
    const loop = this.loop > 0 ? this.loop : 1e-300;
    const g = (v: number, p: number) => formatNumber(v, `%-10.${p}g`);
    const out = ['', 'MPI task timing breakdown:', 'Section |  min time  |  avg time  |  max time  |%varavg| %total',
      '---------------------------------------------------------------'];
    let sum = 0;
    for (let k = 0; k < 7; k++) {
      sum += this.t[k];
      if ((k === BOND && !molecular) || (k === KSPACE && !kspace)) continue;
      const v = this.t[k];
      out.push(`${LABELS[k].padEnd(8)}| ${g(v, 5)} | ${g(v, 5)} | ${g(v, 5)} |${formatNumber(0, '%6.1f')} |${formatNumber((v / loop) * 100, '%6.2f')}`);
    }
    const other = Math.max(0, this.loop - sum);
    out.push(`Other   |            | ${g(other, 4)} |            |       |${formatNumber((other / loop) * 100, '%6.2f')}`);
    return out;
  }
}

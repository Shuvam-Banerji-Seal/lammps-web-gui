import { describe, it } from 'vitest';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import { PairSW } from '../src/engine/force/pair/sw';
import { Neighbor } from '../src/engine/neighbor';

// TEMPORARY debug scaffold for the SW oracle cases; replaced by real unit tests.
const wrap = (obj: unknown, method: string): void => {
  const p = obj as Record<string, unknown>;
  const orig = p[method] as (...a: unknown[]) => unknown;
  p[method] = function (...args: unknown[]) {
    try {
      return orig.apply(this, args);
    } catch (e) {
      const log = `THREW IN ${method}: ${(e as Error).message}\n${(e as Error).stack}`;
      writeFileSync('/tmp/opencode/w2sw/debug.log', log + '\n');
      throw e;
    }
  };
};

describe('debug', () => {
  it('find thrower', async () => {
    wrap(PairSW.prototype, 'compute');
    wrap(PairSW.prototype, 'initOne');
    wrap(Neighbor.prototype, 'build');
    wrap(Neighbor.prototype, 'decide');
    wrap(Neighbor.prototype, 'forwardComm');
    const dir = join(__dirname, 'oracle');
    const swFile = readFileSync(join(dir, 'w2tsw_SiX.sw'), 'utf8');
    const script = `units metal
atom_style atomic
lattice diamond 5.431
region box block 0 2 0 2 0 2
create_box 2 box
create_atoms 1 box
group odd id 1:1000:2
set group odd type 2
mass * 28.0855
variable dx atom 0.06*sin(0.9*x+0.4*y)
variable dy atom 0.06*cos(0.7*y-0.5*z)
variable dz atom 0.06*sin(1.1*z+0.3*x)
displace_atoms all move v_dx v_dy v_dz units box
pair_style sw
pair_coeff * * w2tsw_SiX.sw Si X
timestep 0.001
fix 1 all nve
run 1
`;
    const session = new Session({ emit: () => {}, writeFile: () => {} });
    session.addFile('w2tsw_SiX.sw', swFile);
    let out = '';
    try {
      await session.execute(script);
      out = 'NO ERROR';
    } catch (e) {
      out = `THROWN: ${(e as Error).message}`;
    }
    if (out !== 'NO ERROR') writeFileSync('/tmp/opencode/w2sw/debug.log', (await import('node:fs')).existsSync('/tmp/opencode/w2sw/debug.log') ? (await import('node:fs')).readFileSync('/tmp/opencode/w2sw/debug.log', 'utf8') + `\nouter: ${out}\n` : `outer only: ${out}\n`);
    else writeFileSync('/tmp/opencode/w2sw/debug.log', `NO ERROR\n`);
  });
});

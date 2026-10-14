import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/* pair_style uf3 (src/engine-gpl/uf3): the argument check and the file checks that the native reader makes. */
const runScript = async (script: string, files: Record<string, string>): Promise<string> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
  for (const [name, text] of Object.entries(files)) session.addFile(name, text);
  try {
    await session.execute(script);
  } catch (e) {
    const err = events.find((ev) => ev.kind === 'error');
    return err && 'message' in err ? String(err.message) : String(e);
  }
  const err = events.find((ev) => ev.kind === 'error');
  return err && 'message' in err ? String(err.message) : '';
};

const BOX = ['units metal', 'lattice bcc 3.3', 'region box block 0 2 0 2 0 2', 'create_box 1 box', 'create_atoms 1 box', 'mass 1 92.9'];

describe('pair_style uf3', () => {
  it('accepts only the 2-body and 2 & 3-body forms', async () => {
    const msg = await runScript([...BOX, 'pair_style uf3 4'].join('\n'), {});
    expect(msg).toContain('not (yet) implemented for 4-body terms');
  });

  it('names the missing UNITS metadata in a block header', async () => {
    const pot = '#UF3 POT\n2B Nb Nb 0 3 nk\n8.0 31\n';
    const msg = await runScript([...BOX, 'pair_style uf3 2', 'pair_coeff * * bad.uf3 Nb'].join('\n'), { 'bad.uf3': pot });
    expect(msg).toContain("does not contain the 'UNITS:' metadata");
  });

  it('refuses a 3-body block with rij != rik (native: only rij = rik is implemented)', async () => {
    const pot = [
      '#UF3 POT UNITS: metal',
      '3B Nb Nb Nb 0 3 nk',
      '8.0 4.0 3.0 4 4 4',
    ].join('\n');
    const msg = await runScript([...BOX, 'pair_style uf3 3', 'pair_coeff * * bad.uf3 Nb'].join('\n'), { 'bad.uf3': pot });
    expect(msg).toContain('rij!=rik');
  });

  it('refuses a potential file with no 2-body block', async () => {
    const pot = '#UF3 POT UNITS: metal\n3B Nb Nb Nb 0 3 nk\n8.0 4.0 4.0 4 4 4\n';
    const msg = await runScript([...BOX, 'pair_style uf3 2', 'pair_coeff * * bad.uf3 Nb'].join('\n'), { 'bad.uf3': pot });
    expect(msg).toContain('Possibly no 2B UF3 potential block');
  });
});

describe('pair_style uf3 potential-file layout', () => {
  it('reads the example potential (2B and 3B blocks) without error', async () => {
    const nb = readFileSync(join(__dirname, '..', 'third_party', 'lammps', 'potentials', 'Nb.uf3'), 'utf8');
    const msg = await runScript([...BOX, 'pair_style uf3 3', 'pair_coeff * * Nb.uf3 Nb', 'run 0'].join('\n'), { 'Nb.uf3': nb });
    expect(msg).toBe('');
  });
});

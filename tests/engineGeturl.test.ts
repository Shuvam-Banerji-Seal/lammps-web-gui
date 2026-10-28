// @vitest-environment node
import { createServer, type Server } from 'node:http';
import { gzipSync } from 'node:zlib';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * geturl as native LAMMPS's GetURL::command (src/EXTRA-COMMAND/geturl.cpp at b332f28) reads its arguments and names
 * its output, with the browser engine downloading by fetch into the notebook's files. A local HTTP server stands in for
 * the remote one.
 */

const EAM = 'Cu functions\n29 63.55 3.615 FCC\n';
const GZ = gzipSync(Buffer.from('LAMMPS data file\n'));
let server: Server;
let base = '';

beforeAll(async () => {
  server = createServer((req, res) => {
    if (req.url === '/pot/Cu_u3.eam') { res.writeHead(200); res.end(EAM); return; }
    if (req.url === '/data.gz') { res.writeHead(200); res.end(GZ); return; }
    if (req.url === '/dir/' || req.url === '/') { res.writeHead(200); res.end('<html></html>'); return; }
    res.writeHead(404); res.end('missing');
  });
  await new Promise<void>((ok) => server.listen(0, '127.0.0.1', ok));
  const addr = server.address();
  base = `http://127.0.0.1:${typeof addr === 'object' && addr ? addr.port : 0}`;
});

afterAll(() => new Promise<void>((ok) => server.close(() => ok())));

/** Runs the input; the session's files and the first line of the error (without the engine's "line N: "). */
const run = async (input: string): Promise<{ files: Map<string, string>; error: string | null }> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} } as never);
  let error: string | null = null;
  try { await s.execute(`${input}\n`); } catch (e) { error = (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''); }
  return { files: (s as unknown as { sys: { files: Map<string, string> } }).sys.files, error };
};

describe('geturl', () => {
  it('downloads to the name after the last slash', async () => {
    const r = await run(`geturl ${base}/pot/Cu_u3.eam`);
    expect(r.error).toBeNull();
    expect(r.files.get('Cu_u3.eam')).toBe(EAM);
  });

  it('writes to the output name, and to index.html for an index page or a bare domain', async () => {
    const r = await run(`geturl ${base}/pot/Cu_u3.eam output Cu.eam\ngeturl ${base}/dir/`);
    expect(r.error).toBeNull();
    expect(r.files.get('Cu.eam')).toBe(EAM);
    expect(r.files.get('index.html')).toBe('<html></html>');
    // "pos < 8": with no '/' past the scheme the name is index.html (the bare host is fetched)
    const bare = await run(`geturl ${base}`);
    expect(bare.error).toBeNull();
    expect(bare.files.get('index.html')).toBe('<html></html>');
  });

  it('keeps an existing file with overwrite no', async () => {
    const r = await run(`print "local" file Cu.eam\ngeturl ${base}/pot/Cu_u3.eam output Cu.eam overwrite no`);
    expect(r.error).toBeNull();
    expect(r.files.get('Cu.eam')).toBe('local\n');
    const w = await run(`print "local" file Cu.eam\ngeturl ${base}/pot/Cu_u3.eam output Cu.eam overwrite yes`);
    expect(w.files.get('Cu.eam')).toBe(EAM);
  });

  it('keeps a gzipped file one character per byte', async () => {
    const r = await run(`geturl ${base}/data.gz`);
    expect(r.error).toBeNull();
    const f = r.files.get('data.gz')!;
    expect(f.length).toBe(GZ.length);
    expect(f.charCodeAt(0)).toBe(0x1f);
    expect(f.charCodeAt(1)).toBe(0x8b);
  });

  it('fails as native on an HTTP error, leaving the opened output file empty', async () => {
    const r = await run(`geturl ${base}/nosuch.txt`);
    expect(r.error).toBe('Download of nosuch.txt failed with: HTTP response code said error 404');
    expect(r.files.get('nosuch.txt')).toBe('');
  });

  it('checks the arguments as native, in order', async () => {
    const cases: [string, string][] = [
      ['geturl', 'Illegal geturl command: missing argument(s)'],
      ['geturl www.lammps.org/Cu.eam', "URL 'www.lammps.org/Cu.eam' is not a supported URL"],
      [`geturl ${base}/a output`, 'Illegal geturl output command: missing argument(s)'],
      [`geturl ${base}/a overwrite maybe`, "Expected boolean parameter instead of 'maybe' in input script or data file"],
      [`geturl ${base}/a timeout -1`, 'Invalid timeout -1 for geturl'],
      [`geturl ${base}/a timeout 1.5`, "Expected integer parameter instead of '1.5' in input script or data file"],
      [`geturl ${base}/a verify`, 'Illegal geturl verify command: missing argument(s)'],
      [`geturl ${base}/a verbose yes bogus 1`, 'Unknown geturl keyword: bogus'],
    ];
    for (const [input, msg] of cases) expect((await run(input)).error, input).toBe(msg);
  });
});

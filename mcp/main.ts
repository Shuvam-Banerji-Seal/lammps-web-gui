/**
 * Entry point of the published server (dist/mcp/lammps-mcp.mjs): the MCP stdio transport — one JSON-RPC message per
 * line on stdin, one reply per line on stdout, logs on stderr only. mcp/build.mjs bundles it with the knowledge
 * (mcp/knowledge.ts) built in, so the file runs with plain Node (18 or later) and needs no packages; only
 * get_lammps_doc goes to the network (raw.githubusercontent.com).
 */
import { stdin, stdout, stderr, exit } from 'node:process';
import { createInterface } from 'node:readline';
import knowledge from 'virtual:lammps-mcp-knowledge';
import { createMcpServer } from './server';

const server = createMcpServer(knowledge);
const out = (msg: unknown) => { stdout.write(`${JSON.stringify(msg)}\n`); };

const lines = createInterface({ input: stdin, crlfDelay: Infinity });
let pending = 0;
let closed = false;
lines.on('line', (line) => {
  if (!line.trim()) return;
  let msg: unknown;
  try {
    msg = JSON.parse(line);
  } catch {
    out({ jsonrpc: '2.0', id: null, error: { code: -32700, message: 'Parse error' } });
    return;
  }
  const batch = Array.isArray(msg) ? msg : [msg];
  for (const m of batch) {
    pending++;
    server.handle(m).then((reply) => { if (reply) out(reply); }, (e) => { stderr.write(`lammps-mcp: ${String(e)}\n`); })
      .finally(() => { pending--; if (closed && pending === 0) exit(0); });
  }
});
lines.on('close', () => { closed = true; if (pending === 0) exit(0); });
stderr.write(`lammps-mcp ${knowledge.version}: LAMMPS Web GUI MCP server on stdio (${server.tools.length} tools)\n`);

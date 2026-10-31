# MCP server

This folder builds the site's [Model Context Protocol](https://modelcontextprotocol.io) server. It works with any coding
agent that speaks MCP (Claude Code, Codex, Cursor, VS Code, Windsurf, Cline, Gemini CLI, opencode, ...) and gives it
what the site knows about writing LAMMPS input:

| Tool | What it does |
|---|---|
| `search_lammps_docs` | find LAMMPS commands and styles by name or topic (the whole documentation index) |
| `get_lammps_doc` | the full documentation page of a command or style, fetched from GitHub at the engine's LAMMPS commit |
| `validate_lammps_script` | the site's script checker; every rule cites its docs.lammps.org page |
| `check_browser_support` | what in a script the in-browser engine does not run |
| `list_templates`, `get_template` | the Script Builder's starter scripts |
| `list_examples`, `get_example` | the example notebooks |
| `generate_cmake_build` | a CMake build script for a local LAMMPS |
| `read_guide` | guides: writing inputs, the browser engine, the script checker, file formats, performance |

It also offers resources (the guides, templates, examples, engine support list, and `lammps-doc://{name}` for any doc
page) and a prompt, `write_lammps_input`.

## Connecting

The live page is <https://shuvam-banerji-seal.github.io/lammps-web-gui/mcp/>. In short:

```bash
claude mcp add lammps -- npx -y --allow-remote=all https://shuvam-banerji-seal.github.io/lammps-web-gui/mcp/lammps-mcp.tgz
```

For other clients, use the same `npx` command and arguments in the client's MCP configuration. To avoid npx, download
`lammps-mcp.mjs` and run it with `node` (Node 18 or later, no packages). npm 12 installs from a URL only with
`--allow-remote=all`; npm 10 and 11 ignore that flag.

**Why not a URL endpoint?** GitHub Pages serves static files only. The MCP HTTP transport needs a server that answers
POST requests, so the site publishes the server as a file and the client runs it locally over stdio. Agents without MCP
can fetch `mcp/index.json`, `mcp/knowledge.json` and `llms.txt` directly.

## Files

| File | Role |
|---|---|
| `server.ts` | the server: JSON-RPC handling, tools, resources and the prompt (transport-free; tested in `tests/mcpServer.test.ts`) |
| `main.ts` | the stdio transport, the entry point of the published bundle |
| `knowledge.ts` | builds the knowledge object from the app's sources: Script Builder catalog, templates, example notebooks, the engine's supported commands and styles, the CMake catalog, guides |
| `sources.ts` | reads the files `knowledge.ts` needs (Node only) |
| `site.ts` | the `/mcp/` page, `/mcp/index.json` and `/llms.txt` |
| `build.mjs` | run by `npm run build` after `vite build`: writes `dist/mcp/` (server, npm tarball, knowledge, manifest, page) and `dist/llms.txt`, then runs the bundled server once as a smoke test |
| `guides/writing-inputs.md` | the input-writing guide; each claim quotes its docs.lammps.org page |
| `data/lammps-doc-index.json` | every command and style name of the LAMMPS docs, its page, title and first sentence (LAMMPS doc sources, GPL-2.0) |
| `scripts/doc-index.mjs` | regenerates the doc index from a LAMMPS checkout |

Regenerate the doc index whenever the engine's parity commit changes:

```bash
node mcp/scripts/doc-index.mjs /path/to/lammps <commit>
```

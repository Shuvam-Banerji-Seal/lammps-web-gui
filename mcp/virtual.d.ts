/** The knowledge object mcp/build.mjs builds in when it bundles mcp/main.ts (mcp/knowledge.ts buildKnowledge). */
declare module 'virtual:lammps-mcp-knowledge' {
  const knowledge: import('./knowledge').Knowledge;
  export default knowledge;
}

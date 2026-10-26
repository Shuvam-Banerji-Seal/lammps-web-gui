/**
 * One write to a session file, as the engine reports it (SessionIO.writeFile) and as every holder of the files applies it
 * (System.files, the notebook's run files, the oracle harness): the text replaces the file, is appended to it, or, with
 * an offset `at`, overwrites that many characters from `at` on (a binary dump patching its header: dump dcd updates its
 * frame count after every frame). Binary files are held one character per byte.
 */
export const applyFileWrite = (prev: string | undefined, text: string, append: boolean, at?: number): string => {
  if (at === undefined) return (append ? prev ?? '' : '') + text;
  const old = prev ?? '';
  return old.slice(0, at) + text + old.slice(at + text.length);
};

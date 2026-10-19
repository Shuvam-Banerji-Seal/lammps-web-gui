
/** Download helper shared by script/data exporters. */
/**
 * A file the engine stored as a base64 data URL (dump image writes its PNG / JPEG / PPM frames this way,
 * because the engine's file store holds text): its MIME type and bytes, or null for an ordinary text file.
 */
export const dataUrlBytes = (text: string): { mime: string; bytes: Uint8Array<ArrayBuffer> } | null => {
  const m = /^data:([\w.+/-]+);base64,/.exec(text);
  if (!m) return null;
  const bin = atob(text.slice(m[0].length));
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return { mime: m[1], bytes };
};

/** True for an image stored as a data URL (previewable in an <img>). */
export const isImageDataUrl = (text: string): boolean => /^data:image\/[\w.+-]+;base64,/.test(text);

/** Download an engine file: data URLs as their binary content, everything else as text. */
export const downloadFile = (filename: string, content: string): void => {
  // A .gz file from the engine is a string with one character per byte (gzip.ts); save its bytes, not UTF-8 text.
  if (filename.endsWith('.gz')) {
    const bytes = new Uint8Array(content.length);
    for (let k = 0; k < content.length; k++) bytes[k] = content.charCodeAt(k) & 0xff;
    const url = URL.createObjectURL(new Blob([bytes], { type: 'application/gzip' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 4000);
    return;
  }
  const bin = dataUrlBytes(content);
  if (!bin) { downloadTextFile(filename, content); return; }
  const url = URL.createObjectURL(new Blob([bin.bytes], { type: bin.mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 4000);
};

export const downloadTextFile = (filename: string, text: string): void => {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 4000);
};

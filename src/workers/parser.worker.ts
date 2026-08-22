import { parseFile, detectFormatFromContent } from '../services/fileParser';
import type { FileFormat } from '../types';

/**
 * Off-main-thread parsing worker. Keeps huge structures from freezing the UI.
 * Protocol: { id, text, format? } -> { id, ok, data?, error? }
 */
export interface ParseRequest {
  id: number;
  text: string;
  format?: FileFormat;
}

export interface ParseResponse {
  id: number;
  ok: boolean;
  data?: unknown;
  error?: string;
}

self.onmessage = (e: MessageEvent<ParseRequest>) => {
  const { id, text, format } = e.data;
  try {
    const data = parseFile(text, format ?? detectFormatFromContent(text));
    (self as unknown as Worker).postMessage({ id, ok: true, data } satisfies ParseResponse);
  } catch (err) {
    (self as unknown as Worker).postMessage({
      id,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    } satisfies ParseResponse);
  }
};

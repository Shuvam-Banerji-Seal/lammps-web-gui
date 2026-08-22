import { describe, it, expect } from 'vitest';
import {
  negotiateRecordingFormat,
} from '../src/services/recorder';

describe('recording codec negotiation', () => {
  it('prefers H.264-in-MP4 when available (high quality mp4)', () => {
    const f = negotiateRecordingFormat(m =>
      ['video/mp4;codecs=avc1.42E01E', 'video/webm;codecs=vp9'].includes(m)
    );
    expect(f?.ext).toBe('mp4');
    expect(f?.mime).toBe('video/mp4;codecs=avc1.42E01E');
  });

  it('picks the High profile over baseline when both exist', () => {
    const f = negotiateRecordingFormat(() => true);
    expect(f?.mime).toBe('video/mp4;codecs=avc1.640028');
  });

  it('falls back to WebM VP9 on MP4-less browsers', () => {
    const f = negotiateRecordingFormat(m => m.startsWith('video/webm'));
    expect(f?.ext).toBe('webm');
    expect(f?.label).toContain('VP9');
  });

  it('returns null when nothing is supported', () => {
    expect(negotiateRecordingFormat(() => false)).toBeNull();
  });

  it('survives throwing predicates', () => {
    const f = negotiateRecordingFormat(() => {
      throw new Error('not implemented');
    });
    expect(f).toBeNull();
  });
});

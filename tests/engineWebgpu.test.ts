import { describe, expect, it } from 'vitest';
import { createWebGpuBackend } from '../src/engine/gpu/webgpuForces';

// The WebGPU kernel itself is verified in real Chromium (SwiftShader) against
// the CPU backend: forces within 1e-4, energy and virial within 1e-5, and the
// examples/melt step-0 thermo line through the interpreter. jsdom has no
// WebGPU, which is exactly the fallback path checked here.
describe('createWebGpuBackend', () => {
  it('resolves to null when the browser has no navigator.gpu', async () => {
    expect('gpu' in navigator).toBe(false);
    await expect(createWebGpuBackend()).resolves.toBeNull();
  });

  it('declines a software fallback adapter unless allowFallback is set', async () => {
    const asked: unknown[] = [];
    Object.defineProperty(navigator, 'gpu', {
      configurable: true,
      value: {
        requestAdapter: async (o: unknown) => {
          asked.push(o);
          return { info: { vendor: 'google', architecture: 'swiftshader', isFallbackAdapter: true }, requestDevice: async () => ({ createShaderModule: () => ({}), createComputePipeline: () => ({}), createBuffer: () => ({ destroy() {} }), destroy() {} }) };
        },
      },
    });
    try {
      await expect(createWebGpuBackend()).resolves.toBeNull();
      // a core adapter was asked for first, then a compatibility-mode one
      expect(asked).toEqual([{ powerPreference: 'high-performance' }, { powerPreference: 'high-performance', featureLevel: 'compatibility' }]);
      const sw = await createWebGpuBackend({ allowFallback: true });
      expect(sw?.label).toBe('WebGPU · google swiftshader (software)');
      expect(sw?.software).toBe(true);
    } finally {
      delete (navigator as unknown as { gpu?: unknown }).gpu;
    }
  });

  it('resolves to null when requestAdapter() finds no adapter', async () => {
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: { requestAdapter: async () => null } });
    try {
      await expect(createWebGpuBackend()).resolves.toBeNull();
    } finally {
      delete (navigator as unknown as { gpu?: unknown }).gpu;
    }
  });
});

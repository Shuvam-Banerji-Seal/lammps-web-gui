import React from 'react';

interface State {
  error: Error | null;
}

/**
 * Catches WebGL/Canvas creation failures (headless browsers, GPU-less VMs,
 * context exhaustion, blocked hardware acceleration) and renders a
 * user-friendly fallback instead of letting R3F throw an uncaught error
 * that crashes the React tree.
 */
export class CanvasErrorBoundary extends React.Component<
  { children: React.ReactNode },
  State
> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    // Log at reduced severity — the app is functional without WebGL
    // when the user is on the Script Builder or Compiler modules.
    console.info('[Molecule3D] 3D canvas unavailable:', error.message);
  }

  render(): React.ReactNode {
    if (this.state.error) {
      return (
        <div className="flex h-full w-full flex-col items-center justify-center gap-3 p-8 text-center">
          <div className="text-3xl opacity-40">🔬</div>
          <p className="text-sm font-medium text-gray-400">
            3D view unavailable
          </p>
          <p className="max-w-xs text-xs leading-relaxed text-gray-500">
            WebGL could not be initialised. The Script Builder and Compiler
            Helper remain fully functional. Try enabling hardware acceleration
            or closing other GPU-heavy tabs.
          </p>
          <button
            onClick={() => this.setState({ error: null })}
            className="mt-2 rounded-lg border border-gray-600 px-3 py-1.5 text-xs text-gray-300 transition-colors hover:bg-gray-800"
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

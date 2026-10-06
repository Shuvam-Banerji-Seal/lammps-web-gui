import React from 'react';
import { getThemeTokens, Theme } from '../../theme';

/**
 * MD Notebook — placeholder; the full UI (cells, outputs, live 3D view)
 * is built on src/engine/client.ts (EngineClient) and src/engine/view.ts.
 */
const Notebook: React.FC<{ theme: Theme }> = ({ theme }) => {
  const ct = getThemeTokens(theme);
  return (
    <div className={`h-full p-6 ${ct.bg}`}>
      <h2 className={`text-sm font-semibold ${ct.headerText}`}>MD Notebook</h2>
    </div>
  );
};

export default Notebook;

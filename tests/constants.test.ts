import { describe, it, expect } from 'vitest';
import { ATOM_COLORS, DEFAULT_ATOM_COLOR, ELEMENT_DATA, ELEMENT_RADII } from '../constants';

describe('Constants - Periodic Table', () => {
  it('should have colors for all 118 elements', () => {
    for (let i = 1; i <= 118; i++) {
      expect(ATOM_COLORS[i]).toBeDefined();
      expect(ATOM_COLORS[i]).toMatch(/^#[0-9A-Fa-f]{6}$/);
    }
  });

  it('should have element data for all 118 elements', () => {
    expect(ELEMENT_DATA).toHaveLength(118);

    // Verify sequential atomic numbers
    const numbers = ELEMENT_DATA.map(e => e.number).sort((a, b) => a - b);
    for (let i = 0; i < 118; i++) {
      expect(numbers[i]).toBe(i + 1);
    }
  });

  it('should have radii for all 118 elements', () => {
    for (let i = 1; i <= 118; i++) {
      expect(ELEMENT_RADII[i]).toBeDefined();
      expect(ELEMENT_RADII[i]).toBeGreaterThan(0);
    }
  });

  it('should have correct known element data', () => {
    const hydrogen = ELEMENT_DATA.find(e => e.symbol === 'H');
    expect(hydrogen).toBeDefined();
    expect(hydrogen!.number).toBe(1);
    expect(hydrogen!.mass).toBeCloseTo(1.008, 2);

    const carbon = ELEMENT_DATA.find(e => e.symbol === 'C');
    expect(carbon).toBeDefined();
    expect(carbon!.number).toBe(6);
    expect(carbon!.mass).toBeCloseTo(12.01, 1);

    const gold = ELEMENT_DATA.find(e => e.symbol === 'Au');
    expect(gold).toBeDefined();
    expect(gold!.number).toBe(79);
    expect(gold!.mass).toBeCloseTo(196.97, 1);
  });

  it('should have correct CPK colors for common elements', () => {
    expect(ATOM_COLORS[1]).toBe('#FFFFFF');   // Hydrogen - White
    expect(ATOM_COLORS[6]).toBe('#909090');   // Carbon - Grey
    expect(ATOM_COLORS[7]).toBe('#3050F8');   // Nitrogen - Blue
    expect(ATOM_COLORS[8]).toBe('#FF0D0D');   // Oxygen - Red
  });

  it('should have a default atom color defined', () => {
    expect(DEFAULT_ATOM_COLOR).toBe('#FF00FF');
  });

  it('should have unique element symbols', () => {
    const symbols = ELEMENT_DATA.map(e => e.symbol);
    const uniqueSymbols = new Set(symbols);
    expect(uniqueSymbols.size).toBe(symbols.length);
  });

  it('should have unique element numbers', () => {
    const numbers = ELEMENT_DATA.map(e => e.number);
    const uniqueNumbers = new Set(numbers);
    expect(uniqueNumbers.size).toBe(numbers.length);
  });

  it('should have reasonable atomic masses (all positive)', () => {
    ELEMENT_DATA.forEach(elem => {
      expect(elem.mass).toBeGreaterThan(0);
    });
  });

  it('should have reasonable van der Waals radii', () => {
    // Hydrogen radius should be around 1.2 Å
    expect(ELEMENT_RADII[1]).toBeCloseTo(1.2, 0);
    // Carbon should be around 1.7 Å
    expect(ELEMENT_RADII[6]).toBeCloseTo(1.7, 0);
  });
});

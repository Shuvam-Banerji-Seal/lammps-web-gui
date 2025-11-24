import { Atom, Bond, MoleculeData, ParseSection, AtomTypeInfo } from '../types';
import { ELEMENT_DATA } from '../constants';

export const parseDataFile = (data: string): MoleculeData => {
  const lines = data.split('\n');
  const atoms: Atom[] = [];
  const bonds: Bond[] = [];
  
  // Temporary storage for Masses section parsing
  const masses: Record<number, { mass: number; comment?: string }> = {};

  let currentSection: ParseSection = ParseSection.NONE;

  // Track min/max for bounding box centering
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    // Skip empty lines (but keep processing for section detection logic if needed, usually not)
    if (!line) continue;

    // Check for comments to extract inline labels
    const parts = line.split('#');
    const content = parts[0].trim();
    const comment = parts.length > 1 ? parts[1].trim() : undefined;

    if (!content) continue;

    // Detect Sections
    if (content.match(/^Masses/i)) {
      currentSection = ParseSection.MASSES;
      continue;
    }
    if (content.match(/^Atoms/i)) {
      currentSection = ParseSection.ATOMS;
      continue;
    }
    if (content.match(/^Bonds/i)) {
      currentSection = ParseSection.BONDS;
      continue;
    }
    if (content.match(/^Velocities/i) || content.match(/^Angles/i) || content.match(/^Dihedrals/i)) {
      currentSection = ParseSection.NONE; // Ignored sections for this visualizer
      continue;
    }

    // Parse Content based on section
    const tokens = content.split(/\s+/);

    if (currentSection === ParseSection.MASSES) {
      // Format: ID mass
      if (tokens.length >= 2) {
        const id = parseInt(tokens[0], 10);
        const mass = parseFloat(tokens[1]);
        if (!isNaN(id) && !isNaN(mass)) {
          masses[id] = { mass, comment };
        }
      }
    } else if (currentSection === ParseSection.ATOMS) {
      // Format: ID molecule-tag type charge x y z
      if (tokens.length >= 7) {
        const id = parseInt(tokens[0], 10);
        const molId = parseInt(tokens[1], 10);
        const type = parseInt(tokens[2], 10);
        const charge = parseFloat(tokens[3]);
        const x = parseFloat(tokens[4]);
        const y = parseFloat(tokens[5]);
        const z = parseFloat(tokens[6]);

        if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
          atoms.push({ id, molId, type, charge, x, y, z });

          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          minZ = Math.min(minZ, z);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
          maxZ = Math.max(maxZ, z);
        }
      }
    } else if (currentSection === ParseSection.BONDS) {
      // Format: ID type atom1 atom2
      if (tokens.length >= 4) {
        const id = parseInt(tokens[0], 10);
        const type = parseInt(tokens[1], 10);
        const atom1Id = parseInt(tokens[2], 10);
        const atom2Id = parseInt(tokens[3], 10);
        bonds.push({ id, type, atom1Id, atom2Id });
      }
    }
  }

  // Generate Atom Type Metadata
  const atomTypes: Record<number, AtomTypeInfo> = {};
  
  // First, identify all unique types used in the atoms list
  const usedTypes = new Set<number>();
  atoms.forEach(a => usedTypes.add(a.type));

  usedTypes.forEach(type => {
    let mass = 0;
    let label = `Type ${type}`;
    let element = "X";
    
    // 1. Try to get mass from parsed Masses section
    if (masses[type]) {
      mass = masses[type].mass;
      // If there was a comment like "Carbon" or "C", use it
      if (masses[type].comment) {
        label = masses[type].comment!;
        // Simple heuristic: if label is short, it might be symbol
        if (label.length <= 2) element = label;
      }
    }

    // 2. If no explicit label from comment, try to guess Element from Mass
    if (mass > 0 && element === "X") {
      const match = ELEMENT_DATA.find(e => Math.abs(e.mass - mass) < 0.5);
      if (match) {
        element = match.symbol;
        if (label === `Type ${type}`) {
           label = `${match.name} (Type ${type})`;
        }
      }
    } else if (mass === 0) {
      // 3. Fallback: If no mass provided, check if the Type ID itself matches an Atomic Number (common in some files)
      // e.g. Type 6 is Carbon
      if (ELEMENT_DATA.some(e => e.number === type)) {
         const match = ELEMENT_DATA.find(e => e.number === type)!;
         element = match.symbol;
         if (label === `Type ${type}`) {
             label = `${match.name} (Type ${type})`;
         }
      }
    }

    atomTypes[type] = {
      id: type,
      mass,
      element,
      label,
      count: atoms.filter(a => a.type === type).length
    };
  });

  // Calculate center
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const centerZ = (minZ + maxZ) / 2;

  // Handle case where no atoms parsed
  const safeCenter = atoms.length > 0 
    ? { x: centerX, y: centerY, z: centerZ } 
    : { x: 0, y: 0, z: 0 };

  return {
    atoms,
    bonds,
    atomTypes,
    min: { x: minX, y: minY, z: minZ },
    max: { x: maxX, y: maxY, z: maxZ },
    center: safeCenter
  };
};
#!/usr/bin/env python3
"""
Test script for GMM integration with crystal analyzer and ASE atomic models.

This script demonstrates the new feature that updates atomic columns based on
GMM atom count estimates, placing atoms symmetrically in the Z direction.
"""

import numpy as np
from ase import Atoms
from qem.atomic_column import AtomicColumns

def test_symmetric_z_placement():
    """Test the symmetric Z position calculation."""
    # Create a simple atomic columns object for testing
    lattice = Atoms('Au', positions=[[0, 0, 0]])
    lattice_ref = Atoms('Au', positions=[[0, 0, 0]])
    
    atomic_columns = AtomicColumns(
        lattice=lattice,
        lattice_ref=lattice_ref,
        elements=['Au'],
        tol=0,
        pixel_size=0.1
    )
    
    print("Testing symmetric Z position calculation:")
    
    # Test different numbers of atoms
    for num_atoms in range(1, 6):
        positions = atomic_columns._calculate_symmetric_z_positions(
            center_z=0.0, 
            num_atoms=num_atoms, 
            spacing=2.0
        )
        print(f"  {num_atoms} additional atoms: {positions}")

def test_gmm_integration():
    """Test GMM integration with atomic columns."""
    print("\nTesting GMM integration:")
    
    # Create a simple lattice with multiple atoms
    positions = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]
    lattice = Atoms('Au3', positions=positions)
    lattice_ref = Atoms('Au3', positions=positions)
    
    # Add unit cell information
    lattice.set_cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    lattice_ref.set_cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    
    atomic_columns = AtomicColumns(
        lattice=lattice,
        lattice_ref=lattice_ref,
        elements=['Au'],
        tol=0,
        pixel_size=0.1
    )
    
    print(f"Initial lattice has {len(lattice)} atoms")
    
    # Create mock GMM estimates (different atom counts for each column)
    atom_count_estimates = {
        'Au': [2, 3, 1]  # Column 0: 2 atoms, Column 1: 3 atoms, Column 2: 1 atom
    }
    
    # Update with GMM estimates
    updated_lattice, updated_lattice_ref = atomic_columns.update_atoms_from_gmm(
        atom_count_estimates
    )
    
    print(f"Updated lattice has {len(updated_lattice)} atoms")
    print("Updated positions:")
    for i, pos in enumerate(updated_lattice.positions):
        print(f"  Atom {i}: {pos}")

def test_z_spacing_determination():
    """Test automatic Z-spacing determination from lattice."""
    print("\nTesting Z-spacing determination:")
    
    # Create lattice with multiple Z levels
    positions = [[0, 0, 0], [0, 0, 2], [0, 0, 4]]
    lattice = Atoms('Au3', positions=positions)
    lattice.set_cell([[4, 0, 0], [0, 4, 0], [0, 0, 6]])
    
    atomic_columns = AtomicColumns(
        lattice=lattice,
        lattice_ref=lattice.copy(),
        elements=['Au'],
        tol=0,
        pixel_size=0.1
    )
    
    z_spacing = atomic_columns._determine_z_spacing_from_lattice()
    print(f"Determined Z-spacing: {z_spacing:.2f} Å")
    
    # Test with single Z level
    single_z_lattice = Atoms('Au3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    single_z_lattice.set_cell([[4, 0, 0], [0, 4, 0], [0, 0, 3]])
    
    atomic_columns_single = AtomicColumns(
        lattice=single_z_lattice,
        lattice_ref=single_z_lattice.copy(),
        elements=['Au'],
        tol=0,
        pixel_size=0.1
    )
    
    z_spacing_single = atomic_columns_single._determine_z_spacing_from_lattice()
    print(f"Z-spacing from single Z level (using c-axis): {z_spacing_single:.2f} Å")

if __name__ == "__main__":
    print("Testing GMM integration with atomic models")
    print("=" * 50)
    
    test_symmetric_z_placement()
    test_z_spacing_determination()
    test_gmm_integration()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("\nUsage example:")
    print("1. Run GMM analysis: model.estimate_atom_counts_with_gmm()")
    print("2. Integrate with crystal analyzer: model.integrate_gmm_with_crystal_analyzer()")
    print("3. Export updated structure: model.export_gmm_updated_structure('updated_structure')")
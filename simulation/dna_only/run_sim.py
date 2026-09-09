import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'OpenABC'))

from openabc.forcefields.parsers import MRGdsDNAParser
from openabc.forcefields import MOFFMRGModel
from openabc.utils.insert import insert_molecules

try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit


platform_name = 'CUDA'

DNA_parser = MRGdsDNAParser.from_atomistic_pdb(str(Path(__file__).resolve().parents[1] / 'inputs/all_atom_200bpDNA.pdb'), 'dsDNA_CG.pdb')

a, b, c = 75, 75, 75
n_dna = 1
insert_molecules('dsDNA_CG.pdb', 'start.pdb', n_mol=n_dna, box=[a, b, c])

protein_dna = MOFFMRGModel()
for _ in range(n_dna):
    protein_dna.append_mol(DNA_parser)

top = app.PDBFile('start.pdb').getTopology()
protein_dna.create_system(top, box_a=a, box_b=b, box_c=c)
salt_conc = 82*unit.millimolar
temperature = 300*unit.kelvin
protein_dna.add_dna_bonds(force_group=5)
protein_dna.add_dna_angles(force_group=6)
protein_dna.add_dna_fan_bonds(force_group=7)
protein_dna.add_contacts(force_group=8)
protein_dna.add_elec_switch(salt_conc, temperature, force_group=9)
protein_dna.save_system('system.xml')

friction_coeff = 0.01/unit.picosecond
timestep = 10*unit.femtosecond
integrator = mm.LangevinMiddleIntegrator(temperature, friction_coeff, timestep)
init_coord = app.PDBFile('start.pdb').getPositions()
protein_dna.set_simulation(integrator, platform_name, init_coord=init_coord)
protein_dna.simulation.minimizeEnergy()
output_interval = 100000
protein_dna.add_reporters(output_interval, 'output.dcd')
protein_dna.simulation.context.setVelocitiesToTemperature(temperature)
protein_dna.simulation.step(int(os.environ.get('OPENABC_SIM_STEPS', '500000000')))

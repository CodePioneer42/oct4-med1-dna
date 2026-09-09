reinitialize
set ray_opaque_background, 1
set orthoscopic, on
set cartoon_smooth_loops, 1
set cartoon_sampling, 10
set antialias, 2
bg_color white
load simulation/inputs/OCT4.pdb, oct4
hide everything
show cartoon, oct4
color grey85, oct4
color tv_blue, oct4 and resi 136-217
color violet, oct4 and resi 234-286
orient oct4
zoom oct4, 3
ray 1000, 700
png analysis/outputs/figures/main/fig1_panel_b_oct4_pymol.png, 1000, 700, 300, 1

reinitialize
set ray_opaque_background, 1
set orthoscopic, on
set cartoon_smooth_loops, 1
set cartoon_sampling, 10
set antialias, 2
bg_color white
load simulation/inputs/MED1-alphafold.pdb, med1
hide everything
show cartoon, med1
color grey85, med1
color teal, med1 and resi 77-232
color marine, med1 and resi 252-274
color purple, med1 and resi 287-349
color salmon, med1 and resi 361-518
orient med1
zoom med1, 3
ray 1000, 700
png analysis/outputs/figures/main/fig1_panel_b_med1_pymol.png, 1000, 700, 300, 1

reinitialize
set ray_opaque_background, 1
set orthoscopic, on
set cartoon_smooth_loops, 1
set cartoon_sampling, 10
set antialias, 2
set cartoon_nucleic_acid_mode, 1
bg_color white
load simulation/inputs/all_atom_200bpDNA.pdb, dna
hide everything
show cartoon, dna
color forest, dna
orient dna
zoom dna, 3
ray 1000, 700
png analysis/outputs/figures/main/fig1_panel_b_dna_pymol.png, 1000, 700, 300, 1

reinitialize
set ray_opaque_background, 1
set orthoscopic, on
set cartoon_smooth_loops, 1
set cartoon_sampling, 10
set cartoon_loop_radius, 0.45
set antialias, 2
bg_color white

load simulation/inputs/MED1-alphafold.pdb, med1_full
hide everything
show cartoon, med1_full
color grey65, med1_full
color teal, med1_full and resi 77-232
color marine, med1_full and resi 252-274
color purple, med1_full and resi 287-349
color salmon, med1_full and resi 361-518
orient med1_full
zoom med1_full, 4
ray 1200, 700
png analysis/outputs/figures/main/fig4_panel_a_med1_full_pymol.png, 1200, 700, 300, 1

create med1_n, med1_full and resi 1-807
create med1_c, med1_full and resi 808-1581
disable med1_full
translate [-35, 0, 0], med1_n
translate [35, 0, 0], med1_c
orient med1_n or med1_c
zoom med1_n or med1_c, 4
ray 1200, 700
png analysis/outputs/figures/main/fig4_panel_a_med1_split2_pymol.png, 1200, 700, 300, 1

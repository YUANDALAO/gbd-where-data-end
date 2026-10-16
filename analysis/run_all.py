"""Reproduce every result, table and figure in order.
Set WHO_MDB_DIR to the folder filled by fetch_public_data.py before running.
"""
import subprocess, sys

STEPS = ['build_raw_vr.py', 'h1_h2.py', 'h1_srs_sensitivity.py', 'h3_onset.py', 'h3_robust.py', 'h3_exclude4.py', 'h2_adjr2.py', 'h4_h5.py', 'h4_h5_noise.py', 'h6_biblio.py',
         'figs.py', 'fig_data_country.py', 'figs_country.py', 'supp_tables.py']

for s in STEPS:
    print(f'=== {s}', flush=True)
    subprocess.run([sys.executable, s], check=True)

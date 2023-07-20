# Commands to reproduce each figure in the paper
First see [README.md](README.md) for how to set up the environment.

The figure for each experiment will be written to the `figures` directory.

#### Figure 3
`python experiments/exp_gaussian_power_curves.py mmd_vs_ksd`

#### Figure 4a
`python experiments/exp_gaussian_power_curves.py increasing_n_null_ksd`

#### Figure 4b
`python experiments/exp_gaussian_power_curves.py increasing_n_alt_ksd`

#### Figure 5
`python experiments/exp_gaussian_power_curves.py increasing_n_null_mmd`

#### Figure 6a
`python experiments/exp_gaussian_power_curves.py increasing_d_wild`

#### Figure 6b
`python experiments/exp_gaussian_power_curves.py increasing_d_parametric`

#### Figure 7
`python experiments/exp_gaussian_power_curves.py other_tests`

#### Figure 8
`python experiments/exp_kef_visual_fits.py`

#### Figure 9
Code not included

#### Figure 10 & 11
`python experiments/exp_toggle_switch_increasing_T.py increasing_T`

#### Figure 12
`python experiments/exp_lengthscales.py grid`

#### Figure 13
`python experiments/exp_overlap.py bad_optimizer`

#### Figure 14
`python experiments/exp_lengthscales.py bad_lengthscales_parametric`

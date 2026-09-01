#!/bin/bash -l
# Cross-calibration of SATp1 against Planck.
#
#   ./run_satp1.sh              # full emcee run + plots
#   ./run_satp1.sh minimize     # just the best fit (fast, ~minutes)
#   ./run_satp1.sh single_point # one likelihood evaluation (seconds)
#   ./run_satp1.sh fisher       # best fit + Fisher errors
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BBPOWER="$(cd "${HERE}/../../.." && pwd)"   # repo root
CONFIG="${HERE}/config_satp1.yml"
SAMPLER="${1:-emcee}"

export OMP_NUM_THREADS=1        # emcee parallelises over walkers, not BLAS

cd "${HERE}"

# The fiducial CMB is BBPower's own examples/data/camb_lens_nobb.dat, which
# is already in the repo -- nothing to generate here.

python -u "${BBPOWER}/bbpower/compsep_calib.py" --config "${CONFIG}" --sampler "${SAMPLER}"

OUT=$(python - "${CONFIG}" <<'PY'
import sys, yaml
yaml.SafeLoader.add_constructor("!path",
                                lambda l, n: "/".join(l.construct_sequence(n)))
print(yaml.load(open(sys.argv[1]), Loader=yaml.SafeLoader)['output_dir'])
PY
)
python -u "${BBPOWER}/bbpower/plotter_calib.py" --dir "${OUT}"
echo "done -- results in ${OUT}"

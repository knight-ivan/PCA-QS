#!/bin/zsh
# Full rerun of every experiment in the theory paper: profile design (companion paper's
# PCA-QS: B = 5, k = min(k*, 10), allocation max(1, floor(delta N_g))) as the main design and
# the full grid as the finest comparison; SRS at equal realized size; 1000 replicates; fixed
# seeds; all cores (joblib). Logs in ../figures/logs/.
set -u
# one BLAS thread per worker: joblib already uses every core; nested BLAS threads oversubscribe
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$(dirname "$0")"
LOG=../figures/logs; mkdir -p $LOG
run() { local name=$1; shift; local t0=$(date +%s)
        echo "[$(date +%H:%M:%S)] START $name" | tee -a $LOG/pipeline.log
        python3 "$@" > $LOG/$name.log 2>&1; local ec=$?
        echo "[$(date +%H:%M:%S)] END   $name exit=$ec ($(( $(date +%s)-t0 ))s)" | tee -a $LOG/pipeline.log; }
# theory validation, variance theorem and downstream analyses finished 2026-09-15 (see pipeline_launch2.log)
# --- distributional discrepancies on real data
run realdata_orig_profile   confirm_real_data.py --datasets CreditCard MAGIC EEG Epileptic HIGGS YearPrediction --space original --stratification profile --tag _profile
run realdata_orig_uncapped  confirm_real_data.py --datasets Epileptic YearPrediction --space original --stratification profile --uncapped --tag _profile_uncapped
run realdata_orig_grid      confirm_real_data.py --datasets CreditCard MAGIC EEG Epileptic HIGGS YearPrediction --space original --stratification grid --tag _grid
# --- models, supplementary studies
run coef_recovery          coef_recovery.py --reps 1000
run classification_real    confirm_classification_real.py --reps 1000
run kl_rate                kl_rate_validation.py --reps 1000
run rate_validation        rate_validation.py --reps 1000
run synthetic_distance     confirm_synthetic_distance.py --runs 1000
run classification_synth   confirm_classification.py --runs 1000
cp ../figures/variance_theorem_validation.png ../../manuscript/ 2>/dev/null
run heatmaps               make_realdata_heatmaps.py
echo "[$(date +%H:%M:%S)] PIPELINE DONE" | tee -a $LOG/pipeline.log

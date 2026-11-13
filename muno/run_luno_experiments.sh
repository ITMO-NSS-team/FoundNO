#!/usr/bin/env bash
# Серия LUNO-экспериментов (muno/uncertainty/run_luno_ggn_calibrate.py) подряд.
# Упавший прогон не останавливает серию: запускается следующий, в конце --
# сводка и ненулевой код выхода, если что-то упало.
#
# Запуск из любой папки:
#   bash muno/run_luno_experiments.sh                 # все прогоны
#   bash muno/run_luno_experiments.sh --device 1      # доп. аргументы -> в каждый прогон
#   ONLY=cfno bash muno/run_luno_experiments.sh       # только прогоны, в имени которых есть "cfno"
# Фоном (как остальные GPU-задачи):
#   setsid nohup bash muno/run_luno_experiments.sh > luno_series.log 2>&1 &

set -u
cd "$(dirname "$0")/.."

PY=${PY:-python3}
SCRIPT=muno/uncertainty/run_luno_ggn_calibrate.py
CONFIGS=experiments/scripts/configs
LOG_DIR=${LOG_DIR:-logs/luno_series_$(date +%Y%m%d_%H%M%S)}
ONLY=${ONLY:-}
EXTRA_ARGS=("$@")

# --- ядра ---------------------------------------------------------------
H100=weights/h100_results_pdebench_thewell_alexandra
H100_ARGS=(
    --core-checkpoint "$H100/core"
    --lift-checkpoint-dir "$H100/checkpoints/lift"
    --proj-checkpoint-dir "$H100/checkpoints/proj"
    --in-normalizers "$H100/normalizers_in"
    --out-normalizers "$H100/normalizers_out"
)

CFNO=weights/cfno_9_11
CFNO_ARGS=(
    --core-checkpoint "$CFNO/core"
    --lift-checkpoint-dir "$CFNO/lift"
    --proj-checkpoint-dir "$CFNO/proj"
    --in-normalizers "$CFNO/normalizers_in"
    --out-normalizers "$CFNO/normalizers_out"
    # Новое ядро: сначала сверить forward-mode Jv с reverse-строками на
    # одном сэмпле (падает явно, если CFNO не поддерживает forward-mode AD).
    --jacobian-check
)

# --- серия: имя | yaml | аргументы ядра ------------------------------------
NAMES=()
FAILED=()
DONE=()
SKIPPED=()

run() {
    local name=$1 config=$2
    shift 2
    if [[ -n "$ONLY" && "$name" != *"$ONLY"* ]]; then
        SKIPPED+=("$name")
        return
    fi
    NAMES+=("$name")
    local log="$LOG_DIR/$name.log"
    echo "=================================================================="
    echo "[$(date '+%F %T')] START $name  (config: $config, log: $log)"
    echo "=================================================================="
    local t0=$SECONDS
    "$PY" "$SCRIPT" --config "$config" "$@" "${EXTRA_ARGS[@]}" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    local dt=$(( SECONDS - t0 ))
    local took
    took=$(printf '%dh%02dm' $((dt / 3600)) $((dt % 3600 / 60)))
    if [[ $rc -eq 0 ]]; then
        DONE+=("$name ($took)")
        echo "[$(date '+%F %T')] OK   $name за $took"
    else
        FAILED+=("$name (код $rc, $took)")
        echo "[$(date '+%F %T')] FAIL $name: код $rc за $took -- перехожу к следующему"
    fi
}

mkdir -p "$LOG_DIR"

run h100_kfac "$CONFIGS/luno_turbo_inf_kfac.yaml" "${H100_ARGS[@]}"
run h100_iso  "$CONFIGS/luno_turbo_inf_iso.yaml"  "${H100_ARGS[@]}"
run cfno_lifting "$CONFIGS/luno_cfno_lifting.yaml" "${CFNO_ARGS[@]}"
run cfno_iso     "$CONFIGS/luno_cfno_iso.yaml"     "${CFNO_ARGS[@]}"

echo
echo "================================ ИТОГ ================================"
echo "Логи: $LOG_DIR"
for s in "${DONE[@]+"${DONE[@]}"}";       do echo "  OK    $s"; done
for s in "${FAILED[@]+"${FAILED[@]}"}";   do echo "  FAIL  $s"; done
for s in "${SKIPPED[@]+"${SKIPPED[@]}"}"; do echo "  skip  $s (ONLY=$ONLY)"; done
echo "Внимание: задачи внутри прогона падают по отдельности, не роняя прогон --"
echo "смотрите 'Упали (...)' в конце каждого лога."
[[ ${#FAILED[@]} -eq 0 ]]

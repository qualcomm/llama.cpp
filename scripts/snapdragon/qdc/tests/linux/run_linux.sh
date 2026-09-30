#!/bin/bash
# llama.cpp Hexagon test entry script for QDC Linux IoT (BASH framework).
#
# Placeholders substituted by run_qdc_jobs.py (--platform linux) before upload:
#   {MODEL_URL}   direct URL to a .gguf model file
#   {TEST_MODE}   bench | backend-ops | all
#
# QDC extracts the artifact zip to /data/local/tmp/TestContent/ and invokes
# this script via: /bin/bash /data/local/tmp/TestContent/run_linux.sh
# Any files written under /data/local/tmp/QDC_logs/ are auto-uploaded.

set +e
umask 022

LOG_DIR=/data/local/tmp/QDC_logs
BUNDLE_DIR=/data/local/tmp/TestContent/llama_cpp_bundle
MODEL_DIR=/data/local/tmp/gguf
MODEL_PATH="$MODEL_DIR/model.gguf"
RESULTS_XML="$LOG_DIR/results.xml"

mkdir -p "$LOG_DIR" "$MODEL_DIR"
# Redirect all parent-shell output to script.log so QDC auto-uploads it;
# per-case runs still capture their own stdout/stderr into dedicated logs.
exec > "$LOG_DIR/script.log" 2>&1

echo "=== env ==="
date -u
uname -a
pwd

mount -o rw,remount / 2>/dev/null || true

# ---------------------------------------------------------------------------
# Adreno OpenCL driver: pin to the known-good qcom-adreno1 build.
# QDC QCS9075M units abort in the GPU shader compiler (NumConstRegsError in
# QGPUInstructionValidator.cpp) while another IQ9 with this exact package runs
# the same binaries fine, and the failing units load /usr/lib/libOpenCL.so.1
# instead of the multiarch copy. Best-effort: any failure is logged and the
# tests still run.
# ---------------------------------------------------------------------------
ADRENO_VER="1.855.3+rev2+repack1"
ADRENO_DEB="qcom-adreno1_${ADRENO_VER}_arm64.deb"
ADRENO_URL="https://ppa.launchpadcontent.net/ubuntu-qcom-iot/qcom-ppa/ubuntu/pool/main/q/qcom-adreno/${ADRENO_DEB}"
ADRENO_SHA256="ea7c5f2c3ea0f9dd0ca5f90ab27f29f8dd6c821cc75a783de85698d06016c96f"

# Verbose GPU/OpenCL state dump for debugging IoT GPU failures.
gpu_debug_dump() {
  echo "=== GPU debug ($1) ==="
  echo "--- os-release / kernel / hostname ---"
  head -4 /etc/os-release 2>&1
  uname -r
  hostname
  echo "--- libOpenCL files ---"
  ls -l /usr/lib/libOpenCL* /usr/lib/aarch64-linux-gnu/libOpenCL* /lib/aarch64-linux-gnu/libOpenCL* 2>&1
  for f in /usr/lib/libOpenCL.so.1 /usr/lib/aarch64-linux-gnu/libOpenCL.so.1; do
    [ -e "$f" ] && echo "$f -> $(readlink -f "$f") sha256=$(sha256sum "$f" | cut -d' ' -f1)"
  done
  echo "--- OpenCL ICD vendors ---"
  ls -l /etc/OpenCL/vendors 2>&1
  cat /etc/OpenCL/vendors/* 2>&1
  echo "--- adreno/opencl packages ---"
  dpkg -l 2>&1 | grep -iE 'adreno|opencl|ocl-icd|libgbm|qcom-' || echo "(none / no dpkg)"
  dpkg -L qcom-adreno1 2>&1 | grep -E 'OpenCL|llvm|libgsl' | head -20
  echo "--- ldconfig cache (OpenCL/adreno) ---"
  ldconfig -p 2>&1 | grep -iE 'OpenCL|adreno|libgsl|llvm-qcom'
  echo "--- ldd of bundled ggml-opencl ---"
  ldd "$BUNDLE_DIR/lib/libggml-opencl.so.0" 2>&1 | grep -iE 'OpenCL|not found'
  echo "--- GPU device nodes / kernel driver ---"
  ls -l /dev/kgsl* /dev/dri 2>&1
  dmesg 2>/dev/null | grep -iE 'kgsl|adreno|a6xx|gpu' | tail -20
  cat /sys/class/kgsl/kgsl-3d0/gpu_model /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq 2>&1
  echo "--- firmware ---"
  ls -l /lib/firmware/qcom/*gpu* /lib/firmware/qcom/*zap* /lib/firmware/qcom/*a6* 2>&1 | head -20
  echo "--- clinfo ---"
  if command -v clinfo >/dev/null 2>&1; then
    clinfo 2>&1 | grep -iE 'Platform (Name|Version)|Device (Name|Version)|Driver Version|Compiler' | head -20
  else
    echo "clinfo not installed"
  fi
  echo "--- env ---"
  env | grep -iE 'OCL|OPENCL|LD_LIBRARY|ADSP|GGML|CL_' | sort
  echo "=== end GPU debug ($1) ==="
}

install_adreno_driver() {
  if ! command -v dpkg >/dev/null 2>&1; then
    echo "adreno: no dpkg on this image, leaving the installed driver as-is"
    return 0
  fi
  if dpkg -s qcom-adreno1 2>/dev/null | grep -q "^Version: ${ADRENO_VER}$" \
      && [ ! -e /usr/lib/libOpenCL.so.1 ]; then
    echo "adreno: qcom-adreno1 ${ADRENO_VER} already installed"
    return 0
  fi
  local deb="/data/local/tmp/${ADRENO_DEB}"
  echo "adreno: downloading $ADRENO_URL"
  curl -L -fS --retry 3 --retry-delay 5 -o "$deb" "$ADRENO_URL" \
    || { echo "adreno: download failed"; return 0; }
  ls -l "$deb"
  if [ "$(sha256sum "$deb" | cut -d' ' -f1)" != "$ADRENO_SHA256" ]; then
    echo "adreno: sha256 mismatch, not installing"
    return 0
  fi
  # The package Conflicts/Replaces the generic OpenCL ICD loader; force past it.
  dpkg -i --force-overwrite --force-conflicts --force-depends "$deb" \
    || { echo "adreno: dpkg install failed"; return 0; }
  # A stray non-multiarch loader would still shadow the packaged one.
  for f in /usr/lib/libOpenCL.so*; do
    [ -e "$f" ] && { echo "adreno: moving aside $f"; mv "$f" "$f.qdc-bak"; }
  done
  ldconfig 2>/dev/null || true
}

gpu_debug_dump before
install_adreno_driver
gpu_debug_dump after

cd "$BUNDLE_DIR" || { echo "FATAL: bundle missing at $BUNDLE_DIR"; exit 1; }
chmod +x bin/* 2>/dev/null
export LD_LIBRARY_PATH="$BUNDLE_DIR/lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$BUNDLE_DIR/lib"
export GGML_HEXAGON_EXPERIMENTAL=1

echo "=== download model ==="
MODEL_URL="{MODEL_URL}"
if [ -z "$MODEL_URL" ]; then
  echo "No model URL provided, skipping download"
elif [ ! -f "$MODEL_PATH" ]; then
  curl -L -fS --retry 3 --retry-delay 5 -o "$MODEL_PATH" "$MODEL_URL"
  curl_rc=$?
  if [ $curl_rc -ne 0 ]; then
    echo "FATAL: model download failed (rc=$curl_rc)"
    exit 1
  fi
  ls -la "$MODEL_PATH"
fi

# ---------------------------------------------------------------------------
# JUnit XML helpers
# ---------------------------------------------------------------------------

xml_open() {
  printf '%s\n' \
    '<?xml version="1.0" encoding="utf-8"?>' \
    "<testsuites>" \
    "<testsuite name=\"llama_cpp_linux\">" \
    > "$RESULTS_XML"
}

xml_close() {
  printf '%s\n' '</testsuite>' '</testsuites>' >> "$RESULTS_XML"
}

xml_case_pass() {
  local classname=$1 name=$2
  printf '<testcase classname="%s" name="%s"/>\n' "$classname" "$name" >> "$RESULTS_XML"
}

xml_case_fail() {
  local classname=$1 name=$2 rc=$3 logfile=$4
  {
    printf '<testcase classname="%s" name="%s">\n' "$classname" "$name"
    printf '<failure message="exit %s"><![CDATA[\n' "$rc"
    tail -c 4096 "$logfile" 2>/dev/null | sed 's/]]>/]] >/g'
    printf '\n]]></failure>\n</testcase>\n'
  } >> "$RESULTS_XML"
}

# Map backend name -> "NDEV --device" pair. "none" means no offload (CPU).
backend_env() {
  case "$1" in
    cpu) echo "0 none" ;;
    gpu) echo "0 GPUOpenCL" ;;
    npu) echo "1 HTP0" ;;
  esac
}

backend_log_name() {
  case "$1" in
    cpu) echo "cpu" ;;
    gpu) echo "gpu" ;;
    npu) echo "htp" ;;
  esac
}


backend_device_name() {
  case "$1" in
    cpu) echo "none" ;;
    gpu) echo "GPUOpenCL" ;;
    npu) echo "HTP0" ;;
  esac
}

# Append a diagnostic block when a per-case `timeout N` fires (rc=124). The
# naked log file at that point usually just ends mid-OpenCL-init with no
# stderr, which is hard to read in CI summaries.
note_timeout_if_triggered() {
  local rc=$1 budget=$2 log=$3
  [ "$rc" -eq 124 ] || return 0
  {
    printf '\n'
    printf '=== TIMEOUT after %ss ===\n' "$budget"
    printf 'uptime: '; uptime 2>/dev/null
    printf 'free -m:\n'; free -m 2>/dev/null
    printf 'loadavg: '; cat /proc/loadavg 2>/dev/null
  } >> "$log"
}

completion_extra_args() {
  case "$1" in
    cpu) echo "--device none --ctx-size 2048 -no-cnv -n 32 --seed 42" ;;
    gpu) echo "--device GPUOpenCL --ctx-size 2048 -no-cnv -n 32 --seed 42" ;;
    npu) echo "--device HTP0 --ctx-size 2048 -no-cnv -n 32 --seed 42 --ubatch-size 1024" ;;
  esac
}

run_completion_case() {
  local name=$1
  local parts=($(backend_env "$name"))
  local ndev=${parts[0]} device=${parts[1]}
  local device_log_name=$(backend_device_name "$name")
  local log="$LOG_DIR/llama_completion_${device_log_name}.log"
  local prompt="$LOG_DIR/bench_prompt.txt"
  echo 'What is the capital of France?' > "$prompt"
  local extra
  extra=$(completion_extra_args "$name")
  echo "=== [completion:$name] llama-completion --device $device (NDEV=$ndev) ==="
  timeout 600 env GGML_HEXAGON_NDEV=$ndev ./bin/llama-completion \
      -m "$MODEL_PATH" \
      -f "$prompt" \
      $extra \
      > "$log" 2>&1 < /dev/null
  local rc=$?
  note_timeout_if_triggered "$rc" 600 "$log"
  if [ $rc -eq 0 ]; then
    xml_case_pass "tests.test_bench_posix" "test_llama_completion[$name]"
  else
    xml_case_fail "tests.test_bench_posix" "test_llama_completion[$name]" "$rc" "$log"
  fi
}

run_bench_case() {
  local name=$1
  local parts=($(backend_env "$name"))
  local ndev=${parts[0]} device=${parts[1]}
  local log_suffix=$(backend_log_name "$name")
  local log="$LOG_DIR/llama_bench_${log_suffix}.log"
  local ubatch_arg=""
  [ "$name" = "npu" ] && ubatch_arg="--ubatch-size 1024"
  echo "=== [bench:$name] llama-bench --device $device (NDEV=$ndev) ==="
  timeout 600 env GGML_HEXAGON_NDEV=$ndev ./bin/llama-bench \
      -m "$MODEL_PATH" \
      --device "$device" \
      -ngl 99 \
      $ubatch_arg \
      -t 4 \
      -p 128 \
      -n 32 \
      > "$log" 2>&1
  local rc=$?
  note_timeout_if_triggered "$rc" 600 "$log"
  if [ $rc -eq 0 ]; then
    xml_case_pass "tests.test_bench_posix" "test_llama_bench[$name]"
  else
    xml_case_fail "tests.test_bench_posix" "test_llama_bench[$name]" "$rc" "$log"
  fi
}

run_backend_ops_case() {
  local dtype=$1
  local log="$LOG_DIR/backend_ops_${dtype}.log"
  local pattern
  case "$dtype" in
    q4_0)
      # Matches Android: exclude a known-broken shape on NPU.
      pattern='^(?=.*type_a=q4_0)(?!.*type_b=f32,m=576,n=512,k=576).*$'
      ;;
    *)
      pattern="type_a=${dtype}"
      ;;
  esac
  echo "=== [backend-ops:$dtype] test-backend-ops -b HTP0 -o MUL_MAT ==="
  timeout 600 env GGML_HEXAGON_NDEV=1 GGML_HEXAGON_HOSTBUF=0 ./bin/test-backend-ops \
      -b HTP0 -o MUL_MAT -p "$pattern" \
      > "$log" 2>&1
  local rc=$?
  note_timeout_if_triggered "$rc" 600 "$log"
  if [ $rc -eq 0 ]; then
    xml_case_pass "tests.test_backend_ops_posix" "test_backend_ops_htp0[$dtype]"
  else
    xml_case_fail "tests.test_backend_ops_posix" "test_backend_ops_htp0[$dtype]" "$rc" "$log"
  fi
}

xml_open

case "{TEST_MODE}" in
  bench)
    for b in cpu gpu npu; do run_completion_case "$b"; done
    for b in cpu gpu npu; do run_bench_case "$b"; done
    ;;
  backend-ops)
    for d in mxfp4 fp16 q4_0; do run_backend_ops_case "$d"; done
    ;;
  all)
    for b in cpu gpu npu; do run_completion_case "$b"; done
    for b in cpu gpu npu; do run_bench_case "$b"; done
    for d in mxfp4 fp16 q4_0; do run_backend_ops_case "$d"; done
    ;;
  *)
    echo "FATAL: unsupported TEST_MODE={TEST_MODE}"
    ;;
esac

xml_close
echo "=== done ==="
# Host parses results.xml to decide pass/fail.
exit 0

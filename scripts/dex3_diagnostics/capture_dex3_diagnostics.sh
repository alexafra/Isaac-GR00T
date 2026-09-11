#!/usr/bin/env bash
# Passive Dex3 DDS + packet capture wrapper.
#
# This script never publishes DDS commands.  It also refuses to start, and
# aborts its own diagnostic children, if a known teleoperation/evaluation
# process appears on this host.  It cannot detect command publishers on a
# different computer.

set -Eeuo pipefail
umask 077

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly ANALYZER="$SCRIPT_DIR/analyze_rtps_pcap.py"
readonly DEFAULT_PROBE_MODULE="unitree_lerobot.eval_robot.diagnose_dex3_dropouts"
readonly DEFAULT_CAPTURE_FILTER='udp and udp[8:4] = 0x52545053'
readonly probe_module="$DEFAULT_PROBE_MODULE"

declare -a original_argv=("$@")

network_interface=""
duration="300"
status_hz="1"
output_dir=""
python_bin="${DEX3_DIAGNOSTIC_PYTHON:-python}"
snaplen="512"
capture_filter="$DEFAULT_CAPTURE_FILTER"
pcap_enabled=1
trace_callbacks=0
trace_every="100"
dds_status=0
all_udp=0

output_ready=0
capture_started=0
normal_completion=0
tcpdump_pid=""
tcpdump_status=""
tcpdump_packets_captured=""
tcpdump_packets_received=""
tcpdump_kernel_drops=""
probe_pid=""
guard_pid=""
capture_start_epoch=""

usage() {
    cat <<'EOF'
Usage:
  capture_dex3_diagnostics.sh --network-interface IFACE [options]

Options:
  --duration SEC           Duration in whole seconds, 1..3600 (default: 300)
  --status-hz HZ           Diagnostic heartbeat display rate (default: 1)
  --output-dir DIR         New output directory (default: ./dex3_diagnostics/<UTC>)
  --python PATH            Python interpreter for the diagnostic and analyzer
  --snaplen BYTES          PCAP snap length, 128..65535 (default: 512)
  --all-udp                Capture every UDP packet instead of RTPS-magic packets
  --no-pcap                Run the subscriber-only diagnostic without tcpdump
  --trace-callbacks        Save a sparse callback CSV (default: every 100th callback)
  --trace-every N          Callback trace sampling interval; 1 is full trace
  --dds-status             Enable the probe's private Cyclone DDS status callbacks
  -h, --help               Show this help

Safety:
  The wrapper refuses to coexist with known local teleop, policy evaluation,
  replay, authority-ramp, or other actuation processes. It never kills those
  processes; if one appears, it stops only its own probe/tcpdump children.
  This host-local guard cannot see a command publisher running on another PC.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

need_value() {
    [[ $# -ge 2 ]] || die "$1 requires a value"
}

while (($#)); do
    case "$1" in
        --network-interface)
            need_value "$@"
            network_interface=$2
            shift 2
            ;;
        --duration)
            need_value "$@"
            duration=$2
            shift 2
            ;;
        --status-hz)
            need_value "$@"
            status_hz=$2
            shift 2
            ;;
        --output-dir)
            need_value "$@"
            output_dir=$2
            shift 2
            ;;
        --python)
            need_value "$@"
            python_bin=$2
            shift 2
            ;;
        --snaplen)
            need_value "$@"
            snaplen=$2
            shift 2
            ;;
        --all-udp)
            capture_filter="udp"
            all_udp=1
            shift
            ;;
        --no-pcap)
            pcap_enabled=0
            shift
            ;;
        --trace-callbacks)
            trace_callbacks=1
            shift
            ;;
        --trace-every)
            need_value "$@"
            trace_every=$2
            trace_callbacks=1
            shift 2
            ;;
        --dds-status)
            dds_status=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n $network_interface ]] || die "--network-interface is required"
[[ $network_interface != "lo" ]] || die "the loopback interface is not a robot DDS interface"
[[ $network_interface != *"/"* && $network_interface != *$'\n'* ]] || die "invalid interface name"
[[ -d /sys/class/net/$network_interface ]] || die "network interface does not exist: $network_interface"
[[ $duration =~ ^[0-9]+$ ]] || die "--duration must be a whole number of seconds"
((duration >= 1 && duration <= 3600)) || die "--duration must be in 1..3600"
[[ $status_hz =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || die "--status-hz must be positive"
awk -v value="$status_hz" 'BEGIN { exit !(value > 0 && value <= 100) }' \
    || die "--status-hz must be in (0, 100]"
[[ $snaplen =~ ^[0-9]+$ ]] || die "--snaplen must be an integer"
((snaplen >= 128 && snaplen <= 65535)) || die "--snaplen must be in 128..65535"
[[ $probe_module =~ ^[A-Za-z_][A-Za-z0-9_.]*$ ]] || die "invalid probe module name"
[[ $trace_every =~ ^[0-9]+$ ]] || die "--trace-every must be an integer"
((trace_every >= 1 && trace_every <= 100000)) || die "--trace-every must be in 1..100000"
if ((trace_callbacks && trace_every == 1 && duration > 60)); then
    die "full callback tracing (--trace-every 1) is limited to 60 seconds; use sparse tracing for longer runs"
fi

if [[ $python_bin == */* ]]; then
    [[ -x $python_bin ]] || die "Python is not executable: $python_bin"
else
    python_bin=$(command -v -- "$python_bin") || die "Python was not found: $python_bin"
fi
[[ -f $ANALYZER ]] || die "offline analyzer is missing: $ANALYZER"

# Output is deliberately plain text and never evaluated. Tabs/newlines in a
# hostile argv are replaced so a process cannot forge additional guard lines.
list_active_control_processes() {
    local -a excluded=("$@")
    local -a process_argv=()
    local proc pid cmdline comm skip excluded_pid argument
    for proc in /proc/[0-9]*; do
        [[ -d $proc ]] || continue
        pid=${proc##*/}
        [[ $pid != "$$" ]] || continue
        skip=0
        for excluded_pid in "${excluded[@]}"; do
            if [[ -n $excluded_pid && $pid == "$excluded_pid" ]]; then
                skip=1
                break
            fi
        done
        ((skip == 0)) || continue

        cmdline=""
        process_argv=()
        if [[ -r $proc/cmdline ]]; then
            while IFS= read -r -d '' argument; do
                argument=${argument//$'\n'/ }
                argument=${argument//$'\r'/ }
                argument=${argument//$'\t'/ }
                process_argv+=("$argument")
            done <"$proc/cmdline" 2>/dev/null || true
            if ((${#process_argv[@]})); then
                printf -v cmdline '%s ' "${process_argv[@]}"
            fi
        fi
        comm=""
        if [[ -r $proc/comm ]]; then
            IFS= read -r comm <"$proc/comm" || true
        fi
        [[ -n $cmdline || -n $comm ]] || continue
        case "$cmdline $comm" in
            *teleop_hand_and_arm.py*|*teleop.teleop_hand_and_arm*|\
            *unitree_lerobot.eval_robot.eval_groot_g1*|*eval_groot_g1.py*|\
            *unitree_lerobot.eval_robot.eval_g1*|*/eval_g1.py*|\
            *replay_robot.py*|*unitree_lerobot.eval_robot.replay_robot*|\
            *zero_state_test.py*|*unitree_lerobot.eval_robot.zero_state_test*|\
            *diagnose_g1_authority_ramp*|*diagnose_g1_dds_hold*|\
            *robot_hand_unitree.py*|*unitree_lerobot.eval_robot.probe_dex3_ranges*|\
            *unitree_lerobot.eval_robot.diagnose_dex3_dropouts*)
                printf '%s\t%s\n' "$pid" "$cmdline"
                continue
                ;;
        esac
        if [[ $cmdline == *--actuate* && $cmdline == *unitree* ]]; then
            printf '%s\t%s\n' "$pid" "$cmdline"
        fi
    done
}

assert_no_active_control() {
    local matches
    matches=$(list_active_control_processes "$@")
    if [[ -n $matches ]]; then
        printf 'REFUSING: a local robot-control/teleop process is active:\n%s\n' "$matches" >&2
        printf 'Stop it explicitly and verify no other PC is publishing commands.\n' >&2
        return 1
    fi
}

# This is the first operation that inspects run-time state.  It reads only the
# local process table and must pass before any interface, tcpdump, or DDS work.
assert_no_active_control || exit 3

if [[ -z $output_dir ]]; then
    output_dir="$PWD/dex3_diagnostics/dex3_$(date -u +%Y%m%dT%H%M%S.%NZ)"
fi
output_dir=$(realpath -m -- "$output_dir")
[[ $output_dir != "/" ]] || die "refusing output directory /"
[[ ! -e $output_dir && ! -L $output_dir ]] || die "output directory already exists: $output_dir"
output_parent=$(dirname -- "$output_dir")
mkdir -p -- "$output_parent"
[[ ! -L $output_parent ]] || die "output parent may not be a symlink: $output_parent"
mkdir -m 700 -- "$output_dir"
output_ready=1

metadata_file="$output_dir/metadata.txt"
pre_snapshot_file="$output_dir/host_before.txt"
post_snapshot_file="$output_dir/host_after.txt"
probe_log="$output_dir/probe.log"
probe_events="$output_dir/probe_events.jsonl"
probe_summary="$output_dir/probe_summary.json"
trace_csv="$output_dir/callback_trace.csv"
pcap_file="$output_dir/rtps.pcap"
tcpdump_log="$output_dir/tcpdump.log"
abort_marker="$output_dir/ABORTED_ACTIVE_CONTROL.txt"
capture_failed_marker="$output_dir/CAPTURE_FAILED.txt"

quote_command() {
    local argument
    for argument in "$@"; do
        printf ' %q' "$argument"
    done
    printf '\n'
}

run_logged() {
    local destination=$1 label=$2
    shift 2
    {
        printf '\n[%s]\ncommand:' "$label"
        quote_command "$@"
    } >>"$destination"
    local status=0
    "$@" >>"$destination" 2>&1 || status=$?
    printf 'exit_status=%d\n' "$status" >>"$destination"
    return 0
}

snapshot_host() {
    local phase=$1 destination=$2
    : >"$destination"
    printf 'phase=%s\n' "$phase" >>"$destination"
    run_logged "$destination" date_utc date -u --iso-8601=ns
    run_logged "$destination" uname uname -a
    run_logged "$destination" os_release sed -n '1,160p' /etc/os-release
    run_logged "$destination" ip_link ip -s -d link show dev "$network_interface"
    run_logged "$destination" ip_address ip address show dev "$network_interface"
    run_logged "$destination" ip_route ip route show table all
    if command -v ethtool >/dev/null 2>&1; then
        run_logged "$destination" ethtool_driver ethtool -i "$network_interface"
        run_logged "$destination" ethtool_features ethtool -k "$network_interface"
        run_logged "$destination" ethtool_stats ethtool -S "$network_interface"
    fi
    if command -v nstat >/dev/null 2>&1; then
        run_logged "$destination" nstat nstat -az
    fi
    run_logged "$destination" softnet sed -n '1,512p' /proc/net/softnet_stat
    run_logged "$destination" proc_udp sed -n '1,512p' /proc/net/udp
    run_logged "$destination" proc_udp6 sed -n '1,512p' /proc/net/udp6
    if command -v ss >/dev/null 2>&1; then
        run_logged "$destination" udp_sockets ss -uapn
    fi
    run_logged "$destination" interrupts sed -n '1,512p' /proc/interrupts
    run_logged "$destination" processes ps -eo pid,ppid,lstart,stat,psr,pcpu,pmem,args
    if [[ $phase == "after" && -n $capture_start_epoch ]] && command -v journalctl >/dev/null 2>&1; then
        run_logged "$destination" kernel_journal journalctl -k --since "@$capture_start_epoch" --no-pager
    fi
}

stop_child() {
    local pid=${1:-} signal=${2:-INT}
    [[ -n $pid ]] || return 0
    if kill -0 "$pid" 2>/dev/null; then
        kill -s "$signal" "$pid" 2>/dev/null || true
        local iteration
        for iteration in {1..20}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    fi
    wait "$pid" 2>/dev/null || true
}

signal_tcpdump() {
    local signal=${1:-INT}
    [[ -n ${tcpdump_pid:-} ]] || return 0
    # Keep sudo attached to this terminal so its tty-scoped credential from
    # `sudo -v` remains valid. sudo relays signals to its running command.
    kill -s "$signal" "$tcpdump_pid" 2>/dev/null || true
}

stop_tcpdump() {
    local pid=${tcpdump_pid:-}
    [[ -n $pid ]] || return 0
    if kill -0 "$pid" 2>/dev/null; then
        signal_tcpdump INT
        local iteration
        for iteration in {1..50}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 0.1
        done
        if kill -0 "$pid" 2>/dev/null; then
            signal_tcpdump TERM
        fi
    fi
    if wait "$pid" 2>/dev/null; then
        tcpdump_status=0
    else
        tcpdump_status=$?
    fi
    tcpdump_pid=""
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    stop_child "$guard_pid" TERM
    stop_child "$probe_pid" INT
    stop_tcpdump
    if ((output_ready)) && ((capture_started)) && ((normal_completion == 0)); then
        printf 'wrapper_exit_status=%d\n' "$status" >>"$metadata_file" 2>/dev/null || true
    fi
    exit "$status"
}

on_interrupt() {
    exit 130
}

trap cleanup EXIT
trap on_interrupt INT TERM

{
    printf 'wrapper_started_utc='
    date -u --iso-8601=ns
    printf 'wrapper_pid=%d\n' "$$"
    printf 'cwd=%q\n' "$PWD"
    printf 'network_interface=%q\n' "$network_interface"
    printf 'duration_s=%q\n' "$duration"
    printf 'status_hz=%q\n' "$status_hz"
    printf 'python=%q\n' "$python_bin"
    printf 'probe_module=%q\n' "$probe_module"
    printf 'pcap_enabled=%d\n' "$pcap_enabled"
    printf 'snaplen=%q\n' "$snaplen"
    printf 'capture_filter=%q\n' "$capture_filter"
    printf 'trace_callbacks=%d\n' "$trace_callbacks"
    printf 'trace_every=%q\n' "$trace_every"
    printf 'dds_status=%d\n' "$dds_status"
    printf 'wrapper_command:'
    quote_command "$0" "${original_argv[@]}"
    printf 'safety_scope=local_host_only\n'
} >"$metadata_file"

if ((pcap_enabled)); then
    printf 'WARNING: PCAP capture has no total-byte limit and can fill its filesystem; snaplen bounds each packet only. Free space is recorded below.\n' \
        | tee -a "$metadata_file" >&2
    run_logged "$metadata_file" output_filesystem_space df -h "$output_parent"
fi
if ((all_udp)); then
    printf 'WARNING: --all-udp broadens capture beyond RTPS and can increase disk use substantially.\n' \
        | tee -a "$metadata_file" >&2
fi

run_logged "$metadata_file" python_version "$python_bin" --version
run_logged "$metadata_file" python_packages "$python_bin" -c \
    'import importlib.metadata as m; print("unitree_sdk2py",m.version("unitree_sdk2py")); print("cyclonedds",m.version("cyclonedds"))'
for repository in \
    /home/alex/Development/unitree_sdk2_python \
    /home/alex/Development/xr_teleoperate \
    /home/alex/Development/unitree_lerobot; do
    if [[ -d $repository/.git ]]; then
        run_logged "$metadata_file" "git_head_$repository" git -C "$repository" rev-parse HEAD
        run_logged "$metadata_file" "git_status_$repository" git -C "$repository" status --short
    fi
done

# Locate the module without importing it, then enforce the subscriber-only
# invariant directly against its source before launching it.
probe_source=$(
    "$python_bin" - "$probe_module" <<'PY'
import importlib.machinery
import sys

name = sys.argv[1]
path = None
spec = None
for part in name.split("."):
    spec = importlib.machinery.PathFinder.find_spec(part, path)
    if spec is None:
        raise SystemExit(f"cannot find module {name}")
    path = spec.submodule_search_locations
origin = getattr(spec, "origin", None)
if not origin or origin in {"built-in", "frozen"}:
    raise SystemExit(f"module has no inspectable source: {name}")
print(origin)
PY
)
[[ -f $probe_source ]] || die "probe source is not a file: $probe_source"
grep -qi 'subscriber.only' "$probe_source" || die "probe source lacks its subscriber-only declaration"
if grep -Eq 'ChannelPublisher[[:space:]]*\(' "$probe_source"; then
    die "probe source constructs ChannelPublisher; refusing to run it"
fi
printf 'probe_source=%q\n' "$probe_source" >>"$metadata_file"

assert_no_active_control || exit 3
snapshot_host before "$pre_snapshot_file"
assert_no_active_control || exit 3

if ((pcap_enabled)); then
    command -v tcpdump >/dev/null 2>&1 || die "tcpdump is required unless --no-pcap is used"
    tcpdump -d "$capture_filter" >/dev/null 2>&1 || die "tcpdump rejected the capture filter"
fi

declare -a tcpdump_prefix=()
if ((pcap_enabled)); then
    invoking_user=${SUDO_USER:-$(id -un)}
    [[ $invoking_user =~ ^[A-Za-z_][A-Za-z0-9_.-]*$ ]] || die "invalid invoking user name"
    if ((EUID == 0)); then
        tcpdump_prefix=(tcpdump -Z "$invoking_user")
    else
        command -v sudo >/dev/null 2>&1 \
            || die "tcpdump needs root capture privileges. Install sudo or run as root."
        if ! sudo -n true >/dev/null 2>&1; then
            die "tcpdump needs a cached sudo credential. Run 'sudo -v' in this terminal, then rerun this wrapper."
        fi
        tcpdump_prefix=(sudo -n -- tcpdump -Z "$invoking_user")
    fi
    printf 'tcpdump_invoking_user=%q\n' "$invoking_user" >>"$metadata_file"
fi

declare -a probe_command=(
    "$python_bin" -u -m "$probe_module"
    --network-interface "$network_interface"
    --duration "$duration"
    --status-hz "$status_hz"
    --events-jsonl "$probe_events"
    --summary-json "$probe_summary"
)
if ((trace_callbacks)); then
    probe_command+=(--trace-csv "$trace_csv" --trace-every "$trace_every")
fi
if ((dds_status)); then
    probe_command+=(--dds-status)
fi
capture_start_epoch=$(date +%s)
printf 'capture_started_epoch=%s\n' "$capture_start_epoch" >>"$metadata_file"
capture_started=1

if ((pcap_enabled)); then
    assert_no_active_control || exit 3
    : >"$pcap_file"
    chmod 600 "$pcap_file"
    "${tcpdump_prefix[@]}" \
        -p \
        -Q in \
        -i "$network_interface" \
        -n \
        -s "$snaplen" \
        -B 65536 \
        -w "$pcap_file" \
        "$capture_filter" \
        >"$output_dir/tcpdump.stdout" 2>"$tcpdump_log" &
    tcpdump_pid=$!
    sleep 0.5
    if ! kill -0 "$tcpdump_pid" 2>/dev/null; then
        stop_tcpdump
        printf 'tcpdump_exit_status=%s\n' "$tcpdump_status" >>"$metadata_file"
        die "tcpdump exited before the probe (status $tcpdump_status); see $tcpdump_log"
    fi
fi

assert_no_active_control "$tcpdump_pid" || exit 3
printf 'probe_command:' >>"$metadata_file"
quote_command "${probe_command[@]}" >>"$metadata_file"
"${probe_command[@]}" >"$probe_log" 2>&1 &
probe_pid=$!

guard_monitor() {
    local matches
    while true; do
        if ((pcap_enabled)) && ! kill -0 "$tcpdump_pid" 2>/dev/null; then
            {
                printf 'tcpdump exited while the subscriber probe was still running.\n'
                printf 'No wire-level gap classification is valid for this capture.\n'
            } >"$capture_failed_marker"
            # A background process may inherit SIGINT ignored from its shell.
            # TERM is bounded to the probe PID that this wrapper launched.
            kill -TERM "$probe_pid" 2>/dev/null || true
            return 98
        fi
        matches=$(list_active_control_processes "$tcpdump_pid" "$probe_pid")
        if [[ -n $matches ]]; then
            {
                printf 'A local control process appeared; diagnostic children were stopped.\n'
                printf '%s\n' "$matches"
                printf 'The control process itself was not signalled.\n'
            } >"$abort_marker"
            kill -TERM "$probe_pid" 2>/dev/null || true
            signal_tcpdump INT
            return 99
        fi
        sleep 1
    done
}

guard_monitor &
guard_pid=$!

probe_status=0
wait "$probe_pid" || probe_status=$?
probe_pid=""
stop_child "$guard_pid" TERM
guard_pid=""
stop_tcpdump

if [[ -e $abort_marker ]]; then
    printf 'Diagnostic aborted because control/teleop started; see %s\n' "$abort_marker" >&2
    exit 4
fi
if [[ -e $capture_failed_marker ]]; then
    printf 'Capture failed while the probe was active; see %s and %s\n' \
        "$capture_failed_marker" "$tcpdump_log" >&2
    exit 5
fi
if ((probe_status != 0)); then
    printf 'Probe failed with status %d; see %s\n' "$probe_status" "$probe_log" >&2
    exit "$probe_status"
fi

assert_no_active_control || exit 3
snapshot_host after "$post_snapshot_file"

if ((pcap_enabled)); then
    printf 'tcpdump_exit_status=%s\n' "$tcpdump_status" >>"$metadata_file"
    if [[ $tcpdump_status != 0 && $tcpdump_status != 130 ]]; then
        die "tcpdump exited with unexpected status $tcpdump_status; see $tcpdump_log"
    fi
    tcpdump_packets_captured=$(
        awk '/ packets captured$/ {value=$1} END {if (value != "") print value}' "$tcpdump_log"
    )
    tcpdump_packets_received=$(
        awk '/ packets received by filter$/ {value=$1} END {if (value != "") print value}' "$tcpdump_log"
    )
    tcpdump_kernel_drops=$(
        awk '/ packets dropped by kernel$/ {value=$1} END {if (value != "") print value}' "$tcpdump_log"
    )
    [[ $tcpdump_packets_captured =~ ^[0-9]+$ ]] \
        || die "tcpdump did not report its captured-packet count; see $tcpdump_log"
    [[ $tcpdump_packets_received =~ ^[0-9]+$ ]] \
        || die "tcpdump did not report its received-by-filter count; see $tcpdump_log"
    [[ $tcpdump_kernel_drops =~ ^[0-9]+$ ]] \
        || die "tcpdump did not report its kernel-drop count; see $tcpdump_log"
    [[ -r $pcap_file ]] || die "captured PCAP is not readable by the invoking user: $pcap_file"
    {
        printf 'tcpdump_packets_captured=%s\n' "$tcpdump_packets_captured"
        printf 'tcpdump_packets_received_by_filter=%s\n' "$tcpdump_packets_received"
        printf 'tcpdump_packets_dropped_by_kernel=%s\n' "$tcpdump_kernel_drops"
    } >>"$metadata_file"
    run_logged "$metadata_file" pcap_file stat -- "$pcap_file"
    if ((tcpdump_kernel_drops > 0)); then
        printf 'WARNING: tcpdump reported %s kernel drops; positive wire-continuity evidence is disabled.\n' \
            "$tcpdump_kernel_drops" | tee -a "$metadata_file" >&2
    fi
    declare -a analyzer_command=(
        "$python_bin" "$ANALYZER" "$pcap_file"
        --output-dir "$output_dir/pcap_analysis"
        --capture-kernel-drops "$tcpdump_kernel_drops"
    )
    if ((trace_callbacks)); then
        [[ -s $trace_csv ]] || die "probe succeeded but callback trace is missing or empty: $trace_csv"
        [[ -s $probe_summary ]] || die "probe succeeded but summary JSON is missing or empty: $probe_summary"
        analyzer_command+=(
            --callbacks-csv "$trace_csv"
            --probe-summary-json "$probe_summary"
        )
    fi
    printf 'analyzer_command:' >>"$metadata_file"
    quote_command "${analyzer_command[@]}" >>"$metadata_file"
    "${analyzer_command[@]}" >"$output_dir/analyzer.log" 2>&1
fi

normal_completion=1
{
    printf 'wrapper_finished_utc='
    date -u --iso-8601=ns
    printf 'wrapper_exit_status=0\n'
} >>"$metadata_file"
printf 'Dex3 passive diagnostic complete: %s\n' "$output_dir"

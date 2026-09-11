# Dex3 one-second dropout runbook

This runbook isolates the recurring approximately 1.004-second loss of one Dex3
state stream. The tools here are passive: they subscribe to state topics and,
optionally, capture inbound RTPS packets. They do not publish robot commands.

The evidence so far points most strongly to a per-hand source path (hand bus,
robot-side acquisition/bridge, or its publisher), rather than a whole-PC or
whole-network stall: either hand can pause while the other hand and
`rt/lowstate` continue. The tests below separate those possibilities; they do
not assume the cause in advance.

## Safety gate

Do not run a diagnostic during the current teleoperation, policy, replay, or
actuation session. Before every run:

1. Stop teleoperation/evaluation explicitly and put the robot in its normal
   supervised safe state (harness, operator, and kill switch available).
2. Verify that no other computer is publishing robot or hand commands. The only
   exception is the separately scheduled Stage 4 motion-correlation test,
   where the probe runs on an observer PC and the operator knowingly drives
   from a different PC.
3. Use the correct wired robot interface. Here that is `enp132s0`; do not use
   Wi-Fi or loopback.
4. Use a new output directory. The tools deliberately refuse to overwrite an
   existing one.

The wrapper checks the local process table before starting and every second
during a run. If a known local control process appears, it stops only its own
probe/capture children. This protection is host-local: it cannot see a command
publisher on another PC. Do not start the separately scheduled Stage 4 motion
comparison during the current session.

Do **not** restart a hand service, kill an unknown robot process, change DDS QoS,
or update firmware as an exploratory step. First collect evidence and identify
the exact component. In particular, `unitree_dex3_service` is not a verified
service name on this robot, and the public `dex1_1_service` does not establish
the Dex3 architecture.

## What is measured

The subscriber probe watches five independent streams:

| Probe name | DDS topic | Default gap threshold |
|---|---|---:|
| `hf_left` | `rt/dex3/left/state` | 75 ms |
| `hf_right` | `rt/dex3/right/state` | 75 ms |
| `lf_left` | `rt/lf/dex3/left/state` | 350 ms |
| `lf_right` | `rt/lf/dex3/right/state` | 350 ms |
| `lowstate` | `rt/lowstate` | 75 ms |

It records callback and valid-sample gaps independently, exact monotonic and UTC
gap intervals, source timestamps, DDS publication handles, and raw hand health
fields. The LF feeds are diagnostic references; at roughly 9 Hz they are not a
drop-in source for the current 100 ms policy freshness requirement.

The packet analyzer reads a classic PCAP offline and tracks each RTPS writer's
GUID and sequence numbers. A sparse callback trace maps writers to topics; the
probe summary supplies the exact unsampled gap intervals.

## Stage 1: lightweight event-only baseline

Run this for 10 minutes only after the safety gate passes:

```bash
cd /home/alex/Development/unitree_lerobot
run_dir="/tmp/dex3_event_$(date -u +%Y%m%dT%H%M%SZ)"
bash /home/alex/Development/Isaac-GR00T/scripts/dex3_diagnostics/capture_dex3_diagnostics.sh \
  --network-interface enp132s0 \
  --duration 600 \
  --python /home/alex/miniconda3/envs/unitree_lerobot/bin/python \
  --output-dir "$run_dir" \
  --no-pcap
```

Inspect these files after the command finishes:

- `probe_summary.json`: counts, rates, exact gaps, publication handles, and
  HF/LF/lowstate overlap totals.
- `probe_events.jsonl`: each gap opening/recovery and any health or handle
  change, with exact times.
- `probe.log`: heartbeat and human-readable final summary.
- `host_before.txt` and `host_after.txt`: NIC, kernel, process, and counter
  snapshots.

The baseline answers whether the failure is unilateral, whether HF and LF fail
together on the same side, whether `lowstate` also stops, and whether the DDS
endpoint identity changes.

## Stage 2: RTPS packet/callback correlation

This is the decisive test of the statement “the writer continues, but this
PC's CycloneDDS reader stops delivering the topic.” It is still passive. Packet
capture needs temporary privilege; refresh it once immediately before the run:

```bash
sudo -v
cd /home/alex/Development/unitree_lerobot
run_dir="/tmp/dex3_wire_$(date -u +%Y%m%dT%H%M%SZ)"
bash /home/alex/Development/Isaac-GR00T/scripts/dex3_diagnostics/capture_dex3_diagnostics.sh \
  --network-interface enp132s0 \
  --duration 300 \
  --python /home/alex/miniconda3/envs/unitree_lerobot/bin/python \
  --output-dir "$run_dir" \
  --trace-callbacks \
  --trace-every 100
```

The default filter captures inbound UDP packets beginning with RTPS magic only,
in non-promiscuous mode, with a 512-byte snap length. Do not use `--all-udp`
unless RTPS discovery/encapsulation investigation specifically requires it; it
can create a much larger capture.

The wrapper analyzes the capture automatically. The most useful outputs are:

- `pcap_analysis/rtps_writer_summary.csv`: per-writer rate, largest wire gap,
  sequence loss/reordering, and fragment completeness.
- `pcap_analysis/writer_topic_map.csv`: accepted or rejected writer-to-topic
  mappings and their confidence.
- `pcap_analysis/probe_gap_wire_evidence.csv`: wire evidence inside each exact
  probe gap.
- `pcap_analysis/analysis.json`: machine-readable combined report.
- `tcpdump.log`: capture drops/errors; a lossy capture cannot prove publisher
  silence.

If a capture already exists, analyze it without opening any network interface:

```bash
kernel_drops="$(sed -n 's/^tcpdump_packets_dropped_by_kernel=//p' \
  /tmp/dex3_wire_TIMESTAMP/metadata.txt)"
test -n "$kernel_drops"
/home/alex/miniconda3/envs/unitree_lerobot/bin/python \
  /home/alex/Development/Isaac-GR00T/scripts/dex3_diagnostics/analyze_rtps_pcap.py \
  /tmp/dex3_wire_TIMESTAMP/rtps.pcap \
  --callbacks-csv /tmp/dex3_wire_TIMESTAMP/callback_trace.csv \
  --probe-summary-json /tmp/dex3_wire_TIMESTAMP/probe_summary.json \
  --capture-kernel-drops "$kernel_drops" \
  --output-dir /tmp/dex3_wire_TIMESTAMP/offline_analysis
```

Only interpret a writer by topic when `writer_topic_map.csv` marks the mapping
accepted. Sparse trace rows identify writers; they do not define callback gaps.
The exact gap intervals always come from `probe_summary.json`.

In `probe_gap_wire_evidence.csv`,
`writer_sequences_observed_during_probe_gap` is positive evidence that the
mapped writer reached this host while the subscriber stopped delivering it.
`wire_gap_overlaps_probe_gap` says the capture also went quiet, but cannot by
itself distinguish publisher silence from loss before the capture point. Treat
`capture_reported_kernel_drops`, `insufficient_writer_topic_mapping`, and
`multiple_accepted_writers_for_topic` as inconclusive results that require a
cleaner or more informative run, not causal findings.

`--dds-status` is an optional second-pass diagnostic. It uses a private,
version-sensitive Cyclone listener API and is off by default to avoid perturbing
delivery. Under BestEffort/no-Deadline QoS, an absence of `sample_lost`,
liveliness, or deadline events is not evidence that delivery was healthy.

## Stage 3: second-PC localization

Use the same Stage 1 command concurrently on PC1 and PC2, changing
`--network-interface` to the real wired interface name on each PC and giving
each run a distinct directory. Before starting, check time synchronization on
both hosts:

```bash
timedatectl show -p NTPSynchronized --value
date --iso-8601=ns
```

Start the two 15-minute probes as close together as practical. Do not run
teleoperation or policy control during this stationary comparison. Compare the
UTC intervals in the two `probe_summary.json` files; use monotonic times only
within one host. If the event-only result is ambiguous, repeat Stage 2 on both
PCs so there is an independent PCAP at each observation point.

| Result | Meaning |
|---|---|
| Same side gaps at the same UTC time on both PCs | Failure is upstream of both PCs: robot publisher/bridge/bus is strongly favored. |
| PC1 gaps while PC2 receives continuously | PC1's NIC/kernel/Cyclone/application path is favored. |
| Both PCs gap and both captures lack the writer's DATA | Robot-side writer/acquisition pause, or loss before both taps. |
| One capture contains the writer's DATA during that PC's callback gap | Local delivery path on that PC is implicated. |

Allow for clock-sync error when comparing UTC times. Same-side overlap within
that error bound is meaningful; a raw equality test is not.

## Stage 4: stationary versus motion and power state

Do repeated, labelled comparisons rather than mixing conditions:

1. **Post-boot stationary:** after a normal power-up and service startup, run a
   15-minute Stage 1 probe with no command publisher.
2. **Warm stationary:** repeat for 15 minutes later in the same power session.
3. **Supervised motion:** after the current teleoperation session has ended,
   schedule a separate test. Run the passive probe only on PC2 while normal
   teleoperation runs on PC1. Tell the operator that the observer is active,
   retain the usual harness and kill switch, and note the exact UTC start/end
   of hand motion. Do not run the wrapper on the actuation PC.
4. Repeat enough sessions to compare events per minute, not merely “a gap was
   seen.” Record left and right separately.

Compare, per stream: `gap_count / elapsed_s * 60`, total gap duration, maximum
gap, HF/LF overlap, raw voltage/power/temperature fields, and error/motor-state
changes. Raw health fields may be zero or undocumented; treat changes as clues,
not decoded fault codes.

- More gaps under grasp load or at a repeatable arm pose supports power,
  cabling, EMI, or a hand-controller/bus issue.
- Similar rates while completely stationary weaken a load-induced explanation
  and support a timer/watchdog, firmware/bridge, or communications issue.
- A cold-only pattern supports initialization, connector, or thermal/power
  state; a warm-only pattern supports thermal or accumulated service state.

Do not repeatedly power-cycle the robot merely to provoke a fault. Compare
naturally required power cycles first.

## Stage 5: robot-side read-only evidence

Log in to the robot computer using the normal lab procedure. First discover the
real service/process names; do not guess one:

```bash
systemctl list-units --type=service --all --no-pager | grep -Ei 'dex3|dex|hand|grip|serial|unitree'
systemctl list-unit-files --no-pager | grep -Ei 'dex3|dex|hand|grip|serial|unitree'
ps -eo pid,ppid,lstart,etimes,stat,args | grep -Ei 'dex3|dex|hand|grip|serial|unitree'
```

For each plausible unit found, substitute its exact name for `<unit>`:

```bash
systemctl show '<unit>' \
  -p Id -p LoadState -p ActiveState -p SubState -p MainPID -p NRestarts \
  -p ExecMainStartTimestamp -p ExecMainExitTimestamp -p ExecMainStatus
journalctl -u '<unit>' --since '-30 min' --no-pager
```

Collect kernel evidence over the same window:

```bash
journalctl -k --since '-30 min' --no-pager \
  | grep -Ei 'usb|tty|serial|reset|disconnect|over.current|overcurrent|voltage|dex|hand'
```

Correlate log UTC timestamps with `probe_events.jsonl`. A service PID/start-time
or `NRestarts` change during a gap supports process restart. USB/serial reset,
disconnect, or power warnings at the same time support a lower-level bus/power
fault. Silence in these logs does not rule out an embedded hand bus that the
Linux kernel cannot observe.

These commands are read-only. Do not run `systemctl restart`, kill processes,
edit service files, or flash/update firmware on the strength of a guessed name.

## Hypothesis and decision matrix

| Observation | Supported explanation | Weakened explanation | Next confirmation |
|---|---|---|---|
| One HF hand stops; opposite HF and `lowstate` continue | Per-hand path | Global PC/NIC/robot-state stall | Check same-side LF and wire DATA. |
| Same-side HF and LF gaps overlap | Shared acquisition/bridge before HF/LF fan-out | HF-topic-only issue | Compare two PCs and robot logs. |
| Only HF gaps; same-side LF stays healthy | HF publisher/topic path | Physical hand bus outage | Inspect writer GUID/SN and service logs. |
| Publication handle or writer GUID changes | DDS endpoint/service recreation | A continuously running endpoint | Match service PID/`NRestarts` and logs. |
| Handle/GUID remains stable | Full endpoint recreation is less likely | Does **not** rule out a stuck producer loop | Inspect wire sequence timing. |
| Correct writer DATA/SN continues inside an exact callback gap | Local kernel/Cyclone/reader delivery path | Writer silence | Reproduce on second PC and inspect host load/capture drops. |
| Wire has a gap and next SN is contiguous | Writer probably emitted nothing during interval | Local callback-only stall | Confirm on second PC and robot process logs. |
| Wire has a gap and next SN jumps | Packets were missed by the capture path, or writer skipped SNs | Simple writer sleep | Check capture drops and a second capture point. |
| Both hands and `lowstate` gap together | Shared network/NIC/host or robot-wide source | Isolated hand hardware | Inspect NIC/kernel counters and second PC. |
| Gaps correlate with current/load/pose or USB/serial resets | Power, cable, EMI, controller, or bus fault | Pure reader scheduling issue | Controlled stationary/motion repeat and vendor hardware inspection. |
| DDS status reports nothing | No conclusion under current BestEffort/no-Deadline setup | Nothing is ruled out | Use PCAP sequence evidence. |

## Interpretation cautions

- A source-timestamp gap alone cannot distinguish writer silence from samples
  lost before this reader, especially with KeepLast behavior.
- A stable publication handle lowers the probability of endpoint recreation;
  it does not prove that the producer loop kept running.
- A missing packet in one PCAP can be publisher silence, network loss before the
  capture point, or capture loss. Check `tcpdump.log`, sequence numbers, and the
  second PC.
- DATA_FRAG submessages are fragments of one sample, not duplicate samples.
- The approximately 1.004-second regularity is compatible with a watchdog or
  retry timeout, but timing alone cannot locate that timer.
- Async subscriber initialization cannot explain repeated steady-state gaps:
  initialization occurs once, whereas these interruptions recur mid-run.

Stop after the earliest stage that discriminates the fault. Preserve the full
output directory from each run together with the robot-side log extract and the
power/motion condition label.

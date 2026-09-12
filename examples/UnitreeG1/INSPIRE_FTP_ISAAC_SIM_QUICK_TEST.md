# Inspire E2/FTP colour-only Isaac Sim quick test

This is the short follow-up to the earlier three-terminal Dex3 Isaac Lab test.
The checkpoint and model contract remain Inspire FTP: 26 values in
`7 arm + 7 arm + 6 hand + 6 hand` order, with absolute normalized hand values
in `[0, 1]`. Isaac Lab's existing Inspire model carries those same twelve hand
values over its combined simulator DDS topic; the guarded client selects that
transport only while `--sim` is active.

## Before starting

Do not run the simulator client while this workstation can reach a physical
Unitree robot. DDS domain 1 is not a physical safety boundary. Disconnect the
robot Ethernet/network, and stop any real teleop or robot command publisher.

Check that training produced a complete 20,000-step checkpoint and that its
dataset is FTP-labelled, 26D, and RGB-only:

```bash
cd /home/alex/Development/Isaac-GR00T
bash scripts/run_inspire_e2_ftp_colour_only_isaac_smoke.sh check
```

It must print `READY` before continuing.

## Terminal 1: simulator and embedded TeleImager

```bash
cd /home/alex/Development/Isaac-GR00T
bash scripts/run_inspire_e2_ftp_colour_only_isaac_smoke.sh sim
```

Click the Isaac Sim window so it renders, then wait for the image server, DDS,
and controller success messages. `sim_main.py` starts TeleImager itself; do not
start another image server.

## Terminal 2: GR00T server

```bash
cd /home/alex/Development/Isaac-GR00T
bash scripts/run_inspire_e2_ftp_colour_only_isaac_smoke.sh server
```

Wait for the policy server to report that it is listening on
`127.0.0.1:5555`.

## Terminal 3: one-action smoke first

Reconfirm that the physical robot network is disconnected, then run:

```bash
cd /home/alex/Development/Isaac-GR00T
bash scripts/run_inspire_e2_ftp_colour_only_isaac_smoke.sh client-once
```

The client checks the checkpoint's exact FTP/RGB/26D metadata before creating
publishers, asks for the exact `SIMULATE` confirmation, initializes to XR home,
performs Warmup2, and executes only one policy action. During active execution,
`s` enters a powered hold and `q` releases/exits. The physical emergency stop
is still the primary safety control for any later real-robot run.

For Inspire, the current initialization convergence test verifies arm position,
arm velocity, feedback freshness, and timing. It does not apply a qualified hand
tracking-error threshold. Visually confirm that both simulated hands reached the
expected target before pressing `r`. During blocking startup, XR-home, Warmup2,
and Return-to-Start, either `s` or `q` cancels via orderly release. During active
policy execution, `s` instead enters the acknowledged powered HOLD.

If that completes normally, reset/restart the simulator and run the short
20-chunk test:

```bash
cd /home/alex/Development/Isaac-GR00T
bash scripts/run_inspire_e2_ftp_colour_only_isaac_smoke.sh client-run
```

## What this test proves

The existing scene contains a dark metallic cylinder, not a red cup. There is
no cup asset in this simulator checkout. This quick run therefore checks model
loading, RGB capture, FTP-labelled policy metadata, 26D state/action parsing,
normalized Inspire hand mapping, DDS timing, command conditioning, and guarded
motion. It is not a meaningful red-cup success evaluation.

The simulation adapter does not alter real FTP operation. Real FTP continues
to use the vendor split left/right topics and still requires its additional
explicit unverified-stop acknowledgement.

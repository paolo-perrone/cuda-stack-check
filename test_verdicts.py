#!/usr/bin/env python3
"""Runs decide() against machine states recorded in public failure reports.

Each case cites where its state comes from, so you can check the fixture
against the report before trusting the test. Run: python3 test_verdicts.py
"""
from cuda_stack_check import decide, driver_cmd

CASES = []

# Case 1: the RTX 5090 launch-window state, as reported in
# pytorch/pytorch#159207 and #173237: stable wheels compiled up to sm_90,
# Blackwell consumer cards report sm_120, users hit
# "no kernel image is available for execution on the device".
CASES.append(dict(
    name="5090 vs sm_90 wheel (pytorch#159207/#173237)",
    driver="572.16", ceiling="12.8",
    gpus=[{"name": "NVIDIA GeForce RTX 5090", "cc": "12.0"}],
    nvcc=None,
    torch=dict(version="2.6.0+cu124", cuda="12.4", cudnn=90100,
               archs=["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]),
    expect_code=1, expect_contains="sm_120",
    expect_fix="580",
))

# Case 2: driver below the wheel's bundled CUDA. NVIDIA's release notes put
# the driver floor for every CUDA 13.x at the 580 branch; a 570-branch driver
# advertises a 12.8 ceiling, and a cu130 wheel lands above it.
CASES.append(dict(
    name="cu130 wheel over a 12.8-ceiling driver (CUDA 13 floor = 580)",
    driver="570.86", ceiling="12.8",
    gpus=[{"name": "NVIDIA H100 PCIe", "cc": "9.0"}],
    nvcc=None,
    torch=dict(version="2.12.0+cu130", cuda="13.0", cudnn=91000,
               archs=["sm_80", "sm_90", "sm_100", "sm_120"]),
    expect_code=1, expect_contains="upgrade the DRIVER",
    expect_fix="580 for any 13.x",
))

# Case 3: CPU-only torch on a healthy GPU box; the driver is innocent.
CASES.append(dict(
    name="cpu wheel on a working GPU machine",
    driver="580.65", ceiling="13.0",
    gpus=[{"name": "NVIDIA RTX A6000", "cc": "8.6"}],
    nvcc=None,
    torch=dict(version="2.12.0+cpu", cuda=None, cudnn=None, archs=[]),
    expect_code=1, expect_contains="CPU-only build",
    expect_fix="download.pytorch.org/whl/cu130",
))

# Case 4: coherent stack; toolkit drift is a note, never a verdict.
CASES.append(dict(
    name="coherent stack with nvcc drift",
    driver="580.65", ceiling="13.0",
    gpus=[{"name": "NVIDIA H100 PCIe", "cc": "9.0"}],
    nvcc="12.4",
    torch=dict(version="2.12.0+cu130", cuda="13.0", cudnn=91000,
               archs=["sm_80", "sm_90", "sm_100", "sm_120"]),
    expect_code=0, expect_contains="coherent",
    expect_fix=None,
))


# Case 5: the same Blackwell card once the driver is already on 580. The
# ceiling now clears CUDA 13, so nothing needs the driver and the fix is a
# wheel that carries sm_120.
CASES.append(dict(
    name="5090 on a 580 driver (wheel is the only move)",
    driver="580.65", ceiling="13.0",
    gpus=[{"name": "NVIDIA GeForce RTX 5090", "cc": "12.0"}],
    nvcc=None,
    torch=dict(version="2.6.0+cu124", cuda="12.4", cudnn=90100,
               archs=["sm_70", "sm_80", "sm_86", "sm_90"]),
    expect_code=1, expect_contains="sm_120",
    expect_fix="nightly/cu130",
))


# Case 6: a GPU box with no torch in this environment. The tool can read the
# driver and the card and nothing above them, and saying "coherent" there is a
# clean bill of health it never earned.
CASES.append(dict(
    name="no torch installed (must be inconclusive, not green)",
    driver="580.65", ceiling="13.0",
    gpus=[{"name": "NVIDIA H100 PCIe", "cc": "9.0"}],
    nvcc=None, torch=None,
    expect_code=2, expect_contains="not a clean bill of health",
    expect_fix="cuda_stack_check.py",
))

# Case 7: driver answers, no GPU visible. The container-without---gpus case and
# a card off the bus both land here, and both used to report healthy.
CASES.append(dict(
    name="driver up, zero GPUs visible (container without --gpus)",
    driver="580.65", ceiling="13.0",
    gpus=[],
    nvcc=None,
    torch=dict(version="2.12.0+cu130", cuda="13.0", cudnn=91000, archs=["sm_90"]),
    expect_code=2, expect_contains="no GPU is visible",
    expect_fix="--gpus all",
))

# Case 8: a card newer than every compiled kernel, on a wheel that ships PTX.
# The driver JIT-compiles it, so this is a slow first launch and not the
# no-kernel-image failure; calling it broken contradicts how PTX works.
CASES.append(dict(
    name="newer card, wheel ships PTX (JIT, not failure)",
    driver="580.65", ceiling="13.0",
    gpus=[{"name": "NVIDIA B200", "cc": "10.0"}],
    nvcc=None,
    torch=dict(version="2.12.0+cu130", cuda="13.0", cudnn=91000,
               archs=["sm_80", "sm_90", "compute_90"]),
    expect_code=0, expect_contains="coherent",
    expect_fix=None, expect_note="JIT-compiles on first launch",
))


# driver_cmd is host-dependent by design, so it is pinned against fixed
# /etc/os-release text instead of whatever box the suite runs on. A replay test
# that changes answer with the host is not a replay.
DISTROS = [
    ('ID=ubuntu\nID_LIKE=debian\n', False, "ubuntu-drivers install nvidia:580"),
    ('ID="rocky"\nID_LIKE="rhel centos fedora"\n', False, "dnf module install nvidia-driver:580"),
    ('ID=arch\n', False, "pacman -S nvidia"),
    ('ID=ubuntu\n', True, "the driver belongs to the host"),
    ('', False, "nvidia.com/drivers"),
]


def check_distros():
    bad = 0
    for rel, container, want in DISTROS:
        got = driver_cmd(os_release=rel, in_container=container)
        ok = want in got
        print(f"[{'PASS' if ok else 'FAIL'}] driver_cmd: {want[:38]}")
        if not ok:
            print(f"       got: {got}")
            bad += 1
    return bad


def main():
    failures = check_distros()
    for c in CASES:
        verdict, fix, code, notes = decide(c["driver"], c["ceiling"], c["gpus"],
                                           c["nvcc"], c["torch"])
        want = c["expect_fix"]
        fix_ok = (fix is None) if want is None else (fix is not None and want in fix)
        want_note = c.get("expect_note")
        note_ok = True if want_note is None else any(want_note in n for n in notes)
        ok = (code == c["expect_code"]) and (c["expect_contains"] in verdict) \
            and fix_ok and note_ok
        print(f"[{'PASS' if ok else 'FAIL'}] {c['name']}")
        print(f"       VERDICT: {verdict}")
        if fix:
            print(f"       FIX    : {fix}")
        for n in notes:
            print(f"       note   : {n}")
        if not ok:
            failures += 1
    total = len(CASES) + len(DISTROS)
    print(f"{total - failures}/{total} verdict cases pass")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

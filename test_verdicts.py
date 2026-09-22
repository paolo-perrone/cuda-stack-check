#!/usr/bin/env python3
"""Runs decide() against machine states recorded in public failure reports.

Each case cites where its state comes from, so you can check the fixture
against the report before trusting the test. Run: python3 test_verdicts.py
"""
from cuda_stack_check import decide

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
))

# Case 3: CPU-only torch on a healthy GPU box; the driver is innocent.
CASES.append(dict(
    name="cpu wheel on a working GPU machine",
    driver="580.65", ceiling="13.0",
    gpus=[{"name": "NVIDIA RTX A6000", "cc": "8.6"}],
    nvcc=None,
    torch=dict(version="2.12.0+cpu", cuda=None, cudnn=None, archs=[]),
    expect_code=1, expect_contains="CPU-only build",
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
))


def main():
    failures = 0
    for c in CASES:
        verdict, code, notes = decide(c["driver"], c["ceiling"], c["gpus"],
                                      c["nvcc"], c["torch"])
        ok = (code == c["expect_code"]) and (c["expect_contains"] in verdict)
        print(f"[{'PASS' if ok else 'FAIL'}] {c['name']}")
        print(f"       VERDICT: {verdict}")
        for n in notes:
            print(f"       note   : {n}")
        if not ok:
            failures += 1
    print(f"{len(CASES) - failures}/{len(CASES)} verdict cases pass")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

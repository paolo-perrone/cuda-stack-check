#!/usr/bin/env python3
"""cuda_stack_check.py - reads the CUDA stack already on this machine and names
the one layer to change.

Reads: nvidia-smi (driver + the CUDA ceiling it supports + compute capability),
/usr/local/cuda* (toolkits on disk), and the installed torch wheel (bundled CUDA,
compiled arch list, cuDNN). No arguments, no API key, no GPU required: on a box
with no NVIDIA stack it says so and tells you what that means.

Output: one line per layer, then one VERDICT line. Every number it prints comes
from your machine; every rule it applies is stated next to the finding. The
verdict rules live in decide(), a pure function, and test_verdicts.py runs them
against machine states recorded in public failure reports (pytorch/pytorch
#159207, #173237) plus NVIDIA's documented driver floors.
"""
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""


def find_driver():
    if not shutil.which("nvidia-smi"):
        return None, None
    head = sh(["nvidia-smi"])
    drv = re.search(r"Driver Version:\s*([\d.]+)", head)
    ceil = re.search(r"CUDA Version:\s*([\d.]+)", head)
    return (drv.group(1) if drv else None), (ceil.group(1) if ceil else None)


def find_gpus():
    out = sh(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"])
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            gpus.append({"name": parts[0], "cc": parts[1]})
    return gpus


def find_toolkits():
    kits = []
    for d in sorted(Path("/usr/local").glob("cuda-*")):
        kits.append(d.name.replace("cuda-", ""))
    nvcc = shutil.which("nvcc")
    nvcc_v = None
    if nvcc:
        m = re.search(r"release ([\d.]+)", sh([nvcc, "--version"]))
        nvcc_v = m.group(1) if m else None
    return kits, nvcc_v


def find_torch():
    try:
        import torch  # noqa
    except Exception:
        return None
    info = {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "archs": [],
        "cudnn": None,
    }
    try:
        info["archs"] = torch.cuda.get_arch_list()
    except Exception:
        pass
    try:
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    return info


def vnum(s):
    try:
        return tuple(int(x) for x in str(s).split(".")[:2])
    except Exception:
        return None


def sm_num(sm):
    m = re.search(r"(\d+)", sm or "")
    return int(m.group(1)) if m else -1


def decide(driver, ceiling, gpus, nvcc_v, torch_info):
    """Pure verdict logic over an observed machine state.

    Returns (verdict_line, exit_code, notes). The rules, in order:
      1. no driver -> install the driver first
      2. torch is a CPU build -> reinstall from a cuXXX index
      3. wheel's CUDA above the driver's ceiling -> upgrade the DRIVER
      4. GPU sm newer than every compiled arch -> rebuild/replace the WHEEL
      5. nvcc vs wheel mismatch -> note only (matters when compiling extensions)
    """
    notes = []
    if not driver:
        return ("no NVIDIA driver found; install the driver before touching "
                "toolkits or wheels"), 1, notes
    if torch_info and not torch_info["cuda"]:
        return ("your torch is a CPU-only build; reinstall from a cuXXX index, "
                "the driver is not the problem"), 1, notes
    if torch_info and ceiling and vnum(torch_info["cuda"]) and vnum(ceiling) \
            and vnum(torch_info["cuda"]) > vnum(ceiling):
        return (f"driver {driver} caps you at CUDA {ceiling} but your wheel "
                f"bundles {torch_info['cuda']}; upgrade the DRIVER, not the "
                "toolkit"), 1, notes
    if torch_info and gpus and torch_info["archs"]:
        sms = [a for a in torch_info["archs"] if a.startswith("sm_")]
        newest = max(sms, key=sm_num, default="")
        for g in gpus:
            sm = "sm_" + g["cc"].replace(".", "")
            if sm not in torch_info["archs"] and newest and sm_num(sm) > sm_num(newest):
                return (f"your {g['name']} is {sm} but the wheel compiles only "
                        f"up to {newest}; expect 'no kernel image is available', "
                        f"fix with a wheel built for {sm}, not with a driver "
                        "change"), 1, notes
    if nvcc_v and torch_info and torch_info["cuda"] and vnum(nvcc_v) != vnum(torch_info["cuda"]):
        notes.append(f"nvcc {nvcc_v} differs from the wheel's CUDA "
                     f"{torch_info['cuda']}; harmless for running torch, "
                     "matters only when you compile extensions")
    return ("stack is coherent; if something still fails, the bug is above "
            "the CUDA layer"), 0, notes


def main():
    print("cuda_stack_check")
    print(f"platform : {platform.system().lower()} {platform.machine()}")

    if platform.system() == "Darwin":
        t = find_torch()
        print("driver   : none (macOS has no NVIDIA CUDA driver)")
        if t:
            build = f"cu{t['cuda']}" if t["cuda"] else "cpu/mps"
            print(f"torch    : {t['version']} ({build} build)")
            print("VERDICT  : no NVIDIA driver, this machine cannot run CUDA; "
                  "your torch is a cpu/mps build, and any CUDA fix belongs on "
                  "your Linux box")
        else:
            print("torch    : not installed in this environment")
            print("VERDICT  : no NVIDIA driver, this machine cannot run CUDA; "
                  "any CUDA fix belongs on your Linux box")
        return 1

    driver, ceiling = find_driver()
    gpus = find_gpus()
    kits, nvcc_v = find_toolkits()
    t = find_torch()

    print(f"driver   : {driver or 'none found'}" +
          (f" (supports CUDA <= {ceiling})" if ceiling else ""))
    for g in gpus:
        print(f"gpu      : {g['name']} (compute capability {g['cc']} = "
              f"sm_{g['cc'].replace('.', '')})")
    print(f"toolkits : {', '.join(kits) if kits else 'none in /usr/local'}" +
          (f"; nvcc says {nvcc_v}" if nvcc_v else ""))
    if t:
        print(f"torch    : {t['version']} (bundles CUDA "
              f"{t['cuda'] or 'none, cpu build'}; cuDNN {t['cudnn']})")
        if t["archs"]:
            print(f"kernels  : {' '.join(t['archs'])}")
    else:
        print("torch    : not installed in this environment")

    verdict, code, notes = decide(driver, ceiling, gpus, nvcc_v, t)
    for n in notes:
        print(f"note     : {n}")
    print(f"VERDICT  : {verdict}")
    return code


if __name__ == "__main__":
    sys.exit(main())

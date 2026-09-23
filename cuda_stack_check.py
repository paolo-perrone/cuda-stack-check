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


PYPI = "https://download.pytorch.org/whl"


def driver_cmd(branch="580", os_release=None, in_container=None):
    """The install command for the distro this is running on.

    A fix line is only worth printing if it runs where the reader is standing,
    and `ubuntu-drivers` exists on nothing outside the Debian family. Detection
    reads /etc/os-release ID and ID_LIKE; anything unrecognised, and anything in
    a container where the host owns the driver, falls back to the runfile.
    """
    if os_release is None:
        try:
            os_release = Path("/etc/os-release").read_text()
        except Exception:
            os_release = ""
    rel = os_release.lower()
    ids = " ".join(l.split("=", 1)[1].strip().strip('"')
                   for l in rel.splitlines()
                   if l.startswith(("id=", "id_like=")))
    if in_container is None:
        in_container = Path("/.dockerenv").exists()
    if in_container:
        return ("the driver belongs to the host, not this container; install "
                f"the {branch} branch on the host and restart with --gpus all")
    if any(d in ids for d in ("ubuntu", "debian", "pop", "mint")):
        return f"sudo ubuntu-drivers install nvidia:{branch}"
    if any(d in ids for d in ("rhel", "fedora", "centos", "rocky", "alma")):
        return f"sudo dnf module install nvidia-driver:{branch}"
    if any(d in ids for d in ("arch", "manjaro")):
        return "sudo pacman -S nvidia"
    if any(d in ids for d in ("suse", "sles", "opensuse")):
        return f"sudo zypper install nvidia-video-G0{branch[0]}"
    return f"install the {branch} driver branch from nvidia.com/drivers"


def wheel_tag(ceiling):
    """Newest PyTorch CUDA build tag a driver with this ceiling can serve.

    PyTorch 2.12 publishes exactly cu126 (legacy), cu130 (stable) and cu132
    (experimental); cu128 left the build matrix in April 2026, so it is never
    suggested here. None means no current tag fits, which makes the driver the
    thing that has to move.
    """
    v = vnum(ceiling)
    if not v:
        return None
    if v >= (13, 0):
        return "cu130"
    if v >= (12, 6):
        return "cu126"
    return None


# The CUDA version that first shipped kernels for an architecture. Only the
# entries that change an answer are listed: Blackwell (sm_100 datacenter,
# sm_120 consumer) arrived in CUDA 12.8, so a cu126 wheel can never carry them
# however new the nightly is, and suggesting one sends the reader back to the
# same "no kernel image" they started with.
SM_FLOOR = {"sm_100": (12, 8), "sm_120": (12, 8)}


def tag_covers(tag, sm):
    """Whether wheels on that index can contain kernels for this architecture."""
    floor = SM_FLOOR.get(sm)
    if not floor or not tag:
        return bool(tag)
    return vnum(tag.replace("cu", "")[:2] + "." + tag.replace("cu", "")[2:]) >= floor


def install_line(tag, pre=False):
    """The exact pip line for that tag, or None when no tag fits."""
    if not tag:
        return None
    index = f"{PYPI}/nightly/{tag}" if pre else f"{PYPI}/{tag}"
    flags = "--pre --force-reinstall" if pre else "--force-reinstall"
    return f"pip install {flags} torch --index-url {index}"


def decide(driver, ceiling, gpus, nvcc_v, torch_info):
    """Pure verdict logic over an observed machine state.

    Returns (verdict_line, fix_line, exit_code, notes). The verdict names the
    layer that has to change; the fix is the command for this machine, because
    naming a layer still leaves the reader hunting the incantation. The rules,
    in order:
      1. no driver -> install the driver first
      2. torch is a CPU build -> reinstall from a cuXXX index
      3. wheel's CUDA above the driver's ceiling -> upgrade the DRIVER
      4. GPU sm newer than every compiled arch -> rebuild/replace the WHEEL
      5. nvcc vs wheel mismatch -> note only (matters when compiling extensions)
    """
    notes = []
    tag = wheel_tag(ceiling)

    if not driver:
        verdict = ("no NVIDIA driver found; install the driver before touching "
                   "toolkits or wheels")
        fix = (f"{driver_cmd()}; the 580 branch is the floor for every "
               "CUDA 13.x wheel")
        return verdict, fix, 1, notes

    if torch_info is None:
        verdict = ("torch is not installed in this environment, so nothing above "
                   "the driver could be checked; this is not a clean bill of health")
        fix = ("run this inside the environment that fails: "
               "path/to/venv/bin/python cuda_stack_check.py")
        return verdict, fix, 2, notes

    if not gpus:
        verdict = ("the driver answers but no GPU is visible to it, so the card, "
                   "not the CUDA stack, is the first thing to explain")
        fix = ("check nvidia-smi -L on the host; inside a container this is "
               "usually a missing --gpus all or an unset "
               "NVIDIA_VISIBLE_DEVICES")
        return verdict, fix, 2, notes

    if torch_info and not torch_info["cuda"]:
        verdict = ("your torch is a CPU-only build; reinstall from a cuXXX index, "
                   "the driver is not the problem")
        fix = install_line(tag) or ("raise the driver first: its ceiling sits "
                                    "below every CUDA build PyTorch publishes")
        return verdict, fix, 1, notes

    if torch_info and ceiling and vnum(torch_info["cuda"]) and vnum(ceiling) \
            and vnum(torch_info["cuda"]) > vnum(ceiling):
        verdict = (f"driver {driver} caps you at CUDA {ceiling} but your wheel "
                   f"bundles {torch_info['cuda']}; upgrade the DRIVER, not the "
                   "toolkit")
        fix = ("move the driver to the branch NVIDIA lists for CUDA "
               f"{torch_info['cuda']} in its release notes, 580 for any 13.x")
        back = install_line(tag)
        if back:
            fix += f"; to stay on this driver instead, drop the wheel: {back}"
        return verdict, fix, 1, notes

    if torch_info and gpus and torch_info["archs"]:
        sms = [a for a in torch_info["archs"] if a.startswith("sm_")]
        newest = max(sms, key=sm_num, default="")
        # A compute_XX entry is embedded PTX, which the driver JIT-compiles for
        # an architecture the wheel never saw. Reading only the sm_ entries
        # calls a wheel broken when the driver would have carried it, which is
        # the one mechanism that saves a new card on an old wheel.
        ptx = [a for a in torch_info["archs"] if a.startswith("compute_")]
        # Every card gets evaluated. Returning on the first mismatch hides the
        # other half of a mixed box, and a reader who fixes the card named and
        # still crashes stops trusting the tool.
        broken = []
        for g in gpus:
            sm = "sm_" + g["cc"].replace(".", "")
            if sm in torch_info["archs"] or not newest or sm_num(sm) <= sm_num(newest):
                continue
            if ptx:
                notes.append(
                    f"your {g['name']} is {sm}, newer than every compiled kernel "
                    f"in this wheel, but it ships {ptx[0]} PTX, so the driver "
                    "JIT-compiles on first launch: expect a slow first kernel, "
                    "not a failure")
                continue
            broken.append((g, sm))
        if broken:
            names = "; ".join(f"{g['name']} is {sm}" for g, sm in broken)
            worst = max((sm for _, sm in broken), key=sm_num)
            verdict = (f"{names}, newer than the wheel's newest kernel {newest}; "
                       "expect 'no kernel image is available', fix with a wheel "
                       f"built for {worst}, not with a driver change")
            if len(gpus) > len(broken):
                notes.append(f"{len(gpus) - len(broken)} of your {len(gpus)} GPUs "
                             "are covered by this wheel; the verdict is about the "
                             "rest")
            if tag_covers(tag, worst):
                fix = install_line(tag, pre=True)
            else:
                fix = ("no wheel under your driver's ceiling carries "
                       f"{worst} kernels, so the driver moves first: "
                       f"{driver_cmd()}, then {install_line('cu130')}")
            return verdict, fix, 1, notes

    if nvcc_v and torch_info and torch_info["cuda"] and vnum(nvcc_v) != vnum(torch_info["cuda"]):
        notes.append(f"nvcc {nvcc_v} differs from the wheel's CUDA "
                     f"{torch_info['cuda']}; harmless for running torch, "
                     "matters only when you compile extensions")

    return ("stack is coherent; if something still fails, the bug is above "
            "the CUDA layer"), None, 0, notes


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

    verdict, fix, code, notes = decide(driver, ceiling, gpus, nvcc_v, t)
    for n in notes:
        print(f"note     : {n}")
    print(f"VERDICT  : {verdict}")
    if fix:
        print(f"FIX      : {fix}")
    return code


if __name__ == "__main__":
    sys.exit(main())

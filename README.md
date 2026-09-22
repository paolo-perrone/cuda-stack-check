# cuda-stack-check

For engineers whose PyTorch box just said `no kernel image is available` or
`torch.cuda.is_available() == False`, and who want the one layer to change
named before they reinstall anything.

Run it on the machine that misbehaves:

```
python3 cuda_stack_check.py
```

It prints one line per layer (driver and its ceiling, GPU and its sm number,
toolkits on disk, the wheel's bundled CUDA and kernel list), then one VERDICT
line naming the single layer to change. Standard library only, no GPU needed,
exit 0 on a coherent stack and 1 on a finding.

## Check it before you trust it

`python3 test_verdicts.py` replays four recorded machine states, including the
RTX 5090 launch-week state from pytorch/pytorch#159207 and #173237, and must
print `4/4 verdict cases pass`. Known-good output on a Mac with no NVIDIA
hardware:

```
cuda_stack_check
platform : darwin arm64
driver   : none (macOS has no NVIDIA CUDA driver)
torch    : not installed in this environment
VERDICT  : no NVIDIA driver, this machine cannot run CUDA; any CUDA fix belongs on your Linux box
```

## Support

Tested on 2026-09-21 against Python 3.11+; no dependencies, so there is no
lockfile to drift. The verdict rules encode NVIDIA's documented compatibility
model (driver floor 580 for CUDA 13.x wheels; SASS/PTX coverage per compute
capability). Rules get revisited each time PyTorch changes its CUDA build
matrix; open an issue if a verdict disagrees with your machine.

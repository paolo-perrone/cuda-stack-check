# cuda-stack-check

For engineers whose PyTorch box just said `no kernel image is available` or
`torch.cuda.is_available() == False`, and who want the one layer to change
named, and the command to change it, before they reinstall anything.

Run it on the machine that misbehaves:

```
python3 cuda_stack_check.py
```

It prints one line per layer (driver and its ceiling, GPU and its sm number,
toolkits on disk, the wheel's bundled CUDA and kernel list), then a VERDICT
naming the single layer to change, then a FIX line you can paste. Standard
library only, no GPU needed. Three exit codes, because a checker that cannot
say "insufficient data" will eventually call a broken box healthy: 0 coherent,
1 a finding with the layer named, 2 inconclusive because something it needed to
read was not there (no torch in this environment, or no GPU visible to the
driver).

The FIX is computed against your machine, not looked up. On a 5090 behind a
570-branch driver it moves the driver first, because no wheel under a CUDA 12.8
ceiling carries sm_120 kernels; on the same card behind a 580 driver it goes
straight to a nightly cu130 wheel. Same verdict, different command.

## Check it before you trust it

`python3 test_verdicts.py` replays eight recorded machine states, including the
RTX 5090 launch-week state from pytorch/pytorch#159207 and #173237, plus five
pinned `/etc/os-release` fixtures for the driver command, and must print
`13/13 verdict cases pass`. Each case cites the report its numbers come
from, so you can check the fixture before trusting the rule.

Known-good output on a Mac with no NVIDIA hardware:

```
cuda_stack_check
platform : darwin arm64
driver   : none (macOS has no NVIDIA CUDA driver)
torch    : not installed in this environment
VERDICT  : no NVIDIA driver, this machine cannot run CUDA; any CUDA fix belongs on your Linux box
```

## Real hardware

Every output below is from a machine, never hand-written. The replayed states
above come from public failure reports; this section is the tool against live
silicon.

<!-- PASTE-HERE: run `python3 cuda_stack_check.py` on a GPU box and paste the
     whole block, including the FIX line, between the fences below. Delete this
     comment and the placeholder line once real output is in. -->

```
(no live run recorded yet)
```

## What it will not do

It never suggests a toolkit reinstall, because torch does not read the system
toolkit. It never suggests `cu128`: PyTorch removed those builds from its
matrix in April 2026, so a pin to them decays quietly. It never calls a wheel
broken for a card it has no kernels for when the wheel ships `compute_XX` PTX,
because the driver JIT-compiles that forward; you get a slow first launch, and
the tool says so instead of sending you to reinstall.

## Support

Tested on 2026-09-22 against Python 3.11+; no dependencies, so there is no
lockfile to drift. The rules encode NVIDIA's documented compatibility model
(driver floor 580 for CUDA 13.x wheels; SASS/PTX coverage per compute
capability; Blackwell sm_100 and sm_120 first shipping in CUDA 12.8). They get
revisited each time PyTorch changes its CUDA build matrix; open an issue if a
verdict disagrees with your machine.

MIT licensed.

Written for [The AI Engineer](https://theaiengineer.substack.com), alongside
[What is CUDA?](https://theaiengineer.substack.com/p/what-is-cuda).

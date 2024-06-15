# ComfyUI-ODE
Adaptive ODE Solvers for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

This is an ALPHA release using [torchdiffeq](https://github.com/rtqichen/torchdiffeq). The batch is denoised one-at-a-time.

When time permits, a custom batched RK solver will be written and fixed-step methods will be deprecated.

This node only supports ODE flow models like Stable Diffusion 3.

# Installation
Clone this repo into ``ComfyUI/custom_nodes`` or use [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager).

In your ComfyUI virtual environment, run ``pip install torchdiffeq``.

# Usage

The ODE Solver node is available at <ins>sampling > custom_sampling > samplers > ODE Solver</ins>.

| Solver        | log_relative_tolerance | log_absolute_tolerance | quality | speed |
| :------------ | :--------------------: | :--------------------: | :------ | :---- |
| bosh3         | -2.5                   | -3.5                   | ⭐⭐⭐     | ⭐⭐    |
| fehlberg2     | -4.0                   | -6.0                   | ⭐⭐      | ⭐⭐⭐   |
| adaptive_heun | -2.5                   | -3.5                   | ⭐       | ⭐⭐⭐   |
| dopri5        | -2.0                   | -3.0                   | ⭐       | ⭐     |

# License
The license is the same as ComfyUI, GPL 3.0.

Copyright 2024 by @RedHotTensors and released by [Project RedRocket](https://huggingface.co/RedRocket).

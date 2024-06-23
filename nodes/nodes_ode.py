import comfy
import torch
import torchdiffeq
from tqdm.auto import trange, tqdm

ADAPTIVE_SOLVERS = { "dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun" }
FIXED_SOLVERS = { "euler", "midpoint", "rk4", "heun3", "explicit_adams", "implicit_adams" }
SOLVERS = [ *ADAPTIVE_SOLVERS, *FIXED_SOLVERS ]
SOLVERS.sort()

class ODEFunction:
    def __init__(self, model, t_min, t_max, n_steps, is_adaptive, extra_args=None, callback=None):
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.callback = callback
        self.t_min = t_min.item()
        self.t_max = t_max.item()
        self.n_steps = n_steps
        self.is_adaptive = is_adaptive
        self.step = 0

        if is_adaptive:
            self.pbar = tqdm(
                total=100,
                desc="solve",
                unit="%",
                leave=False,
                position=1
            )
        else:
            self.pbar = tqdm(
                total=n_steps,
                desc="solve",
                leave=False,
                position=1
            )

    def __call__(self, t, y):
        if t <= 1e-5:
            return torch.zeros_like(y)

        denoised = self.model(y.unsqueeze(0), t.unsqueeze(0), **self.extra_args)
        return (y - denoised.squeeze(0)) / t

    def _callback(self, t0, y0, step):
        if self.callback is not None:
            y0 = y0.unsqueeze(0)

            self.callback({
                "x": y0,
                "i": step,
                "sigma": t0,
                "sigma_hat": t0,
                "denoised": y0, # for a bad latent preview
            })

    def callback_step(self, t0, y0, dt):
        if self.is_adaptive:
            return

        self._callback(t0, y0, self.step)

        self.pbar.update(1)
        self.step += 1

    def callback_accept_step(self, t0, y0, dt):
        if not self.is_adaptive:
            return

        progress = (self.t_max - t0.item()) / (self.t_max - self.t_min)

        self._callback(t0, y0, round((self.n_steps - 1) * progress))

        new_step = round(100 * progress)
        self.pbar.update(new_step - self.step)
        self.step = new_step

    def reset(self):
        self.step = 0
        self.pbar.reset()

class ODESampler:
    def __init__(self, solver, rtol, atol, max_steps):
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

    @torch.no_grad()
    def __call__(self, model, x: torch.Tensor, sigmas: torch.Tensor, extra_args=None, callback=None, disable=None):
        t_max = sigmas.max()
        t_min = sigmas.min()
        n_steps = len(sigmas)

        if self.solver in FIXED_SOLVERS:
            t = sigmas
            is_adaptive = False
        else:
            t = torch.stack([t_max, t_min])
            is_adaptive = True

        ode = ODEFunction(model, t_min, t_max, n_steps, is_adaptive=is_adaptive, callback=callback, extra_args=extra_args)

        samples = torch.empty_like(x)
        for i in trange(x.shape[0], desc=self.solver, disable=disable):
            ode.reset()

            samples[i] = torchdiffeq.odeint(
                ode,
                x[i],
                t,
                rtol=self.rtol,
                atol=self.atol,
                method=self.solver,
                options={
                    "min_step": 1e-5,
                    "max_num_steps": self.max_steps,
                    "dtype": torch.float32 if torch.backends.mps.is_available() else torch.float64
                }
            )[-1]

        if callback is not None:
            callback({
                "x": samples,
                "i": n_steps - 1,
                "sigma": t_min,
                "sigma_hat": t_min,
                "denoised": samples, # only accurate if t_min = 0, for now
            })

        return samples

class ODESamplerSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solver": (SOLVERS, { "default": "bosh3" }),
                "log_relative_tolerance": ("FLOAT", { "min": -7, "max": 0, "default": -2.5, "step": 0.1 }),
                "log_absolute_tolerance": ("FLOAT", { "min": -7, "max": 0, "default": -3.5, "step": 0.1 }),
                "max_steps": ("INT", { "min": 1, "max": 500, "default": 30, "step": 1 }),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"

    def get_sampler(self, solver, log_relative_tolerance, log_absolute_tolerance, max_steps):
        rtol = 10 ** log_relative_tolerance
        atol = 10 ** log_absolute_tolerance
        assert rtol >= atol

        return (comfy.samplers.KSAMPLER(ODESampler(solver, rtol, atol, max_steps)),)

NODE_CLASS_MAPPINGS = {
    "ODESamplerSelect": ODESamplerSelect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ODESamplerSelect": "ODE Sampler",
}

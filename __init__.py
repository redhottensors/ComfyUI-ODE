"""
@author: RedHotTensors
@title: ComfyUI-ODE
@nickname: ComfyUI-ODE
@description: Adaptive ODE Solvers for ComfyUI
"""

from .nodes import nodes_ode

NODE_CLASS_MAPPINGS = {
    **nodes_ode.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **nodes_ode.NODE_DISPLAY_NAME_MAPPINGS,
}

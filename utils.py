"""
Snippet generators for serializing models (scikit-learn and PyTorch) into a
copy-pasteable get_model() function string. The generated function is self-contained
and can be pasted into another file to reconstruct the model.

Features:
- Strong typing and clear APIs.
- Optional compression for smaller embedded payloads (zlib, gzip, bz2, lzma, none).
- For scikit-learn: uses pickle-compatible bytes (cloudpickle for dumping if available).
- For PyTorch: prefers TorchScript (self-contained), then full-model pickle with
  PyTorch>=2.6 handling, then state_dict fallback.

Security note:
- Both approaches embed pickled data (TorchScript is safe to load; pickles are not).
- Loading pickled data can execute arbitrary code. Only load from trusted sources.

Usage (brief):
    code = generate_sklearn_loader_snippet(fitted_sklearn_model, compression="zlib")
    print(code)  # paste the printed function into your target codebase

    code = generate_torch_loader_snippet(pytorch_model, example_inputs=example, prefer="auto", compression="zlib")
    print(code)  # paste the printed function into your target codebase
"""

from __future__ import annotations

import base64
import io
import inspect
import pickle
from typing import Any, Literal, Optional

# Optional imports only used when generating PyTorch snippets (runtime still needs torch)
try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None     # type: ignore

Compression = Literal["zlib", "gzip", "bz2", "lzma", "none"]
TorchScriptMode = Literal["auto", "script", "trace"]


# =========================
# Shared helpers
# =========================

def _normalize_torchscript_mode(mode: TorchScriptMode) -> TorchScriptMode:
    m = str(mode).lower()
    return m if m in {"auto", "script", "trace"} else "auto"


def _compress_to_b64(data: bytes, compression: Compression, level: int) -> tuple[str, str, str]:
    """
    Compress bytes and return:
      - base64 string of compressed bytes
      - decomp_loader_code: Python code for the generated snippet to decompress
      - comp_name: normalized compression name
    """
    comp = (compression or "zlib").lower()
    if comp not in {"zlib", "gzip", "bz2", "lzma", "none"}:
        comp = "zlib"

    if comp == "zlib":
        import zlib
        raw = zlib.compress(data, level=level)
        decomp_code = "import zlib as _z; _decomp = _z.decompress"
    elif comp == "gzip":
        import gzip
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=level) as f:
            f.write(data)
        raw = buf.getvalue()
        decomp_code = "import gzip as _gz, io as _io; _decomp = lambda b: _gz.GzipFile(fileobj=_io.BytesIO(b), mode='rb').read()"
    elif comp == "bz2":
        import bz2
        lvl = min(max(level, 1), 9)
        raw = bz2.compress(data, compresslevel=lvl)
        decomp_code = "import bz2 as _bz2; _decomp = _bz2.decompress"
    elif comp == "lzma":
        import lzma
        raw = lzma.compress(data, preset=min(max(level, 0), 9))
        decomp_code = "import lzma as _lz; _decomp = _lz.decompress"
    else:  # none
        raw = data
        decomp_code = "_decomp = (lambda b: b)"

    b64 = base64.b64encode(raw).decode("ascii")
    return b64, decomp_code, comp


def _has_noarg_constructor(cls: type) -> bool:
    try:
        sig = inspect.signature(cls)
        params = list(sig.parameters.values())[1:]  # skip self
        return all(
            p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or p.default is not p.empty
            for p in params
        )
    except Exception:
        return False


# =========================
# Scikit-learn generator
# =========================

def generate_sklearn_loader_snippet(
    model: Any,
    compression: Compression = "zlib",
    level: int = 9,
) -> str:
    """
    Create a copy-pasteable get_model() code string that reconstructs the given
    scikit-learn model (estimator/transformer/pipeline), with optional compression.

    Args:
        model: An instantiated scikit-learn object (instance, not a class).
        compression: Compression algorithm ("zlib", "gzip", "bz2", "lzma", "none").
        level: Compression level (zlib/gzip/bz2; lzma uses preset; ignored for none).

    Returns:
        Python source string defining get_model().

    Raises:
        TypeError: If 'model' is a class instead of an instance.

    Security:
        The embedded payload is pickle-compatible. Only load from trusted sources.
    """
    if isinstance(model, type):
        raise TypeError("Expected an instantiated scikit-learn model (instance), not a class.")

    # Serialize with cloudpickle if available, else pickle.
    try:
        import cloudpickle as _cp  # type: ignore
        blob = _cp.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        blob = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

    b64, decomp_code, comp_name = _compress_to_b64(blob, compression, level)

    # Use pickle.loads in the generated snippet. cloudpickle output is readable by pickle.loads.
    return f'''\
def get_model():
    """
    Reconstruct and return a scikit-learn model from an embedded, base64-encoded {'compressed ' if comp_name!='none' else ''}blob.

    Security note:
      This uses pickle-compatible loading. Only use if you trust the source.
    """
    import base64
    import pickle as _p
    {decomp_code}
    _blob_b64 = "{b64}"
    _raw = _decomp(base64.b64decode(_blob_b64))
    model = _p.loads(_raw)
    return model
'''

  

# =========================
# PyTorch generator
# =========================

def generate_torch_loader_snippet(
    model: "nn.Module",
    example_inputs: Optional[Any] = None,
    prefer: TorchScriptMode = "auto",
    compression: Compression = "zlib",
    level: int = 9,
) -> str:
    """
    Create a copy-pasteable get_model() code string that reconstructs the given PyTorch model.

    Strategy (attempted in order):
      1) TorchScript (self-contained: no dependency on original Python class).
         - Prefer scripting; if prefer='trace' or scripting fails and example_inputs given, try trace.
      2) Full model pickle (torch.save(model)): robust loader for PyTorch >= 2.6
         - Tries safe allowlisting of the class via torch.serialization.safe_globals
         - Falls back to weights_only=False (only for trusted sources)
      3) state_dict fallback:
         - Embeds state_dict. Loader imports the class and instantiates it.
         - If zero-arg constructor is not available, a TODO placeholder is emitted.

    Args:
        model: An instantiated torch.nn.Module (instance, not a class).
        example_inputs: Example input(s) for tracing when needed.
        prefer: "auto", "script", or "trace".
        compression: Compression algorithm.
        level: Compression level.

    Returns:
        Python source string defining get_model(device="cpu", dtype=None).

    Raises:
        RuntimeError: If PyTorch is not available in the current environment.
        TypeError: If 'model' is not an instantiated nn.Module.

    Security:
        - TorchScript payload is safe to load.
        - Pickle-based payloads (full model or state_dict) can execute code on load if misused;
          the generated loader attempts safe_globals first but may fallback to trusted loading.
    """
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is not available in this environment.")
    if not isinstance(model, nn.Module):
        raise TypeError("Expected an instantiated torch.nn.Module (instance), not a class.")

    mode = _normalize_torchscript_mode(prefer)

    # 1) TorchScript attempt (script, then trace if allowed and example provided)
    ts_bytes = _dump_torchscript_bytes(model, mode, example_inputs)
    if ts_bytes is not None:
        b64, decomp_code, comp_name = _compress_to_b64(ts_bytes, compression, level)
        return _render_torchscript_loader(b64, decomp_code, comp_name)

    # 2) Full model pickle
    full_bytes = _dump_full_pickle_bytes(model)
    if full_bytes is not None:
        b64, decomp_code, comp_name = _compress_to_b64(full_bytes, compression, level)
        module_name = model.__class__.__module__
        class_name = model.__class__.__name__
        return _render_full_pickle_loader(b64, decomp_code, comp_name, module_name, class_name)

    # 3) state_dict fallback
    sd_bytes = _dump_state_dict_bytes(model)
    b64, decomp_code, comp_name = _compress_to_b64(sd_bytes, compression, level)
    zero_arg_ok = _has_noarg_constructor(model.__class__)
    module_name = model.__class__.__module__
    class_name = model.__class__.__name__
    return _render_state_dict_loader(b64, decomp_code, comp_name, module_name, class_name, zero_arg_ok)



# ----- PyTorch generator internals -----

def _dump_torchscript_bytes(
    model: "nn.Module",
    mode: TorchScriptMode,
    example_inputs: Optional[Any],
) -> Optional[bytes]:
    try:
        model_eval = model.eval()
        with torch.no_grad():
            if mode == "trace":
                if example_inputs is None:
                    return None
                ts = torch.jit.trace(model_eval, example_inputs, strict=False)
            else:
                try:
                    ts = torch.jit.script(model_eval)
                except Exception:
                    if mode == "auto" and example_inputs is not None:
                        ts = torch.jit.trace(model_eval, example_inputs, strict=False)
                    else:
                        return None
        return ts.save_to_buffer()
    except Exception:
        return None


def _dump_full_pickle_bytes(model: "nn.Module") -> Optional[bytes]:
    try:
        buf = io.BytesIO()
        torch.save(model, buf)
        return buf.getvalue()
    except Exception:
        return None


def _dump_state_dict_bytes(model: "nn.Module") -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def _render_torchscript_loader(b64: str, decomp_code: str, comp_name: str) -> str:
    return f'''\
def get_model(device: str = "cpu", dtype: str | None = None):
    """
    Return a TorchScript model loaded from an embedded, base64-encoded {'compressed ' if comp_name!='none' else ''}blob.
    Self-contained: no need for the original Python class.

    Args:
        device: Where to map the model (e.g., "cpu", "cuda", "cuda:0").
        dtype: Optional dtype to convert parameters/buffers to (e.g., "float32", "float16").
    """
    import base64, io, torch
    {decomp_code}
    _blob_b64 = "{b64}"
    _raw = _decomp(base64.b64decode(_blob_b64))
    buf = io.BytesIO(_raw)
    m = torch.jit.load(buf, map_location=device)
    if dtype is not None:
        dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        for p in m.parameters():
            p.data = p.data.to(dt)
        for b in m.buffers():
            b.data = b.data.to(dt)
    m.eval()
    return m
'''


def _render_full_pickle_loader(
    b64: str,
    decomp_code: str,
    comp_name: str,
    module_name: str,
    class_name: str,
) -> str:
    return f'''\
def get_model(device: str = "cpu", dtype: str | None = None):
    """
    Return the original PyTorch model loaded from an embedded, base64-encoded {'compressed ' if comp_name!='none' else ''}pickle.

    Notes:
      - The original model class should be importable (module: "{module_name}", class: "{class_name}").
      - PyTorch >= 2.6: torch.load defaults to weights_only=True.
        This loader will:
          1) Try to import the class and allowlist it via torch.serialization.safe_globals.
          2) Fall back to weights_only=False (ONLY if you trust this source).

    Args:
        device: Where to map the model (e.g., "cpu", "cuda:0").
        dtype: Optional dtype (string like "float32" or torch.dtype).
    """
    import base64, io, importlib, torch
    {decomp_code}
    _blob_b64 = "{b64}"
    _raw = _decomp(base64.b64decode(_blob_b64))

    # Try to import the class for safe allowlisting
    try:
        mod = importlib.import_module("{module_name}")
        cls = getattr(mod, "{class_name}", None)
    except Exception:
        cls = None

    # Attempt safe load first
    try:
        if cls is not None:
            with torch.serialization.safe_globals([cls]):
                m = torch.load(io.BytesIO(_raw), map_location=device)
        else:
            # Class not importable; last resort: trusted load
            m = torch.load(io.BytesIO(_raw), map_location=device, weights_only=False)
    except Exception:
        # Final fallback: trusted load
        m = torch.load(io.BytesIO(_raw), map_location=device, weights_only=False)

    if dtype is not None:
        dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        m = m.to(dtype=dt)
    m.eval()
    return m
'''


def _render_state_dict_loader(
    b64: str,
    decomp_code: str,
    comp_name: str,
    module_name: str,
    class_name: str,
    zero_arg_ok: bool,
) -> str:
    ctor = f"{class_name}()" if zero_arg_ok else f"{class_name}(# TODO: fill constructor args)"
    return f'''\
def get_model(device: str = "cpu", dtype: str | None = None):
    """
    Return a PyTorch model by instantiating the class and loading an embedded state_dict
    from a base64-encoded {'compressed ' if comp_name!='none' else ''}blob.

    Requirements:
      - The model class must be importable (module: "{module_name}", class: "{class_name}").
      - If the constructor needs arguments, fill them in where indicated.

    Args:
        device: Where to map the tensors (e.g., "cpu", "cuda:0").
        dtype: Optional dtype (string or torch.dtype).
    """
    import base64, io, importlib, torch
    {decomp_code}
    mod = importlib.import_module("{module_name}")
    cls = getattr(mod, "{class_name}")
    model = {ctor}
    sd = torch.load(io.BytesIO(_decomp(base64.b64decode("{b64}"))), map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("Warning: load_state_dict mismatches. Missing:", missing, "Unexpected:", unexpected)
    if dtype is not None:
        dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        model = model.to(dtype=dt)
    model.to(device)
    model.eval()
    return model
'''


# ---


from typing import List, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image

ImageInput = Union[str, np.ndarray, Image.Image]

@dataclass
class MatplotlibImageBrowser:
    fig: plt.Figure
    ax_img: plt.Axes
    slider: Slider
    btn_prev: Button
    btn_next: Button

def show_images_interactive(
    images: List[ImageInput],
    titles: Optional[List[str]] = None,
    cmap: Optional[str] = None,
    figsize: Tuple[float, float] = (6.0, 5.0),
    hide_toolbar: bool = True,
) -> MatplotlibImageBrowser:
    """
    Interactively browse images using Matplotlib Slider/Buttons (no ipywidgets).
    Use an interactive backend in the notebook:
      %matplotlib widget    (JupyterLab/VS Code with ipympl)
      %matplotlib notebook  (classic notebook)

    Parameters:
    - images: list of images (paths, NumPy arrays, or PIL.Image.Image)
    - titles: optional list of per-image titles
    - cmap: optional matplotlib colormap (e.g., 'gray')
    - figsize: figure size
    - hide_toolbar: hides the Matplotlib navigation toolbar if supported

    Returns:
    - MatplotlibImageBrowser with references to keep controls alive
    """
    def to_pil(x: ImageInput) -> Image.Image:
        if isinstance(x, Image.Image):
            return x
        if isinstance(x, np.ndarray):
            arr = x
            if arr.dtype != np.uint8:
                arr = np.nan_to_num(arr, copy=True)
                arr_min = float(arr.min())
                arr_max = float(arr.max())
                if arr_max > arr_min:
                    arr = (arr - arr_min) / (arr_max - arr_min)
                else:
                    arr = np.zeros_like(arr, dtype=np.float32)
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            if arr.ndim == 2:
                return Image.fromarray(arr, mode="L")
            if arr.ndim == 3:
                if arr.shape[2] == 1:
                    return Image.fromarray(arr[..., 0], mode="L")
                if arr.shape[2] == 3:
                    return Image.fromarray(arr)
                if arr.shape[2] == 4:
                    return Image.fromarray(arr, mode="RGBA")
            raise ValueError(f"Unsupported ndarray shape: {arr.shape}")
        if isinstance(x, str):
            return Image.open(x)
        raise TypeError(f"Unsupported image type: {type(x)}")

    pil_images: List[Image.Image] = [to_pil(img) for img in images]
    n = len(pil_images)
    if n == 0:
        raise ValueError("No images provided.")
    if titles is None:
        titles = [f"Image {i}" for i in range(n)]
    elif len(titles) != n:
        raise ValueError("titles must have the same length as images.")

    # Figure
    fig = plt.figure(figsize=figsize, constrained_layout=False)

    # Try to hide toolbar if requested and supported (ipympl exposes this)
    if hide_toolbar:
        try:
            # ipympl backend
            if hasattr(fig.canvas, "toolbar_visible"):
                fig.canvas.toolbar_visible = False  # type: ignore[attr-defined]
        except Exception:
            pass

    img_rect = [0.08, 0.18, 0.84, 0.75]
    slider_rect = [0.1, 0.08, 0.55, 0.05]
    prev_rect = [0.70, 0.08, 0.10, 0.05]
    next_rect = [0.82, 0.08, 0.10, 0.05]

    ax_img = fig.add_axes(img_rect)
    ax_slider = fig.add_axes(slider_rect)
    ax_prev = fig.add_axes(prev_rect)
    ax_next = fig.add_axes(next_rect)

    current_idx = 0
    arr0 = np.asarray(pil_images[current_idx])
    im_artist = ax_img.imshow(arr0, cmap=cmap)
    ax_img.set_title(titles[current_idx], fontdict={"fontsize": 12})
    ax_img.axis("off")

    # Make the slider and buttons compact
    slider = Slider(ax=ax_slider, label="Index", valmin=0, valmax=n - 1, valinit=current_idx, valstep=1)
    btn_prev = Button(ax_prev, label="Prev")
    btn_next = Button(ax_next, label="Next")

    def update_index(new_idx: int) -> None:
        nonlocal current_idx
        current_idx = int(np.clip(new_idx, 0, n - 1))
        img = pil_images[current_idx]
        im_artist.set_data(np.asarray(img))
        ax_img.set_title(titles[current_idx])
        ax_img.set_xlim(0, img.size[0])
        ax_img.set_ylim(img.size[1], 0)
        fig.canvas.draw_idle()

    def on_slider_change(val: float) -> None:
        update_index(int(val))

    def on_prev_clicked(event) -> None:
        slider.set_val(max(0, int(slider.val) - 1))

    def on_next_clicked(event) -> None:
        slider.set_val(min(n - 1, int(slider.val) + 1))

    slider.on_changed(on_slider_change)
    btn_prev.on_clicked(on_prev_clicked)
    btn_next.on_clicked(on_next_clicked)

    browser = MatplotlibImageBrowser(fig=fig, ax_img=ax_img, slider=slider, btn_prev=btn_prev, btn_next=btn_next)
    setattr(fig, "_image_browser", browser)  # keep references alive

    return browser


# ---


from typing import Callable
from grid_adventure.grid import GridState
from grid_adventure.step import Action

from grid_adventure.step import step
from grid_adventure.grid import to_state


def get_level_name(builder: Callable[[], GridState]) -> str:
    return f"Level {' '.join(builder.__name__.split('_')[2:])}"



def get_minimum_total_reward(builder: Callable[[], GridState]) -> int:
    state = to_state(builder())
    assert state.turn_limit is not None
    while not (state.win or state.lose):
        state = step(state, Action.WAIT)
    return state.score


from grid_adventure.env import GridAdventureEnv
from grid_adventure.grid import GridState, from_state

import time
from dataclasses import replace
from grid_adventure.rendering import IMAGE_MAP
from grid_adventure.rendering import DEFAULT_ASSET_ROOT


def evaluate(agent_class: type[Agent], env: GridAdventureEnv) -> tuple[float, bool, bool, list[tuple[GridState,Action]]]:
    """Utility to execute a full episode using the provided agent class.

    Parameters
    ----------
    agent_class : type[Agent]
        Class (not instance). The function will instantiate it via agent_class().
    env : GridAdventureEnv
        An initialized GridAdventureEnv environment. The caller is responsible
        for any seeding / configuration before passing it here.

    Returns
    -------
    total_reward : float
        Cumulative reward collected until termination.
    win : bool
        True if the agent reached a win condition.
    lose : bool
        True if the agent triggered a loss condition.

    Notes
    -----
    - This helper mirrors the Coursemology evaluation loop structure.
    """
    state, _ = env.reset()
    agent = agent_class()
    win = False
    lose = False
    total_reward = 0
    history: list[tuple[GridState,Action]] = []
    while not (win or lose):
        action = agent.step(state)

        # record state+action pairs for analysis / debugging
        assert env.state is not None
        history.append((from_state(env.state), action))

        state, reward, win, lose, _ = env.step(action)
        total_reward += reward

    assert env.state is not None
    history.append((from_state(env.state), Action.WAIT))  # final state
    
    return total_reward, win, lose, history


def create_env(builder: Callable[[], GridState], seed: int = 42, turn_limit: int | None = None, **kwargs) -> GridAdventureEnv:
    sample_state = to_state(builder())
    def _initial_state_fn(*args, **kwargs) -> GridState:
        state = to_state(builder())
        if turn_limit is not None:
            state = replace(state, turn_limit=turn_limit)
        return replace(state, seed=seed)
    return GridAdventureEnv(initial_state_fn=_initial_state_fn, width=sample_state.width, height=sample_state.height, render_image_map=IMAGE_MAP, render_asset_root=DEFAULT_ASSET_ROOT, **kwargs)


def get_performance(total_reward: float, optimal_total_reward: float, minimum_total_reward: float, win: bool) -> float:
    if not win:
        return 0.0
    return max(0, total_reward - minimum_total_reward) / (optimal_total_reward - minimum_total_reward)


def evaluate_level(
        agent_class, 
        builder: Callable[[], GridState], 
        max_total_reward: int,
        min_total_reward: int,
        observation_type: str = 'gridstate', 
        time_limit: int | None = None,
        turn_limit: int | None = None,
        seed: int = 42,
        **kwargs,
    ) -> dict[str, float | str | bool]:
    level_name = get_level_name(builder)
    env = create_env(builder, observation_type=observation_type, seed=seed, turn_limit=turn_limit, **kwargs)
    
    total_reward = 0.0
    win = False
    lose = False
    error = False
    
    runtime = time_limit
    try:
        start_time = time.time()
        total_reward, win, lose, _ = evaluate(agent_class, env)
        runtime = time.time() - start_time
    except Exception as e:
        print(f"Error during evaluation of {level_name}: {e}")
        error = True

    timeout = runtime >= (time_limit if time_limit is not None else float('inf'))
    
    performance = get_performance(total_reward, max_total_reward, min_total_reward, win)
    return {
        'level_name': f"{level_name} ({observation_type})",
        'performance': performance,
        'total_reward': total_reward,
        'win': win,
        'lose': lose,
        'timeout': timeout,
        'error': error,
        'runtime (sec)': round(runtime, 2),
    }


def get_result_string(result: dict[str, float | str | bool]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in result.items())

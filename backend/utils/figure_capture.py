'''Rescue matplotlib figures a run leaves unsaved.

The server is headless, so `plt.show()` is a no-op: a figure the data agent draws
but never saves is lost silently — and for a visualisation task the figure *is*
the deliverable. Explicit `savefig` calls are tracked so they are never
duplicated, figures are left open so they stay editable across calls, and capture
is attributed per session so one conversation never picks up another's figures out
of pyplot's process-global registry.
'''

from __future__ import annotations

import sys
from typing import Callable, Dict, List, Optional, Set, Tuple

from app.config import logger

# Figures the caller saved explicitly, by id(). Never auto-captured.
_explicitly_saved: Set[int] = set()
# fignum -> owning session key.
_figure_owner: Dict[int, str] = {}
# fignum -> path already auto-saved, so a re-capture overwrites in place rather
# than accumulating one file per execution.
_figure_path: Dict[int, str] = {}
_patched = False


def _pyplot():
    '''pyplot, but only if the sandbox has already imported it.

    Returns:
    ----------
    pyplot (module): the already-imported pyplot, or None — importing it here would defeat the point.
    '''

    return sys.modules.get("matplotlib.pyplot")


def install() -> None:
    '''Force a headless backend and start tracking explicit savefig calls.

    Imports matplotlib eagerly so tracking is in place from the very first
    execution; without that, a figure saved during the first call would look
    unsaved and be duplicated.
    '''

    global _patched
    if _patched:
        return
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        from matplotlib.figure import Figure

        original_savefig = Figure.savefig

        def tracking_savefig(self, *args, **kwargs):
            _explicitly_saved.add(id(self))
            return original_savefig(self, *args, **kwargs)

        Figure.savefig = tracking_savefig
        # Figures are deliberately left open so they stay editable across calls;
        # suppress the "too many open figures" warning that would cause.
        matplotlib.rcParams["figure.max_open_warning"] = 0
        _patched = True
    except Exception as exc:  # pragma: no cover - matplotlib optional at import
        logger.debug("Figure capture not installed: %s", exc)


def snapshot_open_figures() -> Tuple[int, ...]:
    '''Figure numbers open before an execution starts.

    Returns:
    ----------
    numbers (tuple): the figure numbers already open, so only figures this call created are considered new.
    '''

    plt = _pyplot()
    if plt is None:
        return ()
    try:
        return tuple(plt.get_fignums())
    except Exception:  # pragma: no cover - defensive
        return ()


def capture_unsaved_figures(
    *,
    session_key: str,
    before: Tuple[int, ...],
    prepare_output_path: Callable[[str], str],
) -> List[str]:
    '''Save figures this session left unsaved; return the paths written.

    Parameters:
    ---------
    session_key (str): the interpreter session that produced the figures.
    before (tuple): the figure numbers open before the call, from `snapshot_open_figures`.
    prepare_output_path (Callable[[str], str]): callable that returns a writable path inside the conversation's scope.

    Returns:
    ----------
    paths (list): the files written. The server is headless, so `plt.show()` is a no-op and an unsaved figure — often the deliverable — would otherwise be lost.
    '''

    plt = _pyplot()
    if plt is None:
        return []
    try:
        current = list(plt.get_fignums())
    except Exception:  # pragma: no cover - defensive
        return []

    before_set = set(before)
    for number in current:
        if number not in before_set:
            _figure_owner[number] = session_key

    saved: List[str] = []
    for number in current:
        if _figure_owner.get(number) != session_key:
            continue
        try:
            figure = plt.figure(number)
        except Exception:  # pragma: no cover - defensive
            continue
        if id(figure) in _explicitly_saved:
            continue
        if not figure.get_axes():
            continue  # nothing drawn yet

        path: Optional[str] = _figure_path.get(number)
        # Already captured and untouched since: rendering it again would rewrite
        # an identical file and repeat the note on every later call. matplotlib
        # sets `stale` back to True as soon as the figure is edited.
        if path is not None and not getattr(figure, "stale", True):
            continue
        try:
            if path is None:
                path = prepare_output_path(f"figure_{len(_figure_path) + 1}.png")
                _figure_path[number] = path
            type(figure).savefig(figure, path, dpi=200, bbox_inches="tight")
            # An auto-save must not count as an explicit one, or the figure would
            # become permanently ineligible for re-capture after an edit.
            _explicitly_saved.discard(id(figure))
            # savefig leaves `stale` set, so mark the captured state ourselves.
            figure.stale = False
            saved.append(path)
        except Exception as exc:
            logger.debug("Could not auto-save figure %s: %s", number, exc)
    return saved


def forget_session(session_key: str) -> None:
    '''Drop bookkeeping for a session whose interpreter was reset or evicted.

    Parameters:
    ---------
    session_key (str): the session whose interpreter was reset or evicted.
    '''

    for number in [num for num, owner in _figure_owner.items() if owner == session_key]:
        _figure_owner.pop(number, None)
        _figure_path.pop(number, None)

"""Console entry points for QEM."""
from __future__ import annotations

import sys
from pathlib import Path


def launch_app() -> int:
    """Launch the QEM Streamlit app (``qem-app`` console script).

    Streamlit must be installed (``pip install qem[gui]``). Extra args are
    forwarded to ``streamlit run`` so users can pass e.g. ``--server.port``.
    """
    try:
        from streamlit.web import cli as stcli
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is required for the QEM GUI. "
            "Install it with `pip install qem[gui]`."
        ) from exc

    app_path = Path(__file__).resolve().parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(launch_app())

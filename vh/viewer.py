"""Shared setup of the viser viewer used by all demos."""

from __future__ import annotations

import viser
from viser.theme import TitlebarButton, TitlebarConfig


def configure(server: viser.ViserServer, control_width: str = "large") -> None:
    """Panel width, and a title bar with links to the lab, the lecture and the code."""
    server.gui.configure_theme(
        control_width=control_width,
        titlebar_content=TitlebarConfig(
            buttons=(
                TitlebarButton(text="Real Virtual Humans Lab · Prof. Gerard Pons-Moll", icon=None,
                               href="https://virtualhumans.mpi-inf.mpg.de/"),
                TitlebarButton(text="Virtual Humans lecture", icon="Description", href="https://virtualhumans.mpi-inf.mpg.de/VH26/"),
                TitlebarButton(text="Code", icon="GitHub", href="https://github.com/gerardponsmoll/vh-toolbox"),
            ),
            image=None,
        ),
    )

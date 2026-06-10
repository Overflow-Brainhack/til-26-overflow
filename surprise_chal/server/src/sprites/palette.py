from pathlib import Path
from typing import NamedTuple

import numpy as np
from coloraide import Color as Base
from coloraide.spaces.okhsl import Okhsl


class Color(Base):
    def is_dark(self) -> bool:
        return self.luminance() < 0.5


Color.register(Okhsl())


class Palette(NamedTuple):
    """An individual color palette."""

    base: Color
    dark: Color
    light: Color

    def get_rgba(self):
        """Returns the palette colors as a dictionary of RGBA tuples."""
        return {
            "light": (np.array((*self.light.convert("srgb").coords(), 1)) * 255).astype(
                np.uint8
            ),
            "base": (np.array((*self.base.convert("srgb").coords(), 1)) * 255).astype(
                np.uint8
            ),
            "dark": (np.array((*self.dark.convert("srgb").coords(), 1)) * 255).astype(
                np.uint8
            ),
        }

    @staticmethod
    def from_base_color(base_color: Color) -> "Palette":
        """Generates a Palette from a single base color by creating lighter and darker variants."""
        hue, saturation, lightness = base_color.convert("okhsl").coords()
        return Palette(
            base=base_color,
            dark=Color("okhsl", [hue, saturation, max(0, lightness - 0.15)], 1),
            light=Color(
                "okhsl",
                [
                    hue,
                    saturation,
                    min(1, lightness + (0.15 if base_color.is_dark() else 0.05)),
                ],
                1,
            ),
        )


hue_sat_color = {
    "red": (30, 1.0, False),
    "green": (145, 1.0, True),
    "blue": (250, 1.0, True),
    "purple": (300, 1.0, True),
    "grey": (0, 0.0, False),
}

base_colors: dict[str, Color] = {}

for name, (hue, saturation, generate_light) in hue_sat_color.items():
    base_colors["dark " + name] = Color("okhsl", [hue, saturation, 0.35], 1)
    base_colors[name] = Color("okhsl", [hue, saturation, 0.55], 1)
    if generate_light:
        base_colors["light " + name] = Color("okhsl", [hue, saturation, 0.75], 1)

# a few manually set colors that didn't work well programmatically otherwise
base_colors["brown"] = Color("okhsl", [50, 1.0, 0.3], 1)
base_colors["orange"] = Color("okhsl", [70, 1.0, 0.65], 1)
base_colors["pink"] = Color("okhsl", [0, 1.0, 0.75], 1)
base_colors["magenta"] = Color("okhsl", [0, 1.0, 0.55], 1)
base_colors["yellow"] = Color("okhsl", [104, 1.0, 0.90], 1)
base_colors["mint"] = Color("okhsl", [155, 0.8, 0.75], 1)
base_colors["white"] = Color("okhsl", [200, 0.1, 0.9], 1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def plot_base_colors(save_path: Path | None = None):
        """Plots a list of discrete ColorAide color objects."""
        num_colors = len(base_colors)
        _, ax = plt.subplots(figsize=(num_colors * 1.5, 2))

        for i, (name, color) in enumerate(base_colors.items()):
            hex_color = color.convert("srgb").to_string(hex=True)
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=hex_color))
            ax.text(
                i + 0.5,
                0.5,
                f"{name}\n({i})",
                color="white" if color.is_dark() else "black",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_xlim(0, num_colors)
        ax.set_ylim(0, 1)
        ax.axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.resolve().as_posix(), dpi=300)
        plt.show()

    plot_base_colors(save_path=Path("assets/base_colors.png"))

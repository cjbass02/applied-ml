"""Tkinter UI for collecting handwriting samples in an EMNIST-compatible format.

Run from the workspace root:

    python -m final.handwriting_app.collect --split train --per-class 10
    python -m final.handwriting_app.collect --split test  --per-class 5

The app prompts you to draw each character (digits, then uppercase, then
lowercase). When you click ``Save`` (or press Enter), the drawing is normalized,
reoriented to EMNIST raw-byte orientation, and written to
``data/personal/<split>/<class>/<timestamp>_<class>.png``.

Sessions can be safely resumed: existing files are counted and the per-class
target picks up where you left off.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from typing import List, Optional, Tuple

from .classes import EMNIST_BYCLASS, existing_count
from .image_io import CANVAS_SIZE, DEFAULT_PEN_RADIUS, save_drawing


class HandwritingApp:
    """Tkinter app that walks the user through collecting EMNIST-aligned samples."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        data_root: Path,
        split: str,
        per_class: int,
        only_classes: Optional[List[str]] = None,
    ) -> None:
        self.root = root
        self.data_root = data_root
        self.split = split
        self.per_class = per_class
        self.classes: List[str] = list(only_classes) if only_classes else list(EMNIST_BYCLASS)
        self.class_idx = 0

        self.strokes: List[List[Tuple[float, float]]] = []
        self._current_stroke: Optional[List[Tuple[float, float]]] = None

        self._build_layout()
        self._refresh_for_current_class()

    def _build_layout(self) -> None:
        self.root.title("Handwriting Collector")
        self.root.resizable(False, False)

        big_font = tkfont.Font(family="Helvetica", size=36, weight="bold")
        med_font = tkfont.Font(family="Helvetica", size=14)
        small_font = tkfont.Font(family="Helvetica", size=10)

        top = tk.Frame(self.root, padx=12, pady=8)
        top.grid(row=0, column=0, sticky="we")

        self.prompt_var = tk.StringVar(value="")
        self.progress_var = tk.StringVar(value="")
        tk.Label(top, text="Please write:", font=med_font).grid(row=0, column=0, sticky="w")
        tk.Label(top, textvariable=self.prompt_var, font=big_font, fg="#0a4").grid(row=0, column=1, padx=8, sticky="w")
        tk.Label(top, textvariable=self.progress_var, font=med_font).grid(row=0, column=2, padx=16, sticky="w")

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            highlightthickness=2,
            highlightbackground="#888",
            cursor="pencil",
        )
        self.canvas.grid(row=1, column=0, padx=12, pady=4)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        btns = tk.Frame(self.root, padx=12, pady=8)
        btns.grid(row=2, column=0, sticky="we")

        tk.Button(btns, text="Save & Next  (Enter)", command=self._save_current, width=20).grid(row=0, column=0, padx=4)
        tk.Button(btns, text="Clear  (C)", command=self._clear_canvas, width=12).grid(row=0, column=1, padx=4)
        tk.Button(btns, text="Undo Stroke  (Z)", command=self._undo_stroke, width=16).grid(row=0, column=2, padx=4)
        tk.Button(btns, text="Skip  (S)", command=self._skip_class, width=10).grid(row=0, column=3, padx=4)
        tk.Button(btns, text="Prev Class", command=self._prev_class, width=10).grid(row=0, column=4, padx=4)

        self.status_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.status_var, font=small_font, anchor="w").grid(
            row=3, column=0, sticky="we", padx=12, pady=(0, 8)
        )

        self.root.bind("<Return>", lambda _e: self._save_current())
        self.root.bind("<KP_Enter>", lambda _e: self._save_current())
        self.root.bind("c", lambda _e: self._clear_canvas())
        self.root.bind("C", lambda _e: self._clear_canvas())
        self.root.bind("z", lambda _e: self._undo_stroke())
        self.root.bind("Z", lambda _e: self._undo_stroke())
        self.root.bind("s", lambda _e: self._skip_class())
        self.root.bind("S", lambda _e: self._skip_class())

    @property
    def current_class(self) -> str:
        return self.classes[self.class_idx]

    def _class_dir(self, class_name: Optional[str] = None) -> Path:
        return self.data_root / self.split / (class_name or self.current_class)

    def _saved_for_current(self) -> int:
        return existing_count(self.data_root, self.split, self.current_class)

    def _refresh_for_current_class(self) -> None:
        cls = self.current_class
        saved = self._saved_for_current()
        self.prompt_var.set(cls)
        self.progress_var.set(f"Sample {min(saved + 1, self.per_class)} of {self.per_class} for class \"{cls}\"")
        self._clear_canvas(refresh_status=False)
        self._set_status(f"Save folder: {self._class_dir()} (so far: {saved})")

        if saved >= self.per_class:
            self._advance_to_next_unfilled()

    def _advance_to_next_unfilled(self) -> None:
        for offset in range(1, len(self.classes) + 1):
            idx = (self.class_idx + offset) % len(self.classes)
            cls = self.classes[idx]
            if existing_count(self.data_root, self.split, cls) < self.per_class:
                self.class_idx = idx
                self._refresh_for_current_class()
                return
        self._set_status("All classes have hit the per-class target. You can keep adding more anyway.")

    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)

    def _on_press(self, event) -> None:
        self._current_stroke = [(event.x, event.y)]
        self.strokes.append(self._current_stroke)
        self._draw_dot(event.x, event.y)

    def _on_drag(self, event) -> None:
        if self._current_stroke is None:
            return
        x0, y0 = self._current_stroke[-1]
        self._current_stroke.append((event.x, event.y))
        self.canvas.create_line(
            x0, y0, event.x, event.y,
            fill="white",
            width=DEFAULT_PEN_RADIUS * 2,
            capstyle=tk.ROUND,
            smooth=True,
        )
        self._draw_dot(event.x, event.y)

    def _on_release(self, _event) -> None:
        self._current_stroke = None

    def _draw_dot(self, x: float, y: float) -> None:
        r = DEFAULT_PEN_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")

    def _clear_canvas(self, refresh_status: bool = True) -> None:
        self.canvas.delete("all")
        self.strokes = []
        self._current_stroke = None
        if refresh_status:
            self._set_status(f"Cleared. Save folder: {self._class_dir()}")

    def _undo_stroke(self) -> None:
        if not self.strokes:
            return
        self.strokes.pop()
        self.canvas.delete("all")
        for stroke in self.strokes:
            if not stroke:
                continue
            for (x0, y0), (x1, y1) in zip(stroke[:-1], stroke[1:]):
                self.canvas.create_line(
                    x0, y0, x1, y1,
                    fill="white",
                    width=DEFAULT_PEN_RADIUS * 2,
                    capstyle=tk.ROUND,
                    smooth=True,
                )
            for (x, y) in stroke:
                self._draw_dot(x, y)

    def _save_current(self) -> None:
        if not any(self.strokes):
            self._set_status("Canvas is empty — draw something before saving.")
            return
        path = save_drawing(
            self.strokes,
            self._class_dir(),
            class_name=self.current_class,
        )
        saved = self._saved_for_current()
        self._set_status(f"Saved {path.name} ({saved}/{self.per_class}).")
        if saved >= self.per_class:
            self._advance_to_next_unfilled()
        else:
            self._refresh_for_current_class()

    def _skip_class(self) -> None:
        self.class_idx = (self.class_idx + 1) % len(self.classes)
        self._refresh_for_current_class()

    def _prev_class(self) -> None:
        self.class_idx = (self.class_idx - 1) % len(self.classes)
        self._refresh_for_current_class()


def _resolve_data_root(cli_value: Optional[str]) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    return Path.cwd().resolve() / "data" / "personal"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Collect handwriting samples for transfer learning.")
    parser.add_argument("--split", choices=("train", "test"), default="train", help="Which split to write into.")
    parser.add_argument("--per-class", type=int, default=10, help="Target number of samples per class.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override the data root. Default: <cwd>/data/personal",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset of class names to focus on (e.g. 'A,B,C').",
    )
    args = parser.parse_args(argv)

    only_classes: Optional[List[str]] = None
    if args.only:
        only_classes = [c.strip() for c in args.only.split(",") if c.strip()]
        for c in only_classes:
            if c not in EMNIST_BYCLASS:
                parser.error(f"Class {c!r} is not in EMNIST byclass.")

    data_root = _resolve_data_root(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    HandwritingApp(
        root,
        data_root=data_root,
        split=args.split,
        per_class=args.per_class,
        only_classes=only_classes,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

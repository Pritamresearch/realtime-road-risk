# Copyright (C) 2026 Pritam Chakraborty <pritam.23.research@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tk window for the dashboard (the installed opencv-python-headless has no imshow). Scales the frame to the window size."""
from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image, ImageTk


class Viewer:
    def __init__(self, title: str, size: tuple[int, int] = (1280, 720)):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{size[0]}x{size[1]}")
        self.root.configure(bg="#090c10")
        self.label = tk.Label(self.root, bg="#090c10", bd=0)
        self.label.pack(fill=tk.BOTH, expand=True)
        self.keys: list[str] = []
        self.closed = False
        self.root.bind("<Key>", lambda e: self.keys.append(e.keysym))
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._photo = None

    def _close(self) -> None:
        self.closed = True

    def show(self, rgb: np.ndarray) -> list[str]:
        """Display one RGB frame; returns the keys pressed since the last call (Tk keysyms, e.g. 'q', 'space', 'Escape')."""
        w, h = max(self.label.winfo_width(), 2), max(self.label.winfo_height(), 2)
        s = min(w / rgb.shape[1], h / rgb.shape[0])
        img = Image.fromarray(rgb)
        if abs(s - 1.0) > 0.01:
            img = img.resize((max(1, int(rgb.shape[1] * s)), max(1, int(rgb.shape[0] * s))), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(img)
        self.label.configure(image=self._photo)
        return self.poll()

    def poll(self) -> list[str]:
        if not self.closed:
            self.root.update()
        keys, self.keys = self.keys, []
        return keys

    def close(self) -> None:
        if not self.closed:
            self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

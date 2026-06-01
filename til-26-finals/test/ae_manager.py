"""
stoopid guys
"""

import random


class AEManager:
    def __init__(self, grid_size: int = 16):
        self.grid_size = grid_size  # lol. lmao even

    def ae(self, observation: dict) -> int:
        return random.randint(0, 5)

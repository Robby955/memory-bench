"""Base class for memory mechanisms."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Optional


class MemoryModule(ABC):
    @abstractmethod
    def wrap_model(self, model: nn.Module, config) -> nn.Module:
        """Modify the GPT model to add memory. Called once at init."""

    def extra_param_groups(self) -> list[dict]:
        """Optimizer param groups for mechanism-specific parameters."""
        return []

    def on_segment_boundary(self, memory_state: torch.Tensor) -> torch.Tensor:
        """Called between segments (RMT). Returns memory for next segment."""
        return memory_state

    def reset(self):
        """Reset per-sequence state."""
        pass

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def num_memory_params(self) -> int:
        return 0

    @property
    def requires_segments(self) -> bool:
        return False

    @property
    def segment_length(self) -> Optional[int]:
        return None

    @property
    def num_memory_tokens(self) -> int:
        return 0

    @property
    def bptt_depth(self) -> int:
        return 1

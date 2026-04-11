from memory_bench.mechanisms.base import MemoryModule
from memory_bench.mechanisms.persistent import PersistentMemory
from memory_bench.mechanisms.rmt import RMTMemory
from memory_bench.mechanisms.ttt import TTTLinearMemory
from memory_bench.mechanisms.deltanet import GatedDeltaNetMemory

MECHANISMS = {
    "none": None,
    "persistent": PersistentMemory,
    "rmt": RMTMemory,
    "ttt": TTTLinearMemory,
    "deltanet": GatedDeltaNetMemory,
}

__all__ = [
    "MemoryModule",
    "PersistentMemory",
    "RMTMemory",
    "TTTLinearMemory",
    "GatedDeltaNetMemory",
    "MECHANISMS",
]

from __future__ import annotations

import ctypes
import importlib.util
import os
import platform
import sys
from dataclasses import dataclass
from typing import Any

from music_api.ai.schemas import ExecutionDevice


@dataclass(frozen=True)
class HardwareCapabilities:
    python_version: str
    operating_system: str
    cpu_count: int
    memory_total_bytes: int | None
    memory_available_bytes: int | None
    torch_available: bool
    torch_version: str | None
    cuda_available: bool
    cuda_devices: tuple[dict[str, Any], ...]
    mps_available: bool

    def as_response(self, configured_device: str) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "operating_system": self.operating_system,
            "cpu_count": self.cpu_count,
            "memory_total_bytes": self.memory_total_bytes,
            "memory_available_bytes": self.memory_available_bytes,
            "torch_available": self.torch_available,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_devices": list(self.cuda_devices),
            "mps_available": self.mps_available,
            "configured_device": configured_device,
        }

    def available_devices(self) -> tuple[ExecutionDevice, ...]:
        devices = [ExecutionDevice.CPU]
        if self.cuda_available:
            devices.append(ExecutionDevice.CUDA)
        if self.mps_available:
            devices.append(ExecutionDevice.MPS)
        return tuple(devices)


def detect_hardware() -> HardwareCapabilities:
    memory_total, memory_available = _system_memory()
    torch_available = importlib.util.find_spec("torch") is not None
    torch_version: str | None = None
    cuda_available = False
    cuda_devices: tuple[dict[str, Any], ...] = ()
    mps_available = False
    if torch_available:
        import torch

        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_devices = tuple(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
                }
                for index in range(torch.cuda.device_count())
            )
        mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    return HardwareCapabilities(
        python_version=platform.python_version(),
        operating_system=platform.system(),
        cpu_count=os.cpu_count() or 1,
        memory_total_bytes=memory_total,
        memory_available_bytes=memory_available,
        torch_available=torch_available,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_devices=cuda_devices,
        mps_available=mps_available,
    )


def _system_memory() -> tuple[int | None, int | None]:
    if importlib.util.find_spec("psutil") is not None:
        import psutil

        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical), int(status.available_physical)
        return None, None
    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (
                int(page_size * os.sysconf("SC_PHYS_PAGES")),
                int(page_size * os.sysconf("SC_AVPHYS_PAGES")),
            )
        except (OSError, ValueError):
            return None, None
    return None, None

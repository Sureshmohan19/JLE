"""JLE.core.hardware_utils"""

import os
import enum
import pathlib
import glob
from typing import Tule, Optional

_GOOGLE_PCI_VENDOR_ID = '0x1ae0'

_NVIDIA_GPU_DEVICES = [
    '/dev/nvidia0',
    '/dev/nvidiactl',  # Docker/Kubernetes
    '/dev/dxg',  # WSL2
]

class TpuVersion(enum.IntEnum):
    v2 = 0 # TPU v2
    v3 = 1 # TPU v3
    plc = 2
    v4 = 3 # TPU v4
    v5p = 4 # TPU v5p
    v5e = 5 # TPU v5e
    v6e = 6 # TPU v6e
    tpu7x = 7 # TPU7x

_TPU_PCI_DEVICE_IDS = {
    '0x0027': TpuVersion.v3,
    '0x0056': TpuVersion.plc,
    '0x005e': TpuVersion.v4,
    '0x0062': TpuVersion.v5p,
    '0x0063': TpuVersion.v5e,
    '0x006f': TpuVersion.v6e,
    '0x0076': TpuVersion.tpu7x,
}

def has_nvidia_gpu() -> bool:
    return any(os.path.exists(d) for d in _NVIDIA_GPU_DEVICES)

def num_chips_and_device_id():
    # just an exact copy from JAX as we have no idea at this point 
    # about how drivers are installed and accessed
    num_chips = 0
    tpu_version: Optional[TpuVersion] = None
    for vendor_file in glob.glob('/sys/bus/pci/devices/*/vendor'):
        if pathlib.Path(vendor_file).read_text().strip() != _GOOGLE_PCI_VENDOR_ID:
            continue

        device_file = os.path.join(os.path.dirname(vendor_file), 'device')
        device_id = pathlib.Path(device_file).read_text().strip()
        if device_id in _TPU_PCI_DEVICE_IDS:
            tpu_version = _TPU_PCI_DEVICE_IDS[device_id]
            num_chips += 1

    return num_chips, tpu_version

def transparent_hugepages_enabled() -> bool:
  # https://docs.kernel.org/admin-guide/mm/transhuge.html for more info about this
  path = pathlib.Path('/sys/kernel/mm/transparent_hugepage/enabled')
  return path.exists() and path.read_text().strip() == '[always] madvise never'
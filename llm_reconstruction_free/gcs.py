"""GCS Utils.
"""

import os
import subprocess
import torch


_INITIALIZED = False


def init():
    global _INITIALIZED
    if not _INITIALIZED:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        _INITIALIZED = True


def local_copy(from_gcs: str, infix: str, name: str) -> str:
    """Copy GCS file to local file."""
    init()
    local_cache = os.path.join(os.getcwd(), infix, name)
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        print(f"Copying {infix} from GCS...")
        print("MKDIR:" + subprocess.run(
            ["mkdir", "-p", local_cache], stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE, check=True).stdout.decode("utf-8"))
        print("GSUTIL:" + subprocess.run(
            ["gsutil", "-m", "cp", "-r",
             os.path.join(from_gcs, infix, name, "*"), local_cache],
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE, check=True).stdout.decode("utf-8"))
        print("LS:" + subprocess.run(
            ["ls", "-l", local_cache],
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE, check=True).stdout.decode("utf-8"))
        print(f"Finished copying {infix} from GCS...")
    print(f">>>>>{local_rank} entering barrier")
    torch.distributed.barrier()
    print(f">>>>>{local_rank} exiting barrier")
    return local_cache

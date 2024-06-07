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
    """Copy GCS file to local file.

    Args:
      from_gcs: GCS path in the format of "gs://<bucket>".
      infix: infix. Suggested values are "models", "datasets", "tokenizers".
      name: name. Suggested values are "microsoft/phi-2", etc.
    
    Returns:
      Local file path.
    """
    init()
    local_cache = os.path.join(os.getcwd(), infix, name)
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        print(f"Copying {infix} from GCS...")
        remove_local_cache = True
        try:
            subprocess.run(["ls", "-l", local_cache], check=True)
        except subprocess.CalledProcessError:
            remove_local_cache = False
        if remove_local_cache:
            subprocess.run(["rm", "-rf", local_cache], check=True)
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
    torch.distributed.barrier()
    return local_cache

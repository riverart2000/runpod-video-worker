from runtime_cache import bootstrap_runtime_dirs


bootstrap_runtime_dirs()

import runpod

from runpod_video_worker import process_runpod_job


def handler(event):
    return process_runpod_job(event)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
from typing import List, Optional

import logging

import os
import pynvml
import psutil

from pydantic import BaseModel, Field, validator

from fastapi import FastAPI, APIRouter

import haystack

from rest_api.utils import get_app
from rest_api.config import LOG_LEVEL

logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")


router = APIRouter()
app: FastAPI = get_app()


class CPUUsage(BaseModel):
    used: float = Field(..., description="REST API average CPU usage in percentage")

    @validator("used")
    @classmethod
    def used_check(cls, v):
        return round(v, 2)


class MemoryUsage(BaseModel):
    used: float = Field(..., description="REST API used memory in percentage")

    @validator("used")
    @classmethod
    def used_check(cls, v):
        return round(v, 2)


class GPUUsage(BaseModel):
    kernel_usage: float = Field(..., description="GPU kernel usage in percentage")
    memory_total: int = Field(..., description="Total GPU memory in megabytes")
    memory_used: Optional[int] = Field(..., description="REST API used GPU memory in megabytes")

    @validator("kernel_usage")
    @classmethod
    def kernel_usage_check(cls, v):
        return round(v, 2)


class GPUInfo(BaseModel):
    index: int = Field(..., description="GPU index")
    usage: GPUUsage = Field(..., description="GPU usage details")


class HealthResponse(BaseModel):
    version: str = Field(..., description="Haystack version")
    cpu: CPUUsage = Field(..., description="CPU usage details")
    memory: MemoryUsage = Field(..., description="Memory usage details")
    gpus: List[GPUInfo] = Field(default_factory=list, description="GPU usage details")


def get_cpu_usage() -> CPUUsage:
    cpu_count = os.cpu_count() or 1
    p = psutil.Process()
    p_cpu_usage = p.cpu_percent() / cpu_count
    return CPUUsage(used=p_cpu_usage)


def get_memory_usage() -> MemoryUsage:
    p = psutil.Process()
    p_memory_usage = p.memory_percent()
    return MemoryUsage(used=p_memory_usage)


def get_gpu_usage() -> List[GPUInfo]:
    gpus: List[GPUInfo] = []
    try:
        pynvml.nvmlInit()
        try:
            gpu_count = pynvml.nvmlDeviceGetCount()
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_total = float(info.total) / 1024 / 1024
                gpu_mem_used = None
                try:
                    for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                        if proc.pid == os.getpid():
                            gpu_mem_used = float(proc.usedGpuMemory) / 1024 / 1024
                            break
                except pynvml.NVMLError:
                    # ignore if nvmlDeviceGetComputeRunningProcesses is not supported
                    # this can happen for outdated drivers
                    pass
                gpu_info = GPUInfo(
                    index=i,
                    usage=GPUUsage(
                        memory_total=round(gpu_mem_total),
                        kernel_usage=pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                        memory_used=round(gpu_mem_used) if gpu_mem_used is not None else None,
                    ),
                )
                gpus.append(gpu_info)
        except pynvml.NVMLError as e:
            logger.warning("Couldn't collect GPU stats: %s", str(e))
        finally:
            pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        # Here we intentionally ignore errors that occur when NVML (NVIDIA Management Library) is not available
        # or found. See the original code's comment for more details.
        pass

    return gpus


@router.get("/health", response_model=HealthResponse, status_code=200)
def get_health_status():
    """
    This endpoint allows external systems to monitor the health of the Haystack REST API.
    """

    return HealthResponse(
        version=haystack.__version__, cpu=get_cpu_usage(), memory=get_memory_usage(), gpus=get_gpu_usage()
    )

from typing import List

import logging

import os
import pynvml
import psutil

from pydantic import BaseModel, Field

from fastapi import FastAPI, APIRouter

import haystack

from rest_api.utils import get_app
from rest_api.config import LOG_LEVEL

logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")


router = APIRouter()
app: FastAPI = get_app()


class CPUUsage(BaseModel):
    avg_1m: str = Field(..., description="Average CPU usage over the last 1 minute")
    avg_5m: str = Field(..., description="Average CPU usage over the last 5 minutes")
    avg_15m: str = Field(..., description="Average CPU usage over the last 15 minutes")


class MemoryUsage(BaseModel):
    total: str = Field(..., description="Total memory in megabytes")
    used: str = Field(..., description="Used memory in megabytes")
    percent: str = Field(..., description="Percentage of used memory")


class GPUUsage(BaseModel):
    memory_total: str = Field(..., description="Total GPU memory in megabytes")
    memory_used: str = Field(..., description="Used GPU memory in megabytes")
    memory_percent: str = Field(..., description="Percentage of used GPU memory")
    kernel_usage: str = Field(..., description="GPU kernel usage in percent")


class GPUInfo(BaseModel):
    index: int = Field(..., description="GPU index")
    usage: GPUUsage = Field(..., description="GPU usage details")


class HaystackInfo(BaseModel):
    version: str = Field(..., description="Haystack version")
    cpu_usage: str = Field(..., description="REST API CPU usage details")
    memory_usage: str = Field(..., description="REST API memory usage details")


class SystemInfo(BaseModel):
    cpu: CPUUsage = Field(..., description="System CPU usage details")
    memory: MemoryUsage = Field(..., description="System memory usage details")
    gpu_info: List[GPUInfo] = Field(
        default_factory=list, description="System GPU usage details"
    )


class HealthResponse(BaseModel):
    haystack: HaystackInfo = Field(..., description="Haystack REST API health details")
    system: SystemInfo = Field(..., description="System health details")


@router.get("/health", response_model=HealthResponse, status_code=200)
def get_health_status():
    """
    This endpoint allows external systems to monitor the health of the Haystack REST API.
    """

    cpu_1m, cpu_5m, cpu_15m = psutil.getloadavg()
    cpu_count = float(os.cpu_count() or 0)
    cpu_1m = (cpu_1m / cpu_count) * 100
    cpu_5m = (cpu_5m / cpu_count) * 100
    cpu_15m = (cpu_15m / cpu_count) * 100
    cpu_current = CPUUsage(
        avg_1m=f"{cpu_1m:.2f}%", avg_5m=f"{cpu_5m:.2f}%", avg_15m=f"{cpu_15m:.2f}%"
    )

    memory_data = psutil.virtual_memory()
    memory_total = memory_data.total / 1024 / 1024
    memory_used = memory_data.used / 1024 / 1024
    memory_percent = memory_data.percent
    memory_usage = MemoryUsage(
        total=f"{memory_total:.2f}MB",
        used=f"{memory_used:.2f}MB",
        percent=f"{memory_percent:.2f}%",
    )

    gpus: List[GPUInfo] = []

    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        gpu_mem_total = ""
        gpu_mem_used = ""
        gpu_mem_free = ""
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem_total = float(info.total) / 1024 / 1024
            gpu_mem_used = float(info.used) / 1024 / 1024
            gpu_info = GPUInfo(
                index=i,
                usage=GPUUsage(
                    memory_total=f"{gpu_mem_total:.2f}MB",
                    memory_used=f"{gpu_mem_used:.2f}MB",
                    memory_percent=f"{(gpu_mem_used/gpu_mem_total)*100:.2f}%",
                    kernel_usage=f"{pynvml.nvmlDeviceGetUtilizationRates(handle).gpu:.2f}%",
                ),
            )
            gpus.append(gpu_info)
    except pynvml.NVMLError:
        logger.warning("No NVIDIA GPU found.")

    p_cpu_usage = 0
    p_memory_usage = 0
    p = psutil.Process()
    with p.oneshot():
        p_cpu_usage = p.cpu_percent() / cpu_count
        p_memory_usage = p.memory_percent()

    return HealthResponse(
        haystack=HaystackInfo(
            version=haystack.__version__,
            cpu_usage=f"{p_cpu_usage:.2f}%",
            memory_usage=f"{p_memory_usage:.2f}%",
        ),
        system=SystemInfo(cpu=cpu_current, memory=memory_usage, gpu_info=gpus),
    )

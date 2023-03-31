variable "HAYSTACK_VERSION" {
  default = "main"
}

variable "GITHUB_REF" {
  default = ""
}

variable "IMAGE_NAME" {
  default = "deepset/haystack"
}

variable "IMAGE_TAG_SUFFIX" {
  default = "local"
}

variable "BASE_IMAGE_TAG_SUFFIX" {
  default = "local"
}

variable "HAYSTACK_EXTRAS" {
  default = ""
}

group "base" {
  targets = ["base-cpu", "base-gpu"]
}

group "api" {
  targets = ["cpu", "gpu"]
}

group "api-latest" {
  targets = ["cpu-latest", "gpu-latest"]
}

group "all" {
  targets = ["base", "base-gpu", "cpu", "gpu"]
}

target "base-cpu" {
  dockerfile = "Dockerfile.base"
  tags = ["${IMAGE_NAME}:base-cpu-${IMAGE_TAG_SUFFIX}"]
  args = {
    build_image = "python:3.10-slim"
    base_image = "python:3.10-slim"
    haystack_version = "${HAYSTACK_VERSION}"
    haystack_extras = notequal("",HAYSTACK_EXTRAS) ? "${HAYSTACK_EXTRAS}" : "[docstores,crawler,preprocessing,ocr,onnx,beir]"
  }
  platforms = ["linux/amd64", "linux/arm64"]
}

target "base-gpu" {
  dockerfile = "Dockerfile.base"
  tags = ["${IMAGE_NAME}:base-gpu-${IMAGE_TAG_SUFFIX}"]
  args = {
    # pytorch/pytorch:1.13.1-cuda11.6 ships Python 3.10.8

    build_image = "pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime"
    base_image = "pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime"
    haystack_version = "${HAYSTACK_VERSION}"
    haystack_extras = notequal("",HAYSTACK_EXTRAS) ? "${HAYSTACK_EXTRAS}" : "[docstores-gpu,crawler,preprocessing,ocr,onnx-gpu]"
  }
  platforms = ["linux/amd64", "linux/arm64"]
}

target "cpu" {
  dockerfile = "Dockerfile.api"
  tags = ["${IMAGE_NAME}:cpu-${IMAGE_TAG_SUFFIX}"]
  args = {
    base_image = "${IMAGE_NAME}"
    base_image_tag = "base-cpu-${BASE_IMAGE_TAG_SUFFIX}"
  }
  platforms = ["linux/amd64", "linux/arm64"]
}

target "cpu-latest" {
  inherits = ["cpu"]
  tags = ["${IMAGE_NAME}:cpu"]
  platforms = ["linux/amd64", "linux/arm64"]
}

target "gpu" {
  dockerfile = "Dockerfile.api"
  tags = ["${IMAGE_NAME}:gpu-${IMAGE_TAG_SUFFIX}"]
  args = {
    base_image = "${IMAGE_NAME}"
    base_image_tag = "base-gpu-${BASE_IMAGE_TAG_SUFFIX}"
  }
  platforms = ["linux/amd64", "linux/arm64"]
}

target "gpu-latest" {
  inherits = ["gpu"]
  tags = ["${IMAGE_NAME}:gpu"]
  platforms = ["linux/amd64", "linux/arm64"]
}

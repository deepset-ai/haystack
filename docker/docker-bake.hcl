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
  targets = ["base-cpu", "base-gpu", "base-cpu-remote-inference"]
}

group "api" {
  targets = ["cpu", "gpu", "cpu-remote-inference"]
}

group "api-latest" {
  targets = ["cpu-latest", "gpu-latest", "cpu-remote-inference-latest"]
}

group "all" {
  targets = ["base", "base-gpu", "cpu", "gpu", "cpu-remote-inference"]
}

target "base-cpu" {
  dockerfile = "Dockerfile.base"
  tags = ["${IMAGE_NAME}:base-cpu-${IMAGE_TAG_SUFFIX}"]
  args = {
    build_image = "python:3.10-slim"
    base_image = "python:3.10-slim"
    haystack_version = "${HAYSTACK_VERSION}"
    haystack_extras = notequal("",HAYSTACK_EXTRAS) ? "${HAYSTACK_EXTRAS}" : "[docstores,inference,crawler,preprocessing,file-conversion,ocr,onnx,metrics,beir]"
  }
  platforms = ["linux/amd64", "linux/arm64"]
}

target "base-cpu-remote-inference" {
  inherits = ["base-cpu"]
  tags = ["${IMAGE_NAME}:base-cpu-remote-inference-${IMAGE_TAG_SUFFIX}"]
  args = {
    haystack_extras = notequal("",HAYSTACK_EXTRAS) ? "${HAYSTACK_EXTRAS}" : "[preprocessing]"
  }
}

target "base-gpu" {
  dockerfile = "Dockerfile.base"
  tags = ["${IMAGE_NAME}:base-gpu-${IMAGE_TAG_SUFFIX}"]
  args = {
    # pytorch/pytorch:1.13.1-cuda11.6 ships Python 3.10.8

    build_image = "pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime"
    base_image = "pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime"
    haystack_version = "${HAYSTACK_VERSION}"
    haystack_extras = notequal("",HAYSTACK_EXTRAS) ? "${HAYSTACK_EXTRAS}" : "[docstores-gpu,inference,crawler,preprocessing,file-conversion,ocr,onnx-gpu,metrics]"
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

target "cpu-remote-inference" {
  dockerfile = "Dockerfile.api"
  tags = ["${IMAGE_NAME}:cpu-remote-inference-${IMAGE_TAG_SUFFIX}"]
  args = {
    base_image = "${IMAGE_NAME}"
    base_image_tag = "base-cpu-remote-inference-${BASE_IMAGE_TAG_SUFFIX}"
  }
  platforms = ["linux/amd64", "linux/arm64"]
}

target "cpu-remote-inference-latest" {
  inherits = ["cpu-remote-inference"]
  tags = ["${IMAGE_NAME}:cpu-remote-inference"]
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

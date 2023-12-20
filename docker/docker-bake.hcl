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

target "base" {
  dockerfile = "Dockerfile.base"
  tags = ["${IMAGE_NAME}:base-${IMAGE_TAG_SUFFIX}"]
  args = {
    build_image = "python:3.10-slim"
    base_image = "python:3.10-slim"
    haystack_version = "${HAYSTACK_VERSION}"
  }
  platforms = ["linux/amd64", "linux/arm64"]
}

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

variable "IS_STABLE" {
  default = "false"
}

target "base" {
  dockerfile = "Dockerfile.base"
  tags = concat(
    ["${IMAGE_NAME}:base-${IMAGE_TAG_SUFFIX}"],
    ${IS_STABLE == "true" ? ["${IMAGE_NAME}:stable"] : []}
  )
  args = {
    build_image = "python:3.12-slim"
    base_image = "python:3.12-slim"
    haystack_version = "${HAYSTACK_VERSION}"
  }
  platforms = ["linux/amd64", "linux/arm64"]
}

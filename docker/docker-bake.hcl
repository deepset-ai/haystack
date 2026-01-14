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
  tags = "${compact([
    "${IMAGE_NAME}:base-${IMAGE_TAG_SUFFIX}",
    equal(IS_STABLE, "true") ? "${IMAGE_NAME}:stable" : ""
  ])}"
  args = {
    build_image = "python:3.12-slim"
    base_image = "python:3.12-slim"
    haystack_version = "${HAYSTACK_VERSION}"
  }
  platforms = ["linux/arm64"]
}

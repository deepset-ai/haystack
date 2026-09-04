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

# 2.Y.Z releases are also tagged as "stable"
# Example: 2.99.0 is tagged as base-2.99.0 and stable

target "base" {
  dockerfile = "Dockerfile.base"
  tags = "${compact([
    "${IMAGE_NAME}:base-${IMAGE_TAG_SUFFIX}",
    equal("${IS_STABLE}", "true") ? "${IMAGE_NAME}:stable" : ""
  ])}"
  args = {
    build_image = "python:3.12-slim@sha256:090ba77e2958f6af52a5341f788b50b032dd4ca28377d2893dcf1ecbdfdfe203"
    base_image = "python:3.12-slim@sha256:090ba77e2958f6af52a5341f788b50b032dd4ca28377d2893dcf1ecbdfdfe203"
    haystack_version = "${HAYSTACK_VERSION}"
  }
  platforms = ["linux/amd64", "linux/arm64"]
}

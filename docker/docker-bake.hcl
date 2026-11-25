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
    build_image = "python:3.12-slim@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2"
    base_image = "python:3.12-slim@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2"
    haystack_version = "${HAYSTACK_VERSION}"
  }
  platforms = ["linux/amd64", "linux/arm64"]
}

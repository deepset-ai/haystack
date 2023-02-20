variable "XPDF_VERSION" {
  default = "4.04"
}

target "xpdf" {
  dockerfile = "Dockerfile.xpdf"
  tags = ["deepset/xpdf:latest"]
  args = {
    XPDF_VERSION = "4.04"
  }
  platforms = ["linux/amd64", "linux/arm64"]
}

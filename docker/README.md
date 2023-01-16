<p align="center">
  <a href="https://www.deepset.ai/haystack/"><img src="https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/haystack_logo_colored.png" alt="Haystack"></a>
</p>

Haystack is an end-to-end framework that enables you to build powerful and production-ready
pipelines for different search use cases. The Docker image comes with a web service
configured to serve Haystack's `rest_api` to ease pipeline deployments in containerized
environments.

To start the Docker container binding the TCP port `8000` locally, run:
```sh
docker run -p 8000:8000 deepset/haystack
```

If you need the container to access other services available in the host, run:
```sh
docker run -p 8000:8000 --network="host" deepset/haystack
```

## Image Variants

The Docker image comes in four variants:
- `haystack:gpu-<version>` contains Haystack dependencies as well as what's needed to run the REST API and UI. It comes with the CUDA runtime and is capable of running on GPUs.
- `haystack:cpu-<version>` contains Haystack dependencies as well as what's needed to run the REST API and UI. It has no support for GPU so must be run on CPU.
- `haystack:base-gpu-<version>` only contains the Haystack dependencies. It comes with the CUDA runtime and is capable of running on GPUs.
- `haystack:base-cpu-<version>` only contains the Haystack dependencies. It has no support for GPU so must be run on CPU.

## Image Development

Images are built with BuildKit and we use `bake` to orchestrate the process.
You can build a specific image by running:
```sh
docker buildx bake gpu
```

You can override any `variable` defined in the `docker-bake.hcl` file and build custom
images, for example if you want to use a branch from the Haystack repo, run:
```sh
HAYSTACK_VERSION=mybranch_or_tag BASE_IMAGE_TAG_SUFFIX=latest docker buildx bake gpu --no-cache
```

### Multi-Platform Builds

Haystack images support multiple architectures. But depending on your operating system and Docker
environment, you might not be able to build all of them locally.

You may encounter the following error when trying to build the image:

```
multiple platforms feature is currently not supported for docker driver. Please switch to a different driver
(eg. “docker buildx create --use”)
```

To get around this, you need to override the `platform` option and limit local builds to the same architecture as
your computer's. For example, on an Apple M1 you can limit the builds to ARM only by invoking `bake` like this:

```sh
docker buildx bake base-cpu --set "*.platform=linux/arm64"
```

# License

View [license information](https://github.com/deepset-ai/haystack/blob/main/LICENSE) for
the software contained in this image.

As with all Docker images, these likely also contain other software which may be under
other licenses (such as Bash, etc from the base distribution, along with any direct or
indirect dependencies of the primary software being contained).

As for any pre-built image usage, it is the image user's responsibility to ensure that any
use of this image complies with any relevant licenses for all software contained within.

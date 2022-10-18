# Haystack Docker image

Haystack is an end-to-end framework that enables you to build powerful and production-ready
pipelines for different search use cases. The Docker image comes with a web service
configured to serve Haystack's `rest_api` to ease pipelines' deployments in containerized
environments.

Start the Docker container binding the TCP port `8000` locally:
```sh
docker run -p 8000:8000 deepset/haystack
```

If you need the container to access other services available in the host:
```sh
docker run -p 8000:8000 --network="host" deepset/haystack
```

## Image variants

The Docker image comes in two variants:
- `haystack:cpu-<version>`: this image is smaller but doesn't support GPU
- `haystack:gpu-<version>`: this image comes with the Cuda runtime and is capable of running on GPUs


## Image development

Images are built with BuildKit and we use `bake` to orchestrate the process.
You can build a specific image by simply run:
```sh
docker buildx bake gpu
```

You can override any `variable` defined in the `docker-bake.hcl` file and build custom
images, for example if you want to use a branch from the Haystack repo:
```sh
HAYSTACK_VERSION=mybranch_or_tag BASE_IMAGE_TAG_SUFFIX=latest docker buildx bake gpu --no-cache
```

### A note about multi-platform builds

Haystack images support multiple architectures, but depending on your operating system and Docker
environment you might not be able to build all of them locally. If you get an error like:
```
multiple platforms feature is currently not supported for docker driver. Please switch to a different driver
(eg. “docker buildx create --use”)
```

you might need to override the `platform` option and limit local builds to the same architecture as
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
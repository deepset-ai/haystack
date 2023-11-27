from importlib import metadata

# haystack is distributed as a separate package called `haystack-ai`.
# We want to keep all preview dependencies separate from the current Haystack version,
# so imports in haystack must only import from haystack.
# Since we need to access __version__ in haystack without importing from
# haystack we must set it here too.
# When installing `haystack-ai` we want to use that package version though
# as `farm-haystack` might not be installed and cause this to fail.
try:
    __version__ = str(metadata.version("haystack-ai"))
except metadata.PackageNotFoundError:
    __version__ = str(metadata.version("farm-haystack"))

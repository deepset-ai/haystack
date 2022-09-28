# Pipeline management in REST API

* Status: proposed <!-- optional -->
* Deciders: @danielbichuetti <!-- optional -->
* Date: 2022-09-28 <!-- optional -->

Technical Story: <https://github.com/deepset-ai/haystack/issues/3175> <!-- optional -->

## Context and Problem Statement

REST API has been originally implemented to load a yaml file, defined in an environment variable, when the FastAPI application was loaded to set up it's pipeline. However, this approach has some limitations, such as not being able to add, remove or update the pipeline without restarting the server, the docker container, or even the kubernetes pod.

## Decision Drivers <!-- optional -->

* Users which are making initial contact may find it dfficult to be stopping and restarting a server/container to test out different configurations.
* When working with large kubernetes deployments the overhead introduced at scale with pod rollouts, can be expensive in terms of time and resources.

## Considered Options

* Container/server restart after file is modified
* Pod rollout after file is modified
* Automatically monitor for changes in the yaml file and reload the pipeline
* Add a pipeline management endpoint to the REST API


## Decision Outcome

Chosen option is to "Add a pipeline management endpoint to the REST API", because it is the most flexible and cost effective option. It allows users to add, remove or update the one or multiple pipelines without restarting the server, the docker container, or even the kubernetes pod.

### Positive Consequences <!-- optional -->

* Users can add, remove or update one or multiple pipelines without touching the infrastructure, just by calling the REST API endpoint.
* It's more cost effective for large enterprise deployments.

### Negative Consequences <!-- optional -->

* It's a bit more complex to implement and maintain (negligible).

## Pros and Cons of the Options <!-- optional -->

### Container/server restart after file is modified

* Good, no code change would need to be made
* Bad, can't update the pipeline in realtime
* Bad, it's not possible to use multiple pipelines at the same time
* Bad, first time users have a bad experience when experimenting with the pipelines
* Bad, large scale deployments may be impacted by the time and resources required to restart the server/container

### Pod rollout after file is modified

* Good, because no code change would need to be made
* Good, in a kubernetes environment, the file can be any mapped volume
* Bad, can't update the pipeline in realtime
* Bad, it's not possible to use multiple pipelines at the same time
* Bad, kubernetes deployments would need to be configured to automatically rollout (IaC) the pods after the file is modified
* Bad, large scale deployments may be impacted the by time and resources required to rollout the pod

### Automatically monitor for changes in the yaml file and reload the pipeline

* Good, there is no need to restart the server/container/pod
* Good, there is no need to implement a manage endpoint
* Good, in a kubernetes environment, the file can be any mapped volume
* Bad, there is an IO overhead to monitor the file
* Bad, in a kubernetes environment the file would be mounted using a volume mount, any changes would have to be made in terms
more connected to the infrastructure team, or using IaC, and not the application team

### Add a pipeline management endpoint to the REST API

* Good, there is no need to restart the server/container/pod
* Good, there is no IO overhead to monitor the file
* Good, developer can add, remove or update the pipeline in realtime
* Good, after deployment the pipeline can be managed by the application team without the need to involve the infrastructure team or code
* Bad, there is an architecture change on the REST API

## Links <!-- optional -->

<https://kubernetes.io/docs/concepts/storage/volumes/#volume-types>
<https://github.com/deepset-ai/haystack/issues/3175>
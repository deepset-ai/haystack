# Haystack - Helm chart

Here you can read the documentation to a haystack deployment using [helm]([https://helm.sh/](https://helm.sh/)) for [Kubernetes]([https://kubernetes.io/](https://kubernetes.io/)).

Haystack has 3 main architectural components:

- haystack API
- haystack UI
- datastore (elasticsearch)

In the following directory you can find helm charts for haystack API (`haystack-api`) and haystack UI (`haystack-ui`) deployments.

*Elasticsearch is not included since it may vary how and where users want to run their Elasticsearch cluster.*

## Deployment guide

### Requirements

- functioning Kubernetes cluster
- network level access to the Kubernetes cluster API
- sufficient permissions to deploy helm charts

### Prerequisites

MISSING: elasticsearch deployment guide

### Simple install from local repository with defaults

1. Go to the helm chart directory in haystack repository (`cd $HAYSTACK_REPOSITORY/helm`)
2. First we deploy the `haystack-api` helm chart, for that we need to run `helm install haystack-api-deployment haystack-api`. This will deploy the API to the namespace in the current Kubernetes context.
3. To validate the deployment you can run `helm list` or `kubectl get pods` and see if the haystack API pods are in running state.
4. If all worked well we can now deploy the UI component, for that we need to run `helm install haystack-ui`. This will deploy the UI to the namespace in the current Kubernetes context. **Note that you should not change context between deployments if you don’t know Kubernetes that well!**
5. Voila! All components are deployed! See the section about how to access your deployments!

### Simple install from local repository with overrides

MISSING

### Install from remote repository with default

MISSING

### Install from remote repository with overrides

MISSING
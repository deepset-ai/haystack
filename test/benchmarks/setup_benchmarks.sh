git reset --hard
git clean -f -x
git checkout main
git pull
pip install .
pip install .[metrics,elasticsearch,weaviate,opensearch]
mkdir out

./run.sh
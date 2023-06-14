git reset --hard
git clean -f -x
git checkout main
git pull
cd ../../
pip install .
pip install .[metrics,elasticsearch,weaviate,opensearch]
cd test/benchmarks
mkdir out

./run.sh
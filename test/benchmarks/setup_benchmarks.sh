to_checkout=${1:-"main"}

git reset --hard
git clean -f -x
git checkout $to_checkout

if [ $? -eq 0 ]; then
    echo "checkout succeeded"
else
    echo "checkout failed..do not continue"
    exit 1
fi

git pull
cd ../../
pip install .
pip install .[metrics,elasticsearch,weaviate,opensearch,benchmarks]
cd test/benchmarks
mkdir +p out

./run.sh
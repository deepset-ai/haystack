git reset --hard
git clean -f -x
git checkout main
git pull
pip install .
pip install .[metrics,elasticsearch,weaviate,opensearch]

mkdir out

for f in ./configs/**/*.(yml|yaml); do 
    name="${f%.*}"
    echo "=== Running benchmarks for $name.json ===";
    python run.py --output "out/$name.json" $f;
done


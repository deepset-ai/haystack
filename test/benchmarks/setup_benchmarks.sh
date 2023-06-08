rm -rf haystack
git clone https://github.com/deepset-ai/haystack.git
pip install .
pip install .[metrics,elasticsearch,weaviate,opensearch]

cd test/benchmarks
mkdir data

wget -O data/msmarco.100_000.tar.bz2 https://deepset-test-datasets.s3.eu-central-1.amazonaws.com/msmarco.100_000.tar.bz2
tar -xf data/msmarco.100_000.tar.bz2 -C data

mkdir out

for f in ./configs/**/*.(yml|yaml); do 
    name="${f%.*}"
    echo "=== Running benchmarks for $name.json ===";
    python run.py --output "out/$name.json" $f;
done


set +e

for f in ./configs/**/*.yml; do 
    echo "kill docker containers"
    docker kill $(docker ps -q)
    start_benchmark=$(date +%s%N)
    name="${f%.*}"
    echo "=== Running benchmarks for $name at $date ===";
    printf "\n\nstarting benchmark for $name\n" >> benchmark_log.txt
    config_name=$(basename $name)
    python run.py --output "out/$config_name.json" $f;
    end_benchmark=$(date +%s%N);
    diff_benchmark=$((end_benchmark-start_benchmark));
    printf "benchmarks for $name took %s.%s seconds \n" "${diff_benchmark:0: -9}" "${diff_benchmark: -9:3}" >> benchmark_log.txt;
    echo "=== benchmarks done (or failed) at $date ==="
done

b=$(date +%s%N)
diff=$((b-a))

printf "benchmarks took %s.%s seconds \n" "${diff:0: -9}" "${diff: -9:3}" >> benchmark_log.txt

python datadog/send_metrics.py
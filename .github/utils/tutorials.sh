#!/bin/bash

export LAUNCH_GRAPHDB=0      # See tut 10 - GraphDB is already running in CI
export TIKA_LOG_PATH=$PWD    # Avoid permission denied errors while importing tika

python_path=$1
files_changed=$2
exclusion_list=$3
make_python_path_editable=$4
containers_policy=$5
no_got_tutorials='4_FAQ_style_QA 5_Evaluation 7_RAG_Generator 8_Preprocessing 10_Knowledge_Graph 15_TableQA 16_Document_Classifier_at_Index_Time'

echo "Files changed in this PR: $files_changed"
echo "Excluding: $exclusion_list"
echo "Python path is editable: $make_python_path_editable"
echo "Containers policy: $containers_policy"

# Collect the tutorials to run
scripts_to_run=""
for script in $files_changed; do
    
    if [[ "$script" != *"tutorials/Tutorial"* ]] || ([[ "$script" != *".py"* ]] && [[ "$script" != *".ipynb"* ]]); then 
        echo "- not a tutorial: $script"
        continue
    fi
    
    skip_to_next=0
    for excluded in $exclusion_list; do
        if [[ "$script" == *"$excluded"* ]]; then skip_to_next=1; fi
    done
    if [[ $skip_to_next == 1 ]]; then 
        echo "- excluded: $script"
        continue
    fi

    scripts_to_run="$scripts_to_run $script"
done


# Run the containers
docker run -d -p 9200:9200 --name elasticsearch -e "discovery.type=single-node" -e "ES_JAVA_OPTS=-Xms128m -Xmx256m" elasticsearch:7.9.2
docker run -d -p 9998:9998 --name tika -e "TIKA_CHILD_JAVA_OPTS=-JXms128m" -e "TIKA_CHILD_JAVA_OPTS=-JXmx128m" apache/tika:1.24.1


failed=""
for script in $scripts_to_run; do

    echo ""
    echo "##################################################################################"
    echo "##################################################################################"
    echo "##  Running $script ..."
    echo "##################################################################################"
    echo "##################################################################################"

    # Do not cache GoT data
    reduce_dataset=1
    for no_got_tut in $no_got_tutorials; do
        if [[ "$script" == *"$no_got_tut"* ]]; then
            reduce_dataset=0
        fi
    done
    
    if [[ $reduce_dataset == 1 ]]; then
        # Copy the reduced GoT data into a folder named after the tutorial 
        # to trigger the caching mechanism of `fetch_archive_from_http`
        echo "Using reduced GoT dataset"
        no_prefix=${script#"tutorials/Tutorial"}
        split_on_underscore=(${no_prefix//_/ })
        cp -r data/tutorials data/tutorial${split_on_underscore[0]}
    else
        echo "NOT using reduced GoT dataset!"
    fi

    # FIXME Make the Python path editable
    # espnet needs to edit files on the PYTHONPATH during execution. However, by default GH runners don't allow
    # workflows to edit files into that directory, so in case of tutorials using espnet, we need to make PYTHONPATH
    # editable first. For now it's only Tutorial 17.
    # Still unclear why it's needed to repeat this operation, but if Tutorial 17 is run twice (once for the .py 
    # and once for .ipynb version) the error re-appears.
    if [[ $make_python_path_editable == "EDITABLE" ]] && [[ "$script" == *"Tutorial17_"* ]]; then
        sudo find $python_path/lib -type f -exec chmod 777 {} \;
    fi

    if [[ "$script" == *".py" ]]; then
        output=$(time python $script)
    else
        output=$(sudo $python_path/bin/ipython -c "%run $script")
    fi

    echo $output > $script-output.txt
    if [ $? -eq 0 ]; then
        echo "Execution completed successfully."
    else
        echo "===================================================="
        echo "|  $script FAILED!"
        echo "===================================================="
        echo "Output of the execution: "
        echo $output
        failed=$failed" "$script
    fi

    # Restart the necessary containers
    # Note: Tika does not store data and therefore can be left running
    if [[ "$make_python_path_editable" == "RESTART" ]]; then
        docker stop elasticsearch
        docker rm elasticsearch        
        docker run -d -p 9200:9200 --name elasticsearch -e "discovery.type=single-node" -e "ES_JAVA_OPTS=-Xms128m -Xmx256m" elasticsearch:7.9.2
    fi

    # Clean up datasets and SQLite DBs to avoid crashing the next tutorial
    git clean -f

done

# causes permission errors on Post Cache
sudo rm -rf data/
sudo rm -rf /home/runner/work/haystack/haystack/elasticsearch-7.9.2/


if [[ $failed == "" ]]; then
    echo ""
    echo ""
    echo "------------------------------------------"
    echo " All tutorials were executed successfully "
    echo "------------------------------------------"
    exit 0

else
    echo ""
    echo "##################################################################################"
    echo "##                                                                              ##"
    echo "##                        Some tutorials have failed!                           ##"
    echo "##                                                                              ##"
    echo "##################################################################################"
    for script in $failed; do
    echo "##  - $script"
    done
    echo "##################################################################################"
    exit 1
fi
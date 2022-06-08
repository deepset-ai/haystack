#!/bin/bash

export LAUNCH_GRAPHDB=0      # See tut 10 - GraphDB is already running in CI
export TIKA_LOG_PATH=$PWD    # Avoid permission denied errors while importing tika
set -e                       # Fails on any error in the following loop

scripts_to_run=$1
exclusion_list=$2
no_got_tutorials='4_FAQ_style_QA 5_Evaluation 7_RAG_Generator 8_Preprocessing 10_Knowledge_Graph 15_TableQA 16_Document_Classifier_at_Index_Time'
reduce_dataset=1

echo "Scripts to run: $scripts_to_run"
echo "Excluding: $exclusion_list"

for script in $scripts_to_run; do

    # Make sure we're running a tutorial file
    if [[ "$script" != *"tutorials/Tutorial"* ]] || ([[ "$script" != *".py"* ]] && [[ "$script" != *".ipynb"* ]]); then
        echo "Skipping $script, not a tutorial"
        continue
    fi
    
    # Exclude long tutorials
    skip_to_next=0
    for excluded in $exclusion_list; do
        if [[ "$script" == *"$excluded"* ]]; then skip_to_next=1; fi
    done
    if [[ $skip_to_next == 1 ]]; then 
        echo "Skipping $script, is in the exclusion list"
        continue
    fi
    
    # Do not cache GoT data
    for no_got_tut in no_got_tutorials; do
        if [[ "$script" == *"$no_got_tut"* ]]; then
            reduce_dataset = 0
        fi
    done
    
    if [[ $reduce_dataset == 1 ]]; then
        # Copy the reduced GoT data into a folder named after the tutorial 
        # to trigger the caching mechanism of `fetch_archive_from_http`
        no_prefix=${script#"tutorials/Tutorial"}
        split_on_underscore=(${no_prefix//_/ })
        cp -r data/tutorials data/tutorial${split_on_underscore[0]}
    fi
    
    echo ""
    echo "#############################################################"
    echo "  Running $script ..."
    echo "#############################################################"
    
    if [[ "$script" == *".py" ]]; then
        time python $script
    else
        sudo ${{ env.pythonLocation }}/bin/ipython -c "%run $script"
    fi
    git clean -f

done

# causes permission errors on Post Cache
sudo rm -rf data/
sudo rm -rf /home/runner/work/haystack/haystack/elasticsearch-7.9.2/
#!/bin/bash

search_dir=./config/best_models/condgce_new_trial

for entry in "$search_dir"/*
do
	echo $entry
        qsub launch.sh run.py $entry 1
done

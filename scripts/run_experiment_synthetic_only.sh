#!/bin/bash

python synthetic_only_main.py --path_to_data='./data/2011 Census Microdata Teaching File.csv' \
                --path_to_metadata='./data/2011 Census Microdata Teaching Discretized.json' \
                    --target_record_id=20\
                    --synthetic_scenario=2\
                    --n_original=1000\
                    --n_pos_test=5\
                    --n_pos_train=10\
                    --nbr_cores=1\
                    --unique='True'\
                     --name_generator='CTGAN'\
                    --cols_to_select="['all']"



python synthetic_only_main.py --path_to_data='./data/2011 Census Microdata Teaching File_sample.csv' \
                --path_to_metadata='./data/2011 Census Microdata Teaching Discretized.json' \
                    --target_record_id=2\
                    --synthetic_scenario=2\
                    --n_original=44\
                    --n_pos_test=4\
                    --n_pos_train=10\
                    --nbr_cores=1\
                    --unique='True'\
                    --name_generator='SYNTHPOP'\
                    --cols_to_select="['all']"

#!/bin/bash

#SemEval
python re_agcn_main.py --do_test --task_name semeval --data_dir ./data/semeval/ --model_path ./RE_AGCN.SEMEVAL.BERT.L/ --do_lower_case

#ACE2005EN
python re_agcn_main.py --do_test --task_name ace05en --data_dir ./data/ace05en/ --model_path ./RE_AGCN.ACE05EN.BERT.L/ --do_lower_case
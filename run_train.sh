#!/bin/bash


#SemEval
python re_agcn_main.py --do_train --do_eval --task_name semeval --data_dir ./data/semeval/ --model_path ./bert_model_path --model_name RE_AGCN.SEMEVAL.BERT.L --do_lower_case

#ACE05EN
python re_agcn_main.py --do_train --do_eval --task_name ace05en --data_dir ./data/ace05en/ --model_path ./bert_model_path --model_name RE_AGCN.ACE05EN.BERT.L --do_lower_case

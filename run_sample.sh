#train
python re_agcn_main.py --do_train --do_eval --task_name sample --data_dir ./data/sample_data/ --model_path ./bert_model_path --dep_type local_global_graph --model_name RE_AGCN.SAMPLE.BERT.L --do_lower_case

# test
python re_agcn_main.py --do_test --task_name sample --data_dir ./data/sample_data/ --model_path ./RE_AGCN.SAMPLE.BERT.L/ --do_lower_case
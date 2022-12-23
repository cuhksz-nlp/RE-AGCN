# RE-AGCN

This is the implementation of [Dependency-driven Relation Extraction with Attentive Graph Convolutional Networks](https://aclanthology.org/2021.acl-long.344/) at ACL 2021.

You can e-mail Yuanhe Tian at `yhtian@uw.edu`, if you have any questions.


**Visit our [homepage](https://github.com/synlp/.github) to find more our recent research and softwares for NLP (e.g., pre-trained LM, POS tagging, NER, sentiment analysis, relation extraction, datasets, etc.).**

## Upgrades of RE-AGCN

We are improving our RE-AGCN. For updates, please visit [HERE](https://github.com/synlp/RE-AGCN).

## Citation

If you use or extend our work, please cite our paper at ACL 2021.

```
@inproceedings{tian-etal-2021-dependency,
    title = "Dependency-driven Relation Extraction with Attentive Graph Convolutional Networks",
    author = "Tian, Yuanhe and Chen, Guimin and Song, Yan and Wan, Xiang",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    pages = "4458--4471",
}
```

## Requirements

Our code works with the following environment.
* `python>=3.7`
* `pytorch>=1.3`

## Dataset

To obtain the data, you can go to [`data`](./data) directory for details.

## Downloading BERT 

In our paper, we use BERT ([paper](https://www.aclweb.org/anthology/N19-1423/)) as the encoder.

For BERT, please download pre-trained BERT-Base and BERT-Large English from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

## Downloading our pre-trained RE-AGCN

For RE-AGCN, you can download the models we trained in our experiments from [Google Drive](https://drive.google.com/drive/folders/1HoVc4y8tZNm7h9MorqgIvRJo64qL_0HM?usp=sharing).

## Run on Sample Data

Run `run_sample.sh` to train a model on the small sample data under the `sample_data` directory.

## Training and Testing

You can find the command lines to train and test models in `run_train.sh` and `run_test.sh`, respectively.

Here are some important parameters:

* `--do_train`: train the model.
* `--do_eval`: test the model.

## To-do List

* Regular maintenance.

You can leave comments in the `Issues` section, if you want us to implement any functions.


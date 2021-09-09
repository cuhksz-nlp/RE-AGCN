# RE-AGCN

This is the implementation of [Dependency-driven Relation Extraction with Attentive Graph Convolutional Networks](https://aclanthology.org/2021.acl-long.344.pdf) at ACL 2021.

You can e-mail Guimin Chen at `chenguimin@foxmail.com` or `cuhksz.nlp@gmail.com` or Yuanhe Tian at `yhtian@uw.edu`, if you have any questions.

## Citation

If you use or extend our work, please cite our paper at ACL 2021.

```
@inproceedings{tian2021dependency,
  title={Dependency-driven Relation Extraction with Attentive Graph Convolutional Networks},
  author={Tian, Yuanhe and Chen, Guimin and Song, Yan and Wan, Xiang},
  booktitle={Proceedings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing},
  year={2021}
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


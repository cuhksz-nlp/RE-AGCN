import os
import re
import json
import argparse
from tqdm import tqdm
from corenlp import StanfordCoreNLP

FULL_MODEL = './stanford-corenlp-full-2018-10-05'
punctuation = ['.', ',', ':', '?', '!', '(', ')', '"', '[', ']', ';', '\'']
chunk_pos = ['NP', 'PP', 'VP', 'ADVP', 'SBAR', 'ADJP', 'PRT', 'INTJ', 'CONJP', 'LST']

def change(char):
    if "(" in char:
        char = char.replace("(", "-LRB-")
    if ")" in char:
        char = char.replace(")", "-RRB-")
    return char

def split_punc(s):
    word_list = ''.join([" "+x+" " if x in punctuation else x for x in s]).split()
    return [w for w in word_list if len(w) > 0]

def tokenize(sentence):
    words = []
    for seg in re.split('(<e1>|</e1>|<e2>|</e2>)', sentence):
        if seg in ["<e1>", "</e1>", "<e2>", "</e2>"]:
            words.append(seg)
        else:
            words.extend(split_punc(seg))
    return words

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        datas = []
        for line in lines:
            splits = line.split('\t')
            if len(splits) < 1:
                continue
            e1, e2, label, sentence = splits
            sentence = sentence.strip()

            e11_p = sentence.index("<e1>")  # the start position of entity1
            e12_p = sentence.index("</e1>")  # the end position of entity1
            e21_p = sentence.index("<e2>")  # the start position of entity2
            e22_p = sentence.index("</e2>")  # the end position of entity2

            ori_sentence = tokenize(sentence)
            splits.append(ori_sentence)
            splits.append([s for s in ori_sentence if s not in ["<e1>", "</e1>", "<e2>", "</e2>"]])

            if e1 in sentence[e11_p:e12_p] and e2 in sentence[e21_p:e22_p]:
                datas.append(splits)
            elif e2 in sentence[e11_p:e12_p] and e1 in sentence[e21_p:e22_p]:
                splits[0], splits[1] = e2, e1
                datas.append(splits)
            else:
                print("data format error: {}".format(line))
    return datas

def request_features_from_stanford(data_dir, flag):
    data_path = os.path.join(data_dir, flag + '.tsv')
    print("request_features_from_stanford {}".format(data_path))
    if not os.path.exists(data_path):
        print("{} not exist".format(data_path))
        return
    all_sentences = read_txt(data_path)
    sentences_str = []
    for e1, e2, label, raw_sentence, ori_sentence, sentence in all_sentences:
        sentence = [change(s) for s in sentence]
        sentences_str.append([e1, e2, label, raw_sentence, ori_sentence, sentence])
    all_data = []
    with StanfordCoreNLP(FULL_MODEL, lang='en') as nlp:
        for e1, e2, label, raw_sentence, ori_sentence, sentence in tqdm(sentences_str):
            props = {'timeout': '5000000','annotators': 'pos, parse, depparse', 'tokenize.whitespace': 'true' ,
                     'ssplit.eolonly': 'true', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
            results=nlp.annotate(' '.join(sentence), properties=props)
            results["e1"] = e1
            results["e2"] = e2
            results["label"] = label
            results["raw_sentence"] = raw_sentence
            results["ori_sentence"] = ori_sentence
            results["word"] = sentence
            all_data.append(results)
    assert len(all_data) == len(sentences_str)
    with open(os.path.join(data_dir, flag + '.txt'), 'w', encoding='utf8') as fout_text, \
            open(os.path.join(data_dir, flag + '.txt.dep'), 'w', encoding='utf8') as fout_dep:
        for data in all_data:
            #text
            fout_text.write("{}\t{}\t{}\t{}\n".format(data["e1"], data["e2"], data["label"], " ".join(data["ori_sentence"])))
            #dep
            for dep_info in data["sentences"][0]["basicDependencies"]:
                fout_dep.write("{}\t{}\t{}\n".format(dep_info["governor"], dep_info["dependent"], dep_info["dep"]))
            fout_dep.write("\n")
            assert len(data["sentences"][0]["basicDependencies"])+4 == len(data["ori_sentence"])

def get_labels_dict(data_dir):
    labels_set = set()
    for flag in ["train", "dev", "test"]:
        data_path = os.path.join(data_dir, flag + '.txt')
        if not os.path.exists(data_path):
            continue
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                e1,e2,label,sentence = line.strip().split("\t")
                labels_set.add(label)
    save_path = os.path.join(data_dir, "label.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(list(labels_set), f, ensure_ascii=False)

def get_dep_type_dict(data_dir):
    dep_type_set = set()
    for flag in ["train", "dev", "test"]:
        data_path = os.path.join(data_dir, flag + '.txt.dep')
        if not os.path.exists(data_path):
            continue
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line == "\n":
                    continue
                governor,dependent,dep_type = line.strip().split("\t")
                dep_type_set.add(dep_type)
    save_path = os.path.join(data_dir, "dep_type.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(list(dep_type_set), f, ensure_ascii=False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, type=str, required=True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    for flag in ["train", "dev", "test"]:
        request_features_from_stanford(args.data_path, flag)
    get_labels_dict(args.data_path)
    get_dep_type_dict(args.data_path)
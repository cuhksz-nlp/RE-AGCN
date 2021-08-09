import os
import re
import torch
import numpy as np
from sklearn.metrics import f1_score,classification_report

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def semeval_official_eval(label_map, preds, labels, outdir):
    proposed_answer = os.path.join(outdir, "proposed_answer.txt")
    answer_key  = os.path.join(outdir, "answer_key.txt")
    with open(proposed_answer, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(labels):
            f.write("{}\t{}\n".format(idx, label_map[pred]))
    with open(answer_key, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(idx, label_map[pred]))

    eval_cmd = "perl ./eval/semeval2010_task8_scorer-v1.2.pl {} {}".format(proposed_answer, answer_key)
    print(eval_cmd)
    p,r,f1 = 0,0,0
    try:
        msg = [s for s in os.popen(eval_cmd).read().split("\n") if len(s) > 0]
        b_official = False
        for i,s in enumerate(msg):
            if "(9+1)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL" in s:
                b_official = True
            if b_official is False:
                continue
            if "MACRO-averaged result (excluding Other)" in s and "F1 =" in msg[i+1]:
                p = float(re.findall('P = (.+?)%', msg[i+1])[0])
                r = float(re.findall('R = (.+?)%', msg[i+1])[0])
                f1 = float(re.findall('F1 = (.+?)%', msg[i+1])[0])
                break

    except Exception as e:
        print(str(e))
        f1 = 0
    print("p: {}, r: {}, f1: {}".format(p, r, f1))
    return {
        "precision": p,
        "recall": r,
        "f1": f1
    }

def write_prediction(relation_labels, output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, relation_labels[pred]))

def compute_micro_f1(preds, labels, label_map, ignore_label, output_dir=None):
    if output_dir is not None:
        proposed_answer = os.path.join(output_dir, "proposed_answer.txt")
        answer_key = os.path.join(output_dir, "answer_key.txt")
        with open(proposed_answer, 'w', encoding='utf-8') as f:
            for idx, pred in enumerate(labels):
                f.write("{}\t{}\n".format(idx, pred))
        with open(answer_key, 'w', encoding='utf-8') as f:
            for idx, pred in enumerate(preds):
                f.write("{}\t{}\n".format(idx, pred))

    target_name = []
    target_id = []
    for name,id in label_map.items():
        if name in ignore_label:
            continue
        target_id.append(id)
        target_name.append(name)
    res = classification_report(labels, preds, labels=target_id, target_names=target_name, output_dict=True)
    print(res)
    return res["micro avg"]["f1-score"]

def compute_metrics(preds, labels, rel_size, ignore_label):
    assert len(preds) == len(labels)
    # return acc_and_f1(preds, labels)
    return measure_statistics(preds, labels, rel_size, ignore_label)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average='micro'):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {
        "acc": acc,
        "f1": f1,
    }

def fbeta_score(precision, recall, beta=1.0):
    beta_square = beta * beta
    if (precision != 0.0) and (recall != 0.0):
        res = ((1 + beta_square) * precision * recall / (beta_square * precision + recall))
    else:
        res = 0.0
    return res

def measure_statistics(preds, labels, rel_size, ignore_label):
    """
    Calculate: True Positives (TP), False Positives (FP), False Negatives (FN)
    GPU & CPU code
    """
    y = torch.from_numpy(preds)
    t = torch.from_numpy(labels)

    label_num = torch.as_tensor([rel_size]).long()
    ignore_label = torch.as_tensor([ignore_label]).long()

    mask_t = torch.eq(t, ignore_label)        # true = no_relation
    mask_p = torch.eq(y, ignore_label)        # pred = no_relation

    true = torch.where(mask_t, label_num, t)  # t: ground truth labels (replace ignored with +1)
    pred = torch.where(mask_p, label_num, y)  # y: output of neural network (replace ignored with +1)

    tp_mask = torch.where(torch.eq(pred, true), true, label_num)
    fp_mask = torch.where(torch.ne(pred, true), pred, label_num)  # this includes wrong positive classes as well
    fn_mask = torch.where(torch.ne(pred, true), true, label_num)

    tp = torch.bincount(tp_mask, minlength=rel_size + 1)[:rel_size]
    fp = torch.bincount(fp_mask, minlength=rel_size + 1)[:rel_size]
    fn = torch.bincount(fn_mask, minlength=rel_size + 1)[:rel_size]
    tn = torch.sum(mask_t & mask_p)

    atp = np.sum(tp.numpy())
    afp = np.sum(fp.numpy())
    afn = np.sum(fn.numpy())
    atn = np.sum(tn.numpy())
    micro_p = (1.0 * atp) / (atp + afp) if (atp + afp != 0) else 0.0
    micro_r = (1.0 * atp) / (atp + afn) if (atp + afn != 0) else 0.0
    micro_f = fbeta_score(micro_p, micro_r)

    return {
        "precision": micro_p,
        "recall": micro_r,
        "f1": micro_f,
    }




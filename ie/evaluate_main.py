import scipy.io as sio
import sys
import os
import numpy as np
#import ie.evaluate as ev
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import ie.evaluate as ev
from common.global_config import get_ie_config, get_job_id, get_schema_set, get_train_para, get_test_file_name

def predicate_label_to_id(predicate_label, predicate_label_map):
    predicate_label_map_length = len(predicate_label_map)
    predicate_label_ids = [0] * predicate_label_map_length
    for label in predicate_label:
        predicate_label_ids[predicate_label_map[label]] = 1
    return predicate_label_ids


def evaluate_main(ie_save_path_re, job_id='', model_idx_info=""):
    schema_set = get_schema_set()
    predicate_label = schema_set.split("#")

    label_map = {}
    for (i, label) in enumerate(predicate_label):
        label_map[label] = i

    # -------------

    predicate_out_file = os.path.join(ie_save_path_re, "log", job_id, "temp_data/predicate_classifiction/classification_data/test", 'predicate_out.txt')

    with open(predicate_out_file, 'r', encoding='utf-8') as label_file:
        label_list = label_file.read().splitlines()

    target = []
    for ele in label_list:
        labels = ele.split()
        target.append(predicate_label_to_id(labels, label_map))

    test_label = np.array(target)

    # -------------------
    path = os.path.join(ie_save_path_re, "log", job_id, "temp_data/predicate_infer_out")
    with open(os.path.join(path, 'predicate_predict.txt'), 'r', encoding='utf-8') as pre_file:
        pre_list = pre_file.read().splitlines()

    target = []
    for ele in pre_list:
        labels = ele.split()
        label = []
        for item in labels:
            if item == "LABEL":
                label.append(1)
            else:
                label.append(0)
        target.append(label)

    predict_label = np.array(target)

    predict_value = np.loadtxt(os.path.join(path,'predicate_score_value.txt'), dtype=np.float32)

    HLoss = ev.HammingLoss(predict_label, test_label)
    RLoss = ev.rloss(predict_value, test_label)
    Oerror = ev.OneError(predict_value, test_label)
    coverage = ev.Coverage(predict_value, test_label)
    aprecision = ev.avgprec(predict_value, test_label)
    auc = ev.MacroAveragingAUC(predict_value, test_label)

    MicroPrecision = ev.MicroPrecision(predict_label, test_label)
    MacroPrecision = ev.MacroPrecision(predict_label, test_label)
    MicroRecall = ev.MicroRecall(predict_label, test_label)
    MacroRecall = ev.MacroRecall(predict_label, test_label)
    MicroF1 = ev.MicroF1(predict_label, test_label)
    MacroF1 = ev.MacroF1(predict_label, test_label)

    text_f = open(os.path.join(path, "metric_mll.txt"), "a+",
                  encoding='utf-8')
    text_f.write('model_idx_info: ' + model_idx_info+ "\n")
    text_f.write('[HLoss,RLoss,Oerror,coverage,aprecision,auc]' + "\n")
    text_f.write(str([HLoss, RLoss, Oerror, coverage, aprecision, auc]) + "\n")
    text_f.write('MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, MicroF1, MacroF1'+ "\n")
    text_f.write(str([MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, MicroF1, MacroF1]) + "\n")


if __name__ == '__main__':
    job_id = get_job_id()
    train_data_path = get_ie_config("re")
    evaluate_main(train_data_path, job_id)

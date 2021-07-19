# -*- coding:utf-8 -*-
import os
import os.path
import shutil
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from ie.src.manager_data_for_clf_model import manager_data_for_classification_test
from ie.src.run_predicate_classification_plcp_test import predicate_classification_model_test
from ie.evaluate_main import evaluate_main
from common.global_config import get_ie_config, get_job_id, get_schema_set, get_test_file_name

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    print("mkdir path = {} ".format(path))
    if not isExists:
        print("path is not exist, create...")
        os.makedirs(path)
        return True
    else:
        print("path is exist, rm then create...")
        shutil.rmtree(path)
        os.makedirs(path)
        return True


def re_step(job_id='None', schema_set='None'):
    if job_id == 'None':
        print("job_id is none, re_step end...")
        return 0
    print(" re_step start...")
    ie_save_path_re = get_ie_config("re")
    train_data_path = ie_save_path_re
    clf_model_path = os.path.join(train_data_path, "model/predicate_infer_out_plp_0115_1v2/")
    lists = os.listdir(clf_model_path)
    lists.sort(key=lambda fn: os.path.getmtime(clf_model_path + "/" + fn))
    clf_model_idx = ""
    model_list = []
    for i in range(1, len(lists) + 1):
        file = lists[-i]
        if file.startswith("model.ckpt") and file.endswith(".meta"):
            clf_model_idx = file.replace(".meta", "")
            #break
            model_list.append(clf_model_idx)
    print(model_list)
    model_list =['model.ckpt-164000']
    
    for clf_model_idx in model_list:
        print("clf_model_idx = {} ".format(clf_model_idx))
        path_re = os.path.join(ie_save_path_re, "log", job_id)
        mkdir(path_re)

        schema_set_set = get_schema_set()
        test_data_file = get_test_file_name()
        raw_file_path = os.path.join(ie_save_path_re, "input", test_data_file)
        raw_file_path_log = os.path.join(path_re, test_data_file)
        shutil.copy(raw_file_path, raw_file_path_log)
        if (not os.path.exists(raw_file_path)) or (not os.path.exists(raw_file_path_log)):
            print("ERR, raw_file not exists...")
            exit(0)

        manager_data_for_classification_test()
        path = os.path.join(path_re, "temp_data/predicate_classifiction/classification_data/test/text.txt")
        if not os.path.exists(path):
            print("ERR, manager_data_for_classification failed...")
            exit(0)
        predicate_classification_model_test(job_id, schema_set_set, train_data_path, clf_model_idx)
        path = os.path.join(path_re, "temp_data/predicate_infer_out/predicate_predict.txt")
        if not os.path.exists(path):
            print("ERR, predicate_classification failed...")
            exit(0)

        evaluate_main(ie_save_path_re, job_id)

    return 0


def re_process(job_id='None', schema_set='None'):
    return re_step(job_id, schema_set)


if __name__ == '__main__':
    job_id = get_job_id()
    train_data_path = get_ie_config("re")
    schema_set = get_schema_set()
    re_process(job_id, schema_set)

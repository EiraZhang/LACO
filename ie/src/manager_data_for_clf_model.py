# -*- coding:utf-8 -*-
import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import ie.src.bert.tokenization as tokenization
from common.global_config import get_ie_config, get_job_id, get_train_file_name, get_schema_set, \
    get_test_file_name, get_schema_dict, get_pretrain_model_path

class Model_data_preparation(object):
    def __init__(self, RAW_DATA_INPUT_DIR="", RAW_DATA_FILENAME="",
                 DATA_OUTPUT_DIR="predicate_classifiction/classfication_data",
                 vocab_file_path="vocab.txt", do_lower_case=True, is_train_Mode=False):
        self.bert_tokenizer = tokenization.FullTokenizer(vocab_file=self.get_vocab_file_path(vocab_file_path),
                    do_lower_case=do_lower_case)
        self.DATA_INPUT_DIR = RAW_DATA_INPUT_DIR
        self.DATA_INPUT_FILE = RAW_DATA_FILENAME
        self.DATA_OUTPUT_DIR = DATA_OUTPUT_DIR
        self.is_train_Mode = is_train_Mode

    def get_vocab_file_path(self, vocab_file_path):
        pretrain_model_path = get_pretrain_model_path()
        vocab_file_path = os.path.join(pretrain_model_path,  "cased_L-12_H-768_A-12", vocab_file_path)
        print("vocab_file_path",vocab_file_path)
        return vocab_file_path

    def separate_raw_data_and_token_labeling(self):
        if not os.path.exists(self.DATA_OUTPUT_DIR):
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "train"))
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "valid"))
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "test"))
        print('DATA_OUTPUT_DIR',self.DATA_OUTPUT_DIR)
        file_set_type_list = ["train", "valid", "test"]
        if not self.is_train_Mode:
            file_set_type_list = ["test"]
        for file_set_type in file_set_type_list:
            if file_set_type in ["train", "valid", "test"]:
                try:
                    predicate_out_f = open(
                    os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "predicate_out.txt"), "w",
                    encoding='utf-8')
                except:
                    print('cant write predicate_out.txt')
            text_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "text.txt"), "w",
                          encoding='utf-8')
            token_in_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "token_in.txt"), "w",
                              encoding='utf-8')
            token_in_not_UNK_f = open(
                os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "token_in_not_UNK.txt"), "w",
                encoding='utf-8')

            def predicate_to_predicate_file(spo_list):
                schema_dict = get_schema_dict()
                predicate_list = []
                for spo in spo_list:
                    predicate = spo['predicate']
                    object_type = spo["object_type"]
                    for k, v in object_type.items():
                        key = predicate + "|" + v
                        pred_item = schema_dict[key]
                        if pred_item in predicate_list:
                            continue
                        predicate_list.append(pred_item)
                predicate_list_str = " ".join(predicate_list)
                predicate_out_f.write(predicate_list_str + "\n")

            if file_set_type == "train":
                path_to_raw_data_file = "train_data.json"
            elif  file_set_type == "valid":
                path_to_raw_data_file = "test_data.json"
            else:
                path_to_raw_data_file = "test_data.json"

            with open(os.path.join(self.DATA_INPUT_DIR, path_to_raw_data_file), 'r', encoding='utf-8') as f:
                while True:
                    line = f.readline()
                    if line:
                        r = json.loads(line)
                        spo_list = r["spo_list"]
                        text = r["text"]
                        text_tokened = self.bert_tokenizer.tokenize(text)
                        text_tokened_not_UNK = self.bert_tokenizer.tokenize_not_UNK(text)
                        text_f.write(text + "\n")
                        token_in_f.write(" ".join(text_tokened) + "\n")
                        token_in_not_UNK_f.write(" ".join(text_tokened_not_UNK) + "\n")
                        if file_set_type in ["train", "valid", "test"]:
                            predicate_to_predicate_file(spo_list)
                    else:
                        break
            text_f.close()
            token_in_f.close()
            token_in_not_UNK_f.close()

def manager_data_for_classification_test():
    job_id = get_job_id()
    ie_save_path_re = get_ie_config("re")
    path_re = os.path.join(ie_save_path_re,"log", job_id)
    raw_data_filename = get_test_file_name()
    data_output_dir = os.path.join(path_re, "temp_data", "predicate_classifiction/classification_data")
    model_data = Model_data_preparation(
        RAW_DATA_INPUT_DIR=path_re, RAW_DATA_FILENAME=raw_data_filename, DATA_OUTPUT_DIR=data_output_dir,
        is_train_Mode=False)
    model_data.separate_raw_data_and_token_labeling()

def manager_data_for_classification_train(job_id='None', schema_set='None', train_data_path = 'None'):
    raw_data_filename = get_train_file_name()
    print('raw_data_filename',raw_data_filename)
    data_output_dir= os.path.join(train_data_path, "temp_data", "predicate_classifiction/classification_data")
    print('data_output_dir',data_output_dir)
    model_data = Model_data_preparation(
        RAW_DATA_INPUT_DIR=train_data_path+"/input", RAW_DATA_FILENAME=raw_data_filename, DATA_OUTPUT_DIR=data_output_dir,
        is_train_Mode=True)
    model_data.separate_raw_data_and_token_labeling()


if __name__ == "__main__":
    job_id = get_job_id()
    schema_set = get_schema_set()
    train_data_path = get_ie_config("re")
    manager_data_for_classification_train(job_id, schema_set, train_data_path)

# -*- coding:utf-8 -*-
import os
import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from ie.src.manager_data_for_clf_model import manager_data_for_classification_train

from ie.src.run_predicate_classification_clcp import predicate_classification_model_train
from common.global_config import get_ie_config, get_job_id, get_schema_set, get_train_para



def ot_clf_process(job_id='None', schema_set='None', train_data_path='None', train_para ='None'):
    
    if True:
        path =train_data_path
        if not os.path.exists(path):
            print("train data not exists, failed...")
            return
    
        '''step1: init train data for model'''
        manager_data_for_classification_train(job_id, schema_set, train_data_path)
        path = os.path.join(train_data_path, "temp_data/predicate_classifiction/classification_data/train/text.txt")
        
        if not os.path.exists(path):
            print("manager_data_for_classification failed...")
            return

        '''step2: train  model'''
        predicate_classification_model_train(job_id, schema_set, train_data_path, train_para)
        return 0



if __name__ == '__main__':
    job_id = get_job_id()
    train_data_path = get_ie_config("re")
    print('train_data_path ',train_data_path )
    schema_set = get_schema_set()
    print("schema_set ",schema_set )
    train_para = get_train_para()
    print('train_para',train_para)
    ot_clf_process(job_id, schema_set, train_data_path, train_para)

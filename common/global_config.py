#!/usr/bin/env python
# -*- coding: utf-8 -*-

def get_ie_config(methold):
    if methold == "re":
        path = "../log/re_model"
        return path

def get_schema_set():
    schema_set = "label0#label1#label2#label3#label4#label5#label6#label7#label8#label9#label10#label11#label12#label13#label14#label15#label16#label17#label18#label19#label20#label21#label22#label23#label24#label25#label26#label27#label28#label29#label30#label31#label32#label33#label34#label35#label36#label37#label38#label39#label40#label41#label42#label43#label44#label45#label46#label47#label48#label49#label50#label51#label52#label53"
    return schema_set

def get_schema_dict():
    schema_dict = {'label0|Text': 'label0', 'label1|Text': 'label1', 'label2|Text': 'label2', 'label3|Text': 'label3', 'label4|Text': 'label4',
                   'label5|Text': 'label5', 'label6|Text': 'label6', 'label7|Text': 'label7', 'label8|Text': 'label8', 'label9|Text': 'label9',
                   'label10|Text': 'label10', 'label11|Text': 'label11', 'label12|Text': 'label12', 'label13|Text': 'label13',
                   'label14|Text': 'label14', 'label15|Text': 'label15', 'label16|Text': 'label16', 'label17|Text': 'label17',
                   'label18|Text': 'label18', 'label19|Text': 'label19', 'label20|Text': 'label20', 'label21|Text': 'label21',
                   'label22|Text': 'label22', 'label23|Text': 'label23', 'label24|Text': 'label24', 'label25|Text': 'label25',
                   'label26|Text': 'label26', 'label27|Text': 'label27', 'label28|Text': 'label28', 'label29|Text': 'label29',
                   'label30|Text': 'label30', 'label31|Text': 'label31', 'label32|Text': 'label32', 'label33|Text': 'label33',
                   'label34|Text': 'label34', 'label35|Text': 'label35', 'label36|Text': 'label36', 'label37|Text': 'label37',
                   'label38|Text': 'label38', 'label39|Text': 'label39', 'label40|Text': 'label40', 'label41|Text': 'label41',
                   'label42|Text': 'label42', 'label43|Text': 'label43', 'label44|Text': 'label44', 'label45|Text': 'label45',
                   'label46|Text': 'label46', 'label47|Text': 'label47', 'label48|Text': 'label48', 'label49|Text': 'label49',
                   'label50|Text': 'label50', 'label51|Text': 'label51', 'label52|Text': 'label52', 'label53|Text': 'label53'}

    return schema_dict

def get_pretrain_model_path():
    pretrain_model_path = "../pretrained_model"
    return pretrain_model_path

def get_train_para():
    train_para = {
        "para_batch_size":32,
        "para_epochs":100,
        "para_max_len":320,
        "para_learning_rate":5e-5,
        }
    return train_para

def get_job_id():
    job_id = "20210710"
    return job_id

def get_train_file_name():
    train_file_name = "train_data.json"
    return train_file_name

def get_test_file_name():
    test_file_name = "test_data.json"
    return test_file_name

def get_dev_file_name():
    test_file_name = "dev_data.json"
    return test_file_name

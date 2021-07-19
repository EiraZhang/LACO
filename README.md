#Enhancing Label Correlation Feedback in Multi-Label Text Classification via Multi-Task Learning
This repository contains the code for the ACL 2021 paper
>["Enhancing Label Correlation Feedback in Multi-Label Text Classification via Multi-Task Learning"](https://arxiv.org/abs/2106.03103).

If you use LACO in your work, please cite it as follows:
``` bibtex
@article{zhang2021enhancing,
  title={Enhancing Label Correlation Feedback in Multi-Label Text Classification via Multi-Task Learning},
  author={Zhang, Ximing and Zhang, Qian-Wen and Yan, Zhao and Liu, Ruifang and Cao, Yunbo},
  journal={arXiv preprint arXiv:2106.03103},
  year={2021}
}
```
##Settings
Environment Requirements

- python 3.6+

- Tensorflow 1.12.0+

Environmental preparation

- You can change the experimental settings in LACO/common/global_config.py

- The initial content under directory LACO/ie/src/bert is primarily from [Google bert](https://github.com/google-research/bert). Citation information is recorded in the corresponding file. You can download and unzip it at LACO/pretrained_model/ .

##Datasets
- [AAPD](https://git.uwaterloo.ca/jimmylin/Castor-data/tree/master/datasets/AAPD)
- [RCV1-V2](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm)

Data Preparation

The sample data are in the directory LACO/log/re_model/input. 
Note that the "text" field stores the text content, the "spo_list" field stores the relevant labels in "predicate", and the other fields can be ignored.

##How To Run
-  Train_mltc_with_plcp:    python ie/train_main_plcp.py
-  Test_mltc_with_plcp:     python ie/test_main_plcp.py
-  Train_mltc_with_clcp:    python ie/train_main_clcp.py
-  Test_mltc_with_clcp:     python ie/test_main_clcp.py

##Results
The best model of +PLCP of AAPD dataset and and RCV1V2 dataset can be found at https://share.weiyun.com/5EXHqEPE (password: 8yrgji) for your reference.

##© Copyright
	Ximing Zhang (ximingzhang@bupt.edu.cn),
	Qian-Wen Zhang (cowenzhang@tencent.com),
	Zhao Yan (zhaoyan@tencent.com),
	Ruifang Liu (lrf@bupt.edu.cn),
	Yunbo Cao (yunbocao@tencent.com),
	Tencent Cloud Xiaowei, Beijing, China  && Beijing University of Posts and Telecommunications, Beijing, China  

This code package can be used freely for academic, non-profit purposes. For other usage, 
please contact us for further information (Ximing Zhang: ximingzhang@bupt.edu.cn).
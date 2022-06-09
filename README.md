# SG-BERT

This repository contains the implementation of ''Self-Gudied Contrastive Learning for BERT Sentence Representations (ACL 2021)''.
(Disclaimer: the code is a little bit cluttered as this is not a cleaned version.)
 
When using this code for the following work, please cite our paper with the BibTex below.

	@inproceedings{kim-etal-2021-self,
    title = "Self-Guided Contrastive Learning for {BERT} Sentence Representations",
    author = "Kim, Taeuk  and
      Yoo, Kang Min  and
      Lee, Sang-goo",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.197",
    doi = "10.18653/v1/2021.acl-long.197",
    pages = "2528--2540",
    abstract = "Although BERT and its variants have reshaped the NLP landscape, it still remains unclear how best to derive sentence embeddings from such pre-trained Transformers. In this work, we propose a contrastive learning method that utilizes self-guidance for improving the quality of BERT sentence representations. Our method fine-tunes BERT in a self-supervised fashion, does not rely on data augmentation, and enables the usual [CLS] token embeddings to function as sentence vectors. Moreover, we redesign the contrastive learning objective (NT-Xent) and apply it to sentence representation learning. We demonstrate with extensive experiments that our approach is more effective than competitive baselines on diverse sentence-related tasks. We also show it is efficient at inference and robust to domain shifts.",}



## Pre-requisite Python Libraries

Please install the following libraries specified in the **requirements.txt** first before running our code.

    certifi==2022.5.18.1
    charset-normalizer==2.0.12
    click==8.1.3
    dataclasses==0.6
    dill==0.3.5.1
    filelock==3.7.1
    future==0.18.2
    idna==3.3
    importlib-metadata==4.11.4
    joblib==1.1.0
    nltk==3.7
    numpy==1.21.6
    packaging==21.3
    protobuf==3.20.0
    pyparsing==3.0.9
    regex==2022.6.2
    requests==2.27.1
    sacremoses==0.0.53
    scikit-learn==1.0.2
    scipy==1.7.3
    sentence-transformers==0.3.9
    sentencepiece==0.1.91
    six==1.16.0
    threadpoolctl==3.1.0
    tokenizers==0.9.3
    torch==1.7.0
    tqdm==4.64.0
    transformers==3.5.1
    typing_extensions==4.2.0
    urllib3==1.26.9
    zipp==3.8.0


## How to Run Code

> python training.py 
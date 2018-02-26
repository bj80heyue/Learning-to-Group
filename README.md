# Merge or Not? Learning to Group Faces via Imitation Learning
### Yue He, Kaidi Cao, Cheng Li, Chen Change Loy
Code for the AAAI 2018 publication "Merge or Not? Learning to Group Faces via Imitation Learning".You can read a preprint on [Arxiv](https://arxiv.org/abs/1707.03986)

### Abstract
Given a large number of unlabeled face images, face grouping aims at clustering the images into individual identities present in the data. This task remains a challenging problem despite the remarkable capability of deep learning approaches in learning face representation. In particular, grouping results can still be egregious given profile faces and a large number of uninteresting faces and noisy detections. Often, a user needs to correct the erroneous grouping manually. In this study, we formulate a novel face grouping framework that learns clustering strategy from ground-truth simulated behavior. This is achieved through imitation learning (a.k.a apprenticeship learning or learning by watching) via inverse reinforcement learning (IRL). In contrast to existing clustering approaches that group instances by similarity, our framework makes sequential decision to dynamically decide when to merge two face instances/groups driven by short- and long-term rewards. Extensive experiments on three benchmark datasets show that our framework outperforms unsupervised and supervised baselines. 

### Dataset - GFW(Group Face in the Wild)
You can download the dataset from [here](https://www.dropbox.com/s/aktxy4phqaevmr7/GFW_RELEASE.tar?dl=0)
In the main folder,each subfolder represents an album.
An album contains number of identities person(id from 3 to N).
Specially,"id = 1" means passerby(apperaed only once in the album),
"id = 2" means low-quality faces which cannot be recognize as a normal human face.



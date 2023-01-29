## KBGNN

This is the pytorch implementation for our WSDM 2023 [paper](https://arxiv.org/abs/2210.16591):
> Yifang Qin, Yifan Wang, Fang Sun, Wei Ju, Xuyang Hou, Zhe Wang, Jia Cheng, Jun Lei and Ming Zhang(2022). 
> DisenPOI: Disentangling Sequential and Geographical Influence for Point-of-Interest Recommendation

In this paper, we propose DisenPOI, a novel Disentangled dual-graph framework for POI recommendation.
DisenPOI jointly utilizes sequential and geographical relationships on two separate graphs and disentangles
the two influences with self-supervision.

### Environment Requirement

The code has been tested running under Python 3.8.13. The required packages are as follows:

- pytorch == 1.11.0
- torch_geometric == 2.0.4
- pandas == 1.4.1
- sklearn == 0.23.2

Please cite our paper if you use the code.


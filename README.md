# GazeCommunication

ETRA work (under Construction)
- Overleaf link [ETRA 2023 paper under construction](https://www.overleaf.com/8542516856cjphkgqhqzvt)

## Intuition

**Following Gaze and Recognizing Patterns in Interpersonal Communication**

Analyzing gaze behaviors is crucial to interpret the nature of communication. But current studies of gaze patterns just focus on detecting a single specific pattern such as finding the Looking-At-Each Other pattern or shared attention pattern. It would be nice if we can have a framework to find all these patterns at once. Also, current studies on finding gaze patterns do not make use of the development in gaze estimation and gaze follow. One intuitive thought is that if the model can correctly predicts the spatial coordinates of a gaze point, it won't take much effort to learn and classify gaze patterns. 

Thus, in this work, we propose a novel framework that tackles the task of gaze pattern recognition. We propose a new dataset containing videos of interpersonal interaction scenes with gaze pattern annotations including both people looking at each other and share attention.

## Proposed Contribution

1. We propose a novel **what the method looks like?** framework for **recognizing gaze patterns with psychological implications during interpersonal communication**.
2. We introduce a new dataset **dataset information** 
3. 

## Progress


- [x] Dataloader for gazefollow 
- [x] Feature extractors: (head feature extractor: ResNet34, scene feature extractor: ResNet50)
- [x] Basic GazeFollow Model (Finetune on GazeFollow)

**Note by 12/11**
- Comparison on gazefollow dataset

| Method                     | AUC   | Avg Dist | Min Dist |
|----------------------------|-------|----------|----------|
| **Ours**                   | **0.861** | **0.210**    | **0.142**    |
| SOTA(CVPR 2021 with Depth) | 0.922 | 0.124    | 0.067    |
| Chong(CVPR 2020 Chong)     | 0.860 | 0.137    | 0.077    |

Some parameters:
1. Best performance epoch: 25
2. randoms seed = 2022
3. initial learning rate = 2.5*e-4

- Refractor the simple extractor to the general gaze pattern recognition tasks:
- [ ] AVA-LEAO. 
- [ ] UCO-LEAO 
- [ ] OIMG 
- [ ] ShareAttention

- Experiments on our dataset
- [ ] Dataloader for our dataset
- [ ] Refractor to our dataset
- [ ] Train on our dataset

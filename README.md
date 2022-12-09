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

[ ] Dataloader for gazefollow
[ ] Dataloader for our dataset
[ ] Feature extractors: (head feature extractor: ResNet34, scene feature extractor: ResNet50)
[ ] Basic GazeFollow Model (Finetune on GazeFollow)
[ ] Refractor to our dataset
[ ] Train on our dataset

# Gaze Pattern Recognition in Dyadic Communication

This is the repo for work Gaze Pattern Recognition in Dyadic Communication at ETRA 2023. 

## Overview
Analyzing gaze behaviors is crucial to interpret the nature of communication. Current studies on gaze have focused primarily on the detection of a single pattern, such as the Looking-At-Each-Other pattern or the shared attention pattern. In this work, we re-define five static gaze patterns that cover all the status during a dyadic communication and propose a network to recognize these mutual exclusive gaze patterns given an image. We annotate a benchmark, called GP-Static, for the gaze pattern recognition task, on which our method experimentally outperforms other alternate solutions. Our method also achieves the state-of-art performance on other two single gaze pattern recognition tasks. The analysis of gaze patterns on preschool children demonstrates that the statistic of the proposed static gaze patterns conforms with the findings in psychology.

## Method Overview
![figure](method.png)

## 
##  🕐 Evaluation on Gaze Pattern Classification
### Dataset
To obtain the dataset, please fill this [form](https://forms.gle/Qhx2M3KGf4WEN2xX8). 
The videos and annotations will be send to the provided email within 1-2 weeks.

### Implementation [TODO]
Run: [command wil be updated soon] to obtain the model's performance on the benchmark dataset.


##  🕐 Evaluation on Single Gaze Pattern Detection [TODO]
### Dataset
To demonstrate the applicability of our method to previously defined tasks of gaze pattern recognition at image level, we evaluate the performance of our model on two single gaze detection tasks: detecting the **mutual** gaze pattern and the **shared attention** pattern.

For the evaluation on detecting mutual gaze (people looking-at-each-other), we use [UCO-LAEO](https://www.robots.ox.ac.uk/~vgg/research/laeonet/main_cvpr2019.html), [AVA-LAEO](https://www.robots.ox.ac.uk/~vgg/research/laeonet/main_cvpr2019.html) and [OI-MG](https://research.google/resources/datasets/google-open-images-mutual-gaze-dataset/).

For the evaluation on detecting the shared attention, we use [VideoCoAtt](http://www.stat.ucla.edu/~lifengfan/shared_attention). 

\* *For downloading these datasets, please refer to the dataset page by clicking on each each dataset*

### Implementation [TODO]

Run: [command wil be updated soon] to obtain the model's performance on the each dataset.

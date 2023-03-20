# Gaze Pattern Recognition in Dyadic Communication

ETRA work (under Construction)

:clock1: Overleaf link [ETRA 2023 paper under construction](https://www.overleaf.com/8542516856cjphkgqhqzvt)

## Intuition

**Gaze Pattern Recognition in Dyadic Communication**

Analyzing gaze behaviors is crucial to interpret the nature of communication. Current studies on gaze have focused primarily on the detection of a single pattern, such as the Looking-At-Each-Other pattern or the shared attention pattern. In this work, we re-define five static gaze patterns that cover all the status during a dyadic communication and propose an end-to-end network to recognize these mutual exclusive gaze patterns given a static image. We annotate a benchmark, namely GP-Static, for the gaze pattern recognition task, on which our model experimentally outperforms other alternate solutions. On other two single gaze pattern tasks, our model also achieves the state-of-art performance. Gaze pattern analysis on preschool children demonstrates that the statistic of the proposed classification of the static gaze patterns conforms with the findings in psychology.

# Method
![figure](method.png)

## Progress

### 🕐 Basic GazeFollow Model (Pretrain on GazeFollow)
### 🕐 Training and Performance on our dataset


### 🕐 Refractor the GazeFollow model to the general Gaze Pattern Recognition Tasks


| Dataset                                                          | UCO-LAEO | AVA-LAEO              | OI-MG                 | Shared Attention      |
|------------------------------------------------------------------|----------|-----------------------|-----------------------|-----------------------|
| Ours                                                             |**75.02**  | **82.52** | **72.1** | :white_medium_square: |
| LAEONet reported                                                 | **79.5** | 50.6                  | -                     | -                     |
| LAEONet Sinlge Frame (reported in *3)                            | 55.9     | 70.2                  | 59.8                  | -                     |
| Pseudo3DGaze(2021AAAI)                                        | 65.1     | **72.2**              | **70.1**              | -                     |
| Gaze+RP+LSTM (Inferring Shared Attention in Social Scene Videos) | -        | -                     | -                     | **71.4**              |


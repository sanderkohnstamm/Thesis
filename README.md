# Thesis
Invariant Domain Adaptation graduation project for TNO: Intelligent Imaging

##Instructions
create data folder next to thesis

cd data

git clone https://github.com/MachineLearning2020/Homework3-PACS

## Abstract

The goal of this thesis is to study and improve a deep neural networks robustness to a shifting domain. In computer vision and object recognition, domains can represent many different groups of images, such as images made at a specific time of day, in a certain style, with a certain kind of weather or from a specific angle. Deep neural networks often struggle when the domain changes between training and testing data. Improving out-of-domain performance within image recognition, improves a neural network’s efficiency, application and robustness. We have found that aligning feature representation embeddings for images from different domains can learn a network to be more robust to domain shifts, resulting in higher accuracies when testing on a new domain. We propose an algorithm for a three domain setting, two of which are labelled. By minimising the element-wise distance between the feature embeddings of two training domains we improve the accuracy on the third target domain. In this thesis the three domains are represented by three distinct styles of image. We test our proposal in two versions. One version does not use the target domains data, and the other does, but without its labels. These respective forms are called domain generalisation and unsupervised domain adaptation. We provide an analysis of the performance of our proposal on these two different forms and its domain shifting implications. We achieve a small performance increase for the unsupervised domain adaptation setting and learn the importance of distinctive information in both these situations.

## Overview figure
![algo](https://user-images.githubusercontent.com/25148544/193779452-4f0b8159-ea05-484e-b3b2-fd94b9134379.jpg)
Overview of the feature alignment on three domains by introducing pseudo-labels in the unlabelled third target domain. This is proposed in this thesis and its resulting target domain embeddings are shown. The resulting embeddings after feature alignment on the bottom right show better and more domain invariant class structures and more with our method.

## Result highlights

![Screenshot 2022-10-04 at 11 03 02](https://user-images.githubusercontent.com/25148544/193779791-77ccdd44-bf1e-4f56-9872-894e630efbab.jpg)

![Screenshot 2022-10-04 at 11 03 22](https://user-images.githubusercontent.com/25148544/193779874-fb203d3e-6eee-4c6e-ae9b-12fd1cefd9f2.jpg)

## Conclusion

In this thesis, we have empirically shown that feature alignment positively im- pacts domain invariant performance in a unsupervised domain adaptation setting when cor- rectly scaled. However, for domain generalisation this impact is only marginal with the correct starting accuracy. We have also introduced a novel three-stage algorithm for unsupervised do- main adaptation using feature alignment. Moreover, we have conducted an ablation study on different distance measures for feature alignment and their respective optimal hyperparame- ters, revealing the balance of a network between domain invariance and distinctive ability, or empirical and invariant risk minimisation. From these studies we have also learned that for unsupervised domain adaptation a second pass of training with pseudo-labels for the target domain is often beneficial, even more so with a correct selection of pseudo-labels, embedding alignment and a lower base accuracy.

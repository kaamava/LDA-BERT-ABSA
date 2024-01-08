## LDA-BERT-ABSA
This repository contains part of the code and pre-trained models for our paper "An LDA Model Augmented with BERT for Aspect Level Sentiment Analysis", which has been submitted to ICJAI2024. The complete code will be released right after the conference announces the acceptance results.

## Contents
- Abstract
- Overview
- Model
- Train
- Results

## Abstract
Aspect level sentiment analysis is a fine-grained task in affective analysis. It extracts aspects and their corresponding emotional polarities from opinionated texts. The first sub task of identifying the aspects with comments is called aspect extraction, which is the focus of the work. Social media platform is a huge untagged data resource. However, data annotation for fine-grained tasks is very expensive and laborious. Therefore, the unsupervised model is highly appreciated. The proposed model is an unsupervised aspect extraction method, a guided potential Dirichlet assignment (LDA) model, which uses the smallest aspect seed words from each aspect category to guide the model to identify hidden topics of interest to users. The LDA model is enhanced by using regular expressions that guide input based on language rules. The model is further enhanced through a variety of filtering strategies, including a BERT-based semantic filter, which integrates semantics to strengthen the situation where co-occurrence statistics may not be able to be used as a distinguishing factor. The threshold values of these semantic filters are estimated using Particle Swarm Optimization strategy. The proposed models are expected to overcome the shortcomings of the basic LDA models, which cannot distinguish overlapping topics representing each aspect category.

## Overview
·Proposed a guided LDA model, which uses only a few general seed words from each aspect category, and combines with the automatic seed set expansion module based on BERT similarity to achieve better and faster topic convergence. The supervision required by the model is the minimum seed aspect terms from each aspect category, which can be learned from the general knowledge of the
domain.

·Proposed specially designed regular expression (RE) based language rules, which helps to better target the multi word aspect. The inputs guiding the LDA model are filtered using multiple pruning strategies, mainly including BERT based semantic filters, so as to combine semantic strength when co-occurrence statistics may not be used as a differentiating factor.

·Proposed particle swarm optimization (PSO) strategy which is used to adjust the threshold parameters of seed set expansion and RE-based input filter.
![image](https://github.com/kaamava/LDA-BERT-ABSA/assets/106901273/98e7b9ad-d455-41b8-b42d-17b45d2c2f52)

## Model
### • BERT

[Bidirectional Encoder Representations from Transformers](https://github.com/google-research/bert) derives representations of words based on nlp models ran over open-source Wikipedia data. These representations are then leveraged to derive corpus topics.


<a id="lda"></a>

### • LDA 
[Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. In the case of kwx, documents or text entries are posited to be a mixture of a given number of topics, and the presence of each word in a text body comes from its relation to these derived topics.

Although not as computationally robust as some machine learning models, LDA provides quick results that are suitable for many applications. Specifically for keyword extraction, in most settings the results are similar to those of BERT in a fraction of the time.

### • PSO
<a target="_blank" href="https://en.wikipedia.org/wiki/Particle_swarm_optimization">Particle swarm optimization</a> (PSO) can be used to find optimal values for the parameters by using a randomly chosen aspect level labeled dataset.
## Train
## Results
*These two parts will be released after the conference announces the acceptance results.*

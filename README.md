# deep-learning-forest-satellite-images

This project focuses on using deep learning to analyse high-resolution forest satellite images for sustainability purposes. The aim is to accurately segment and count trees and predict their height to estimate above-ground biomass (AGB) for carbon storage calculations.

The project builds on existing research by [Li et al. (2023)](https://academic.oup.com/pnasnexus/article/2/4/pgad076/7073732?login=false) and uses a U-net architecture for model development. It uses higher resolution data and Google Cloud tools for improved scalability and efficiency. 

My contribution presented here was the development of a tree height prediction model using Huber loss, which showed superior performance in absolute error reduction compared to other loss functions. The project was successfully completed, but the code repository is currently private. I will share a small sample of code here, the loss functions file contains the Huber loss implemented with a delta of 30cm - the inflection point where we switch from a quadratic to a linear penalty.

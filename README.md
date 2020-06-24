# Boundary-Detection

# Introduction

Boundary detection and image classification are two well
known computer vision problems. The challenging part of
the boundary detection is when we are doing the boundary
detection from a single image. Is this case we cannot use most
of the recent deep learning methods. In this project I use the most recent
pb (probability of boundary) boundary detection algorithm and compare it to [Canny](https://ieeexplore.ieee.org/document/4767851) and [Sobel](https://en.wikipedia.org/wiki/Sobel_operator) baselines. The following shows the overall baseline.

![Repo List](fig1.png)

## Filter bank implementation
I’m implementing three kind of filter banks; oriented deriva-
tive of Gaussian (DoG) filter, Leung-Malik filter, and Gabor
filter.


## Texton, brightness, color map computation
In the next step, we implement texton, brightness, and color
map. I generate texton map by convolving the previous filter
banks with the images and map them into 64 clusters and
taking the average. The concept of the brightness map is as
simple as capturing the brightness changes in the image and
cluster it into 16 groups. Also, The concept of the color map
is to capture the color changes. Then cluster it into 16 clusters.
Illustration of the generated map are shown in following figures.

![Repo List](fig2.png)





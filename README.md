# Image-reconstruction-with-pieces

Hi everyone, this code was an assigment for Image Processing class. The objective was to be able to take little pieces of a large image that have
some overlapping with eachother and reconstruct the full image. 

Rapidly explaining how it works. First we take one of the images and find the ones that overlap with that one vertically and horizontally, with that we determine how
much overlap the images had. After that we calculate how many pieces will the full image have horizontally and vertically, taking into consideration the overlap and the size
of each piece and the full image. Then we create a matrix where it will be saving the position of each piece when its found. After that, with the starting point image,
we compare the overlaps to find each consecutive image until it completes the matrix. In the end, it just reconstruct the full image merging the pixels from each piece in the
respective order and considering the overlap.

This project was made using OpenCV.

The current version of the code is written in Spanish.

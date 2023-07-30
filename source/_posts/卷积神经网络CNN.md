---
title: 卷积神经网络
data: 2023-07-30
update: 2023-07-30
tags: 深度学习
categories: 深度学习
keywords: 深度学习
cover: img/1.jpg
---



# 卷积神经网络CNN

<img src="https://upload-images.jianshu.io/upload_images/145616-6623cc06ea526763.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp" alt="img" style="zoom:50%;" />



在计算机中图像由像素点组成，通过逐一对照来进行图像匹配的话就如上图所示，the result is so unreasonable，我们希望，对于那些仅仅只是做了一些像平移，缩放，旋转，微变形等简单变换的图像，计算机仍然能够识别出图中的"X"和"O"。就像下面这些情况，我们希望计算机依然能够很快并且很准的识别出来：

<img src="https://upload-images.jianshu.io/upload_images/145616-3f48c6e95ae88138.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp" alt="img" style="zoom:50%;" />

这也就是CNN需要做的事情  

<img src="https://upload-images.jianshu.io/upload_images/145616-98c20551ed2a378b.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp" alt="img" style="zoom:50%;" />

对于CNN，他是一块一块逐一比较的，这个比较的小块我们称之为Feature（特征），在两幅图中大致相同的位置找到一些粗糙的特征进行匹配，CNN能够看到更好的相似性。  

每一个feature就像是一个小图（就是一个比较小的有值的二维数组）。不同的Feature匹配图像中不同的特征。在字母"X"的例子中，那些由对角线和交叉线组成的features基本上能够识别出大多数"X"所具有的重要特征。

<img src="https://upload-images.jianshu.io/upload_images/145616-cfca7ae2d5c9034c.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp" alt="img" style="zoom:50%;" />

将这些features和原图进行卷积来得到特征矩阵。

>卷积操作：
>
><img src="https://upload-images.jianshu.io/upload_images/145616-94f12c6cf431dc89.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp" alt="img" style="zoom:50%;" />
>
>当给你一张新的图时，CNN并不能准确地知道这些features到底要匹配原图的哪些部分，所以它会在原图中每一个可能的位置进行尝试。这样在原始整幅图上每一个位置进行匹配计算，我们相当于把这个feature变成了一个过滤器。这个我们用来匹配的过程就被称为卷积操作，这也就是卷积神经网络名字的由来。
>
>这个卷积操作背后的数学知识其实非常的简单。要计算一个feature和其在原图上对应的某一小块的结果，只需要简单地将两个小块内对应位置的像素值进行乘法运算，然后将整个小块内乘法运算的结果累加起来，最后再除以小块内像素点总个数即可。如果两个像素点都是白色（也就是值均为1），那么1*1 = 1，如果均为黑色，那么(-1)*(-1) = 1。不管哪种情况，每一对能够匹配上的像素，其相乘结果为1。类似地，任何不匹配的像素相乘结果为-1。如果一个feature（比如n*n）内部所有的像素都和原图中对应一小块（n*n）匹配上了，那么它们对应像素值相乘再累加就等于n2，然后除以像素点总个数n2，结果就是1。同理，如果每一个像素都不匹配，那么结果就是-1。

最后整张图计算完的样子：

<img src="https://upload-images.jianshu.io/upload_images/145616-e6cafdcdd570e535.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp" alt="img" style="zoom:50%;" />

当换用其它的feature进行同样的操作，最后的得到的结果：

<img src="https://upload-images.jianshu.io/upload_images/145616-cdf0a3911ba67b0f.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp" alt="img"  />

为了完成我们的卷积，我们不断地重复着上述过程，将feature和图中每一块进行卷积操作。最后通过每一个feature的卷积操作，我们会得到一个新的二维数组。这也可以理解为对原始图像进行过滤的结果，我们称之为feature map，它是每一个feature从原始图像中提取出来的“特征”。其中的值，越接近为1表示对应位置和feature的匹配越完整，越是接近-1，表示对应位置和feature的反面匹配越完整，而值接近0的表示对应位置没有任何匹配或者说没有什么关联。

这样我们的原始图，经过不同的feature卷积操作就变成了一系列的feature map
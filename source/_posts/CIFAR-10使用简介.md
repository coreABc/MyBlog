---
title: 深度学习数据集
data: 2023-07-30
update: 2023-07-30
tags: 数据集
categories: 深度学习数据
keywords: 数据集
---



# CIFAR-10使用简介

 CIFAR-10是一个更接近普适物体的彩色图像数据集。CIFAR-10 是由Hinton 的学生Alex Krizhevsky 和Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含10 个类别的RGB 彩色图片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。   

每个图片的尺寸为32 × 32 ，每个类别有6000个图像，数据集中一共有50000 张训练图片和10000 张测试图片。

![image-20230322102758755](C:\Users\26611\AppData\Roaming\Typora\typora-user-images\image-20230322102758755.png)

![image-20230322102944711](C:\Users\26611\AppData\Roaming\Typora\typora-user-images\image-20230322102944711.png)

meta文件中包含每个batch中的用例数量（num_cases_per_batch），标签含义（label_names），每张图片的大小（num_vis,32\*32\*3=3072）

每个batch文件中包含每个batch的标签即在batch1中batch_label(b'training batch 1 of 5')，label(0-9),每个数字表示一种类，图片的numpy数据(data)，图片的名称。

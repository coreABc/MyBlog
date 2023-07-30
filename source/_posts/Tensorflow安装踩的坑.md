---
title: Tensorflow环境安装的问题
data: 2023-07-30
update: 2023-07-30
tags: Tensorflow
categories: 深度学习
keywords: 深度学习环境
---



# Tensorflow安装踩的坑

***



| 环境                   | tensorflow-cpu==2.x | tensorflow==2.x |
| ---------------------- | ------------------- | --------------- |
| 只有GPU                | cpu运行             | cpu运行         |
| 有GPU且安装Cuda和Cudnn | cpu运行             | gpu运行         |
| 有GPU未安装Cuda和Cudnn | cpu运行             | cpu运行         |

在tensorflow 2.x后不再区分是否有gpu，当检测到gpu并安装cuda后，自动调用gpu  

但是，有些人不需要或没有gpu，gpu适配对这部分群体是浪费的（占用不必要的资源），于是有了tensorflow-cpu，我们可以理解其为cpu only版本
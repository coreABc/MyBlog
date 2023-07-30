---
title: VGG16深度学习框架
data: 2023-07-30
update: 2023-07-30
tags: VGG16
categories: 深度学习
keywords: 深度学习框架
---



```
from keras.applications.vgg16 import VGG16

# 下载VGG16模型，下载地址为 c:\user(用户)\.keras\models\vgg16_weights_tf_dim_ordering_tf_kernels.h5
model = VGG16(weights='imagenet', include_top=False)
# 显示模型结构
model.summary()
```

可以使用`keras.applications 模块`进行导入
重要参数：

+ `include_top `：是否包含顶端的全连接层

>`include_top`的作用：
>
>这些模型中的大多数是一系列卷积层，后跟一个或几个密集（或完全连接）层。
>
>`Include_top`允许您选择是否需要最终的密集层。
>
>- 卷积层用作特征提取器。它们识别图像中的一系列图案，每一层都可以通过查看图案的图案来识别更复杂的图案。
>- 密集层能够解释发现的模式以进行分类：此图像包含猫、狗、汽车等。
>
>关于权重：
>
>- 卷积层中的权重是固定大小的。它们是内核 x 过滤器的大小。示例：包含 3 个筛选器的 3x10 内核。卷积层不关心输入图像的大小。它只是进行卷积，并根据输入图像的大小呈现结果图像。（如果不清楚，请搜索一些关于卷积的图解教程）
>- 现在，密集层中的权重完全取决于输入大小。它是输入的每个元素一个权重。因此，这要求你的输入始终是相同的大小，否则你将没有适当的学习权重。
>
>因此，删除最终的密集层允许您定义输入大小（请参阅文档中）。（输出大小将相应增加/减少）。
>
>但是您将丢失解释/分类图层。（您可以添加自己的任务，具体取决于您的任务）

+ `weight`：None代表随机，`imagenet`初始化，代表加载在ImageNet上预训练的权值
+ `input_tensor`：可选，Keras张量作为模型的输入（即layers.Input()输出的tensor）。
+ `input_shape`：可选，输入尺寸元组，当仅include_top=False时有效值（否则输入形状必须是(299, 299, 3)，因为预训练模型是以这个大小训练的）它必须拥有3个输入通道，且宽高必须不小于71例如。(150, 150, 3)是一个合法的输入尺寸。
+ pooling：可选，当include_top为False时，该参数指定了特征提取时的池化方式。
  + None 代表不池化，直接输出最后一层卷积层的输出，该输出是一个4D张量。
  + ‘avg’ 代表平均值平均池化（GlobalAveragePooling2D），相当于在最后一层卷积层后面再加一层平均池化层，输出是一个2D张量。
  + ‘max’ 代表最大池化。
+ `classes`：可选，图片分类的类别数，仅当include_top为和True不加载预训练权值时可用。



### 构建完整的模型

构建序列模型 ==> 添加VGG16模型（输入） ==> 添加全局平均化层 ==> 添加全连接层（输出）


```python
# 构建模型，增加全连接层
model = keras.Sequential()
model.add(conv_base)
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid’))
```

> ```python
> tf.keras.layers.Dense(
>     units,                                 # 正整数，输出空间的维数
>     activation=None,                       # 激活函数，不指定则没有
>     use_bias=True,						   # 布尔值，是否使用偏移向量
>     kernel_initializer='glorot_uniform',   # 核权重矩阵的初始值设定项
>     bias_initializer='zeros',              # 偏差向量的初始值设定项
>     kernel_regularizer=None,               # 正则化函数应用于核权矩阵
>     bias_regularizer=None,                 # 应用于偏差向量的正则化函数
>     activity_regularizer=None,             # Regularizer function applied to the output of the layer (its "activation")
>     kernel_constraint=None,                # Constraint function applied to the kernel weights matrix.
>     bias_constraint=None, **kwargs         # Constraint function applied to the bias vector
> )
> 
> ```

`keras.Sequential()`：建立 Sequential 模型，Sequential 是 Keras 中的一种神经网络框架，可以被认为是一个容器，其中封装了神经网络的结构。Sequential 模型只有一组输入和一组输出。各层之间按照先后顺序进行堆叠。前面一层的输出就是后面一次的输入。通过不同层的堆叠，构建出神经网络。

`GlobalAveragePooling2D()`：是平均池化的一个特例，它不需要指定pool_size和strides等参数，操作的实质是将输入特征图的每一个通道求平均得到一个数值。它的输入和输出维度为：

```python
  Input shape:
      - If `data_format='channels_last'`:
          4D tensor with shape:
          `(batch_size, rows, cols, channels)`
      - If `data_format='channels_first'`:
          4D tensor with shape:
          `(batch_size, channels, rows, cols)`

  Output shape:
      2D tensor with shape:
      `(batch_size, channels)`
```

`activation`：激活函数，无则没有

>**激活函数**
>
>激活函数（Activation Function）是一种添加到人工神经网络中的函数，旨在帮助网络学习数据中的复杂模式。在神经元中，输入的input经过一系列加权求和后作用于另一个函数，这个函数就是这里的激活函数。类似于人类大脑中基于神经元的模型，激活函数最终决定了是否传递信号以及要发射给下一个神经元的内容。在人工神经网络中，一个节点的激活函数定义了该节点在给定的输入或输入集合下的输出。
>
>激活函数可以分为**线性激活函数**（线性方程控制输入到输出的映射，如f(x)=x等）以及**非线性激活函数**（非线性方程控制输入到输出的映射，比如Sigmoid、Tanh、ReLU、LReLU、PReLU、Swish 等）
>
>>**为什么要使用激活函数**
>
>>因为神经网络中每一层的输入输出都是一个线性求和的过程，下一层的输出只是承接了上一层输入函数的线性变换，所以如果没有激活函数，那么无论你构造的神经网络多么复杂，有多少层，最后的输出都是输入的线性组合，纯粹的线性组合并不能够解决更为复杂的问题。而引入激活函数之后，我们会发现常见的激活函数都是非线性的，因此也会给神经元引入非线性元素，使得神经网络可以逼近其他的任何非线性函数，这样可以使得神经网络应用到更多非线性模型中。
>
><img src="https://pic1.zhimg.com/v2-91e1b17ef9b61256739749feff3cea10_r.jpg" alt="img" style="zoom: 80%;" />
>
>**常见的激活函数**：
>
>1. **sigmoid函数**
>
>Sigmoid函数也叫Logistic函数，用于隐层神经元输出，取值范围为(0,1)，它可以将一个实数映射到(0,1)的区间，可以用来做二分类。在特征相差比较复杂或是相差不是特别大时效果比较好。sigmoid是一个十分常见的激活函数，函数的表达式如下：
>
>​	
>$$
>f(x)=\frac{1}{1+e^x}
>$$
>**在什么情况下适合使用 Sigmoid 激活函数呢？**
>
>- Sigmoid 函数的输出范围是 0 到 1。由于输出值限定在 0 到1，因此它对每个神经元的输出进行了归一化；
>- 用于将预测概率作为输出的模型。由于概率的取值范围是 0 到 1，因此 Sigmoid 函数非常合适；
>- 梯度平滑，避免「跳跃」的输出值；
>- 函数是可微的。这意味着可以找到任意两个点的 sigmoid 曲线的斜率；
>- 明确的预测，即非常接近 1 或 0。
>
>**Sigmoid 激活函数存在的不足：**
>
>- **梯度消失**：注意：Sigmoid 函数趋近 0 和 1 的时候变化率会变得平坦，也就是说，Sigmoid 的梯度趋近于 0。神经网络使用 Sigmoid 激活函数进行反向传播时，输出接近 0 或 1 的神经元其梯度趋近于 0。这些神经元叫作饱和神经元。因此，这些神经元的权重不会更新。此外，与此类神经元相连的神经元的权重也更新得很慢。该问题叫作梯度消失。因此，想象一下，如果一个大型神经网络包含 Sigmoid 神经元，而其中很多个都处于饱和状态，那么该网络无法执行反向传播。
>- **不以零为中心**：Sigmoid 输出不以零为中心的,，输出恒大于0，非零中心化的输出会使得其后一层的神经元的输入发生偏置偏移（Bias Shift），并进一步使得梯度下降的收敛速度变慢。
>- **计算成本高昂**：exp() 函数与其他非线性激活函数相比，计算成本高昂，计算机运行起来速度较慢。
>
>
>
>2. **ReLU激活函数**
>
>ReLU函数又称为修正线性单元（Rectified Linear Unit），是一种分段线性函数，其弥补了sigmoid函数以及tanh函数的梯度消失问题，在目前的深度神经网络中被广泛使用。ReLU函数本质上是一个斜坡（ramp）函数，公式及函数图像如下：
>$$
>f(x)=\begin{cases}
>x, x\geqslant0\\
>0, x\leqslant0
>\end{cases}=max(0,x)
>$$
>
>
>
>
>
>
>**全连接层的作用**
>
>全连接层在整个网络卷积神经网络中起到“特征提取器”的作用。如果说卷积层、池化层和激活函数等操作是将原始数据映射到隐层特征空间的话，全连接层则起到将学到的特征表示映射到样本的标记空间的作用。
>
>一段来自知乎的通俗理解：
>
>从卷积网络谈起，卷积网络在形式上有一点点像咱们正在召开的“人民代表大会”。卷积核的个数相当于候选人，图像中不同的特征会激活不同的“候选人”（卷积核）。池化层（仅指最大池化）起着类似于“合票”的作用，不同特征在对不同的“候选人”有着各自的喜好。
>
>全连接相当于是“代表普选”。所有被各个区域选出的代表，对最终结果进行“投票”，全连接保证了receiptive field 是整个图像，既图像中各个部分（所谓所有代表），都有对最终结果影响的权利。
>
>**全连接层的原理**
>
>在卷积神经网络的最后，往往会出现一两层全连接层，全连接一般会把卷积输出的二维特征图转化成一维的一个向量，这是怎么来的呢？目的何在呢？
>
>![img](https://img-blog.csdnimg.cn/20210413094056578.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MjUxNjE2,size_16,color_FFFFFF,t_70#pic_center)
>
>最后的两列小圆球就是两个全连接层的输出，在最后一层卷积结束后，进行了最后一次池化，得到20个12\*12的图像，经过全连接层变成了1\*100的向量，再次经过一次全连接层变成的1\*10的向量输出。
>
>从第一步是如何到达第三步的呢，其实就是有20\*100个12\*12的不同卷积核卷积出来的，我们也可以这样想，就是每个神经元的输出是12\*12\*20个输入值与对应的权值乘积的和。对于输入的每一张图，用了一个和图像一样大小的核卷积，这样整幅图就变成了一个数了，如果厚度是20就是那20个核卷积完了之后相加求和。这样就能把一张图高度浓缩成一个数了。

### 模型的初步训练

#### 训练步骤

1. 在预训练卷积上添加自定义层
2. 使用`conv_base.trainable = False`冻结卷积基所有层
3. 训练添加的分类层
4. 解冻卷积基的一部分层
5. 联合训练解冻的卷积层和添加的自定义层

在初步训练中要**锁定**卷积基的参数值，因为我们自己加入的全连接层的参数是随机初始化的，在初步训练中会影响卷积基的参数，导致降低准确率。

```python
conv_base.trainable = False  # 使得VGG卷积中的参数不可训练
# 因为在训练模型是，最后的两层全连接层是随机初始化的参数，有可能会影响到VGG16卷积层的参数

# 优化模型
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
             loss='binary_crossentropy',
             metrics=['acc’])

# 进行训练
history = model.fit(    train_image_ds,
                        steps_per_epoch=train_count//BATCH_SIZE,
                        epochs=5,
                        validation_data=test_image_ds,
                        validation_steps=test_count//BATCH_SIZE
                    )

>>>Train for 62 steps, validate for 31 steps
Epoch 1/5
62/62 [==============================] - 407s 7s/step - loss: 0.6301 - acc: 0.6190 - val_loss: 0.5986 - val_acc: 0.6552
...
```

>**model.compile(optimizer = 优化器，**
>
>​            **loss = 损失函数，**
>
>​            **metrics = ["准确率”])**
>
>其中：
>
>**optimizer**可以是字符串形式给出的优化器名字，也可以是函数形式，使用函数形式可以设置学习率、动量和超参数
>
>例如：`sgd`或者`tf.optimizers.SGD(lr=学习率,decay=学习衰减率,momentum=动量参数)`
>
>`adagrad`或者`tf.keras.optimizers.Adagrad(lr=学习率,decay=学习衰减率)`
>
>`adadelta`或者` tf.keras.optimizers.Adadelta(lr=学习率,decay=学习衰减率)`
>
>`adam`或者`  tf.keras.optimizers.Adam(lr=学习率,decay=学习衰减率)`
>
>**loss**可以是字符串形式给出的损失函数的名字，也可以是函数形式
>
>例如：`msc`或者`tf.keras.losses.MeanSquaredError()`
>
>`sparse_categorical_crossentropy` 或者` tf.keras.losses.SparseCatagoricalCrossentropy(from_logits = False)`损失函数经常需要使用softmax函数来将输出转化为概率分布的形式，在这里from_logits代表是否将输出转为概率分布的形式，为False时表示转换为概率分布，为True时表示不转换，直接输出
>
>
>
>**Metrics标注网络评价指标**
>
>例如：`accuracy`：y\_ 和 y 都是数值，如y_ = [1] y = [1] #y\_为真实值，y为预测值
>
>`sparse_accuracy`：y\_和y都是以独热码 和概率分布表示，如y\_ = [0, 1, 0], y = [0.256, 0.695, 0.048]
>
>`sparse_categorical_accuracy`：y\_是以数值形式给出，y是以独热码给出，如y\_ = [1], y = [0.256 0.695, 0.048]



![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9zdGF0aWMub3NjaGluYS5uZXQvdXBsb2Fkcy9zcGFjZS8yMDE4LzAzMTQvMDIzMDQ0X1g0OVJfODc2MzU0LnBuZw?x-oss-process=image/format,png)

1、输入224x224x3的图片，经64个3x3的卷积核作两次卷积+ReLU，卷积后的尺寸变为224x224x64
2、作max pooling（最大化池化），池化单元尺寸为2x2（效果为图像尺寸减半），池化后的尺寸变为112x112x64
3、经128个3x3的卷积核作两次卷积+ReLU，尺寸变为112x112x128
4、作2x2的max pooling池化，尺寸变为56x56x128
5、经256个3x3的卷积核作三次卷积+ReLU，尺寸变为56x56x256
6、作2x2的max pooling池化，尺寸变为28x28x256
7、经512个3x3的卷积核作三次卷积+ReLU，尺寸变为28x28x512
8、作2x2的max pooling池化，尺寸变为14x14x512
9、经512个3x3的卷积核作三次卷积+ReLU，尺寸变为14x14x512
10、作2x2的max pooling池化，尺寸变为7x7x512
11、与两层1x1x4096，一层1x1x1000进行全连接+ReLU（共三层）
12、通过softmax输出1000个预测结果






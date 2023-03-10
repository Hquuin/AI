# 卷积神经网络CNN

## 基本原理

构成

- 卷积层：提取图像中的局部特征
  
  每个卷积核带若干权重和1个偏置
  
  1. 卷积核大小：1×1、3×3、5×5
  
  2. 卷积核步长：一般为1
  
  3. 卷积核个数：若干个，不同结构选择不同
  
  4. 卷积核零填充大小：有两种方式，SAME和VALID
  
     - SAME：越过边缘取样，取样的面积和输入图像的像素宽度一致。
     - VALID：不越过边缘取样，取样的面积小于输入人的图像的像素宽度。
  
  5. 输入大小公式：
  
     <img src="https://img2018.cnblogs.com/blog/1569451/201906/1569451-20190608161534963-2057401102.png" alt="img" style="zoom: 50%;" />
  
- 非线性激励层：增加非线性分割能力

  激活函数：

  <img src="https://img2018.cnblogs.com/blog/1569451/201906/1569451-20190608165330235-820842391.png" alt="img" style="zoom: 33%;" />

  

  <img src="https://img2018.cnblogs.com/blog/1569451/201906/1569451-20190608165355579-886884867.png" alt="img" style="zoom:50%;" />

  

  

- 池化层：大幅降低参数量级（降维）

  通常池化层采用 2x2大小、步长为2窗口

   输入大小公式：同卷积核

- 全连接层：输出结果

​       前面的卷积和池化相当于做特征工程，最后的全连接层在整个卷积神经网络中起到“分类器”的作用（如果FC层作为最后一层，再加上softmax或者wx+b，则可以分别作为分类或回归的作用，即“分类器”或“回归器”的作用）；如果作为倒数第2，3层的话，FC层的作用是信息融合，增强信息表达





##  CNN总结

**<img src="https://img2018.cnblogs.com/blog/1569451/201906/1569451-20190608170400177-1669453074.png" alt="img" style="zoom:50%;" />**

- **优点**
  - 共享卷积核，优化计算量
  - 无需手动选取特征，训练好权重，即得特征
  -  深层次的网络抽取图像信息丰富，表达效果好
- **缺点**
  - 需要调参，需要大样本量， GPU等硬件依赖
  - 物理含义不明确（可解释性不强）
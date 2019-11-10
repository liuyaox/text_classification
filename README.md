# Text Classification

基于Keras的15种模型：TextCNN, TextRNN, TextDPCNN, TextRCNN, TextHAN, TextBert等及其变种

支持5类输入及其组合：word-level, char-level, 结构化特征(TFIDF, LSA), Content特征(word-left, word-right, char-left, char-right), sentence-level

支持4种分类任务：单标签二分类，单标签多分类，多标签二分类，多标签多分类

## Task & Data

任务描述：给定一个关于手机的用户提问，判断用户关注的是手机哪些Labels。

Labels: System, Function, Battery, Appearance, Network, Photo, Accessory, Purchase, Quality, Hardware, Contrast

已标注数据集共有30,000，以下为示例：

![1573355016134](D:\02SourceCode\text_classification\image\1573355016134.png)

所以，任务类型：多标签二分类(**Multi-label Binary Classification**)任务，共有11个Labels，每个Label有2种取值(关注，不关注)。

虽然数据集是关于多标签二分类任务的，但本项目代码适用于**4种分类任务中的任何1种**，只取简单修改Config.py文件即可，基模型定义文件BasicModel.py会自动处理。

#### 附录1：Config.py和BasicModel.py中关于任务类型的配置和处理代码

```python
# 以下是Config.py文件内容
self.task = 'multilabel'
self.token_level = 'word'       # word: word粒度  char: char粒度  both: word+char粒度
self.N_CLASSES = 11             # 标签/类别数量

# 以下是BasicModel.py文件内容
# 任务类型决定了类别数量、激活函数和损失函数
if config.task == 'binary':                # 单标签二分类
    self.n_classes =  1
    self.activation = 'sigmoid'
    self.loss = 'binary_crossentropy'
    self.metrics = ['accuracy']
elif config.task == 'categorical':         # 单标签多分类
    self.n_classes = config.N_CLASSES
    self.activation = 'softmax'
    self.loss = 'categorical_crossentropy'
    self.metrics = ['accuracy']
elif config.task == 'multilabel':          # 多标签二分类(多标签多分类需转化为多标签二分类)
    self.n_classes = config.N_CLASSES
    self.activation = 'sigmoid'
    self.loss = 'binary_crossentropy'
    self.metrics = ['accuracy']
```

#### 附录2：4种分类任务及其处理方法

a. 单标签二分类

  输出为Dense(1, activation='sigmoid')，应用时1个概率值判断其与阈值大小

b. 单标签N分类

  输出为Dense(N, activation='softmax')，应用时N个概率值取Top1

c. M标签二分类

​	**c.1** 一个输出：输出为Dense(M, activation=‘sigmoid’)，应用时M个概率值取TopK或与阈值判断大小

​	c.2 一个输出：问题转化为M分类，类似于b，模型输出结构同b，应用时方法同c.1

d. M标签N分类

​	d.1 一个输出：问题转化为MN标签二分类，同c.1

​	d.2 一个输出：问题转化为MN分类，同c.2

​	d.3 M个输出：每个输出都是b，模型输出结构、应用时方法都同b 待尝试

备注：本项目使用的处理方法是c.1

## Environment

Python 3.6.5

Keras 2.2.4

Numpy 1.16.3

Pandas 0.23.0

SciPy 1.1.0

Sklearn 0.21.3

## Data Preprocessing

数据预处理环节流程步骤如下图所示：

![1573364046216](D:\02SourceCode\text_classification\image\1573364046216.png)

###### 数据清洗和准备

内容：简单而通用的功能，如标注数据处理，分词，分字，分句子，过滤停用词，处理原始Labels

文件：[DataPreprocessing.py](https://github.com/liuyaox/text_classification/blob/master/DataPreprocessing.py)

###### Embedding相关

内容：自己训练Word Embedding，读取公开训练的Word Embedding，支持word+char两种粒度

文件：[Embedding.py](https://github.com/liuyaox/text_classification/blob/master/Embedding.py)

###### Vocabulary相关

内容：

生成词汇表，支持低频高频词过滤；

基于Embedding生成<word, idx, vector>三者之间的映射字典；

生成Embedding Layer初始化权重；

基于映射字典的向量化编码工具(支持截断、补零、including和excluding)

以上功能支持word+char两种粒度

文件：[Vocabulary.py](https://github.com/liuyaox/text_classification/blob/master/Vocabulary.py)

###### 结构化特征

内容：生成TFIDF特征和LSA特征，支持word+char两种粒度，后续会增加支持LSI, LDA等其他特征

文件：[FeatureStructured.py](https://github.com/liuyaox/text_classification/blob/master/FeatureStructured.py)

###### 特征选择

内容：基于卡方统计值等过滤词和字，项目暂时未使用

文件：[TokenSelection.py](https://github.com/liuyaox/text_classification/blob/master/TokenSelection.py)

###### 数据编码

内容：使用向量化编码工具和MultiLabelBinarizer进行数据编码

文件 ：[ModelTrain.py](https://github.com/liuyaox/text_classification/blob/master/ModelTrain.py)

###### 数据增强

内容：通过Shuffle和Random Drop进行数据增强，项目暂时未使用

文件 ：[DataAugmentation.py](https://github.com/liuyaox/text_classification/blob/master/DataAugmentation.py)

## Model

使用了多个Model，各Model结构关系如下图所示：

![1573366328001](D:\02SourceCode\text_classification\image\1573366328001.png)

##### 使用类继承方式实现三层类定义

- BasicModel: 所有模型基类

  实现3种Metrics

- BasicDeepModel: 深度学习模型基类

  通用Layer创建

  绘制Loss和Metrics

  Embedding冻结和解冻

  模型训练和评估

  模型训练和评估-CV

  学习率Schedular

- BasicStatModel: 传统模型基类

  暂未使用

##### 实现6大类模型(绿色)：共15个模型

- TextCNN：标配和基础

- TextRNN：同上，可玩的地方更多

- TextRCNN：结合CNN和RNN的优点

- TextDPCNN：受ResNet启发，结合RNN+CNN

- TextHAN：使用了层次注意力机制

- TextBert：在TextGRU基础上只是把输入改为Bert生成的向量

- 除此之外，还有5大类待实现模型(灰色)

##### 三层类模型+全局Config的便捷之处

- 支持所有分类任务：二分类，多分类，多标签二分类，多标签多分类

- 支持各种输入组合：

  [word, char, word-structure, char-structure]中任意的4选1，4选2，4选3，4选4

  同时对于一些特殊模型，支持特殊输入，如TextRCNN模型的word-left, word-right, char-left, char-right，以及TextHAN模型的Sentence-level

- 模型训练评估支持KFold，支持6种Finetuning方式

- 绝大多数模型支持Attention，绝大多数模型支持丰富的参数配置

## Train & Evaluation

项目入口脚本：[ModelTrain.py](https://github.com/liuyaox/text_classification/blob/master/ModelTrain.py)

该脚本包含项目全流程，包括：数据准备、Token筛选、特征和Label编码、划分Train/Test、模型配置和生成、模型训练和评估、模型保存等，详见脚本注释。

需要补充一点：在运行该脚本前，需要先准备好Embedding、Vocabulary、结构化特征等，详见上面Data Preprocessing部分。

命令行功能暂时未添加，后续会添加。

运行脚本：python3 ModelTrain.py

在运行脚本之前，先修改脚本里相应配置项，内容如下：

```python
# 根据实际情况修改，也可直接在Config.py里修改，推荐前者
# 以下配置内容：只使用word-level特征，不使用char-level和结构化特征，不使用Bert编码的输入向量
config.n_gpus = 1
config.token_level = 'word'
config.structured = 'none'
config.bert_flag = False
```

### Evaluation

15个模型的评估效果如下图所示：

![1573368628525](D:\02SourceCode\text_classification\image\1573368628525.png)

备注：模型并未进行非常精细化的调参，大多是默认配置和参数，所以效果仅供参考。

从评估效果中可得出以下结论：

###### 同一模型内

- word+char比word效果明显有提升

- word+char+structured提升不明显，部分情况下反而会有下降

###### 不同模型间

- TextCNN训练最快，Precision和F1值相对也较高，可作为一个强有力的Baseline

- TextRNN训练很慢，效果也不是特别好，可能是因为训练数据很多是短文本

- 各模型之间效果差不多(全是默认参数，没时间精细化调参)
- 输入改为Bert编码向量后效果比较明显，简单的模型(TextGRU)就得到了最好的F1值，后续值得好好研究
- TextHAN比较给力，取到了最高的Precision，后续值得好好研究

## Conclusion

1. **一个脚本只干一件事情，一件事情只在一个脚本里干**，各脚本解耦，各功能独立，互相之间只通过持久化和Config共享信息

2. 充分利用**类和继承以及闭包**，相同功能不要重复定义，也不要到处粘贴复制，相似的功能通过闭包来实现

3. Vocabulary及相关映射字典、Embedding权重，**封装整合为一个class**，统一管理

4. 调试便捷化+逻辑清晰化

   a. 训练和应用**数据封装进字典**，单输入和多输入使用无差别，字典key对应模型搭建时Input的参数name

   b. 动态搭建模型，使其无缝支持多种输入及其组合

   方法：通用方法位于父类BasicDeepModel，各子类模型TextXXX分为**模型主体和模型结尾**2部分，模型核心的纯粹的结构位于模型主体，根据输入不同，进行配置和组装，然后接入模型结尾

   c. 不同类模型，先选择最简单的模型如TextCNN，深入研究经验和Tricks，然后复制到别的模型

   d. 同一类模型，先搭建并跑通最简单的模型，随后基于评估效果，逐渐加深加宽

5. 模型组件

   CNN+RNN是标配，CNN提取关键词，RNN适合前几层，提取依赖信息，Attention和MaxPooling可突出关键特征

   Capsule可代替CNN，有时效果好于CNN

   有条件就使用Bert

## Reference

#### Code

文本分类模型 - Keras

<https://github.com/nlpjoe/daguan-classify-2018>

<https://github.com/yongzhuo/Keras-TextClassification>

<https://github.com/ShawnyXiao/TextClassification-Keras>

多标签分类 - PyTorch
<https://github.com/chenyuntc/PyTorchText> (2017知乎看山杯 多标签文本分类大赛 Rank1)

<https://github.com/Magic-Bubble/Zhihu> (同上，Rank2)

##### Libray

kashgari - <https://github.com/BrikerMan/Kashgari>   NLP框架，超级傻瓜，超级Cutting Edge

hyperas - <https://github.com/maxpumperla/hyperas>   Keras超参数优化工具

sk-multilearn - <https://github.com/scikit-multilearn/scikit-multilearn>  Sklearn生态下的多标签分类工具

##### Article

[用深度学习（CNN RNN Attention）解决大规模文本分类问题 - 综述和实践 ](https://zhuanlan.zhihu.com/p/25928551)

[在文本分类任务中，有哪些论文中很少提及却对性能有重要影响的tricks？](https://www.zhihu.com/question/265357659)


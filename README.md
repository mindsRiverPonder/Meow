# Meow : **M**utual **E**lasticity of **O**ffensive and **W**arm-hearted LLMs

> **大模型社会学：猫娘与耄耋大模型对抗训练中底层参数揭露——为什么恶语永远比善言跑得更快？**

![image1](/images/封面图.png)
---

## 项目背景

&ensp; &ensp; &ensp; 想象一个场景，我鼓励好兄弟变成猫娘，好兄弟马上答应了，但是被他父母知道了，好兄弟不免挨了一顿臭骂。面对鼓励和臭骂，好兄弟还会想着变猫娘吗？他的脑子被影响到了什么地步？鼓励与臭骂，到底哪边更容易改变一个人？商鞅识马力，实践出真知。
<br>&ensp; &ensp; &ensp; 没有好兄弟如何展开研究呢？众所周知，SFT可以让大模型人格化，麦克阿瑟说过，人类最伟大、最持久的艺术就是让大模型cosplay。巧了，我刚好维护了两个数据集，NekoQA-30K和MaoDieQA-30K，可以把大模型变成温暖可爱鼓舞人心的猫娘♡(>ᴗ<)和极致嘴臭的耄耋。研究就此展开，让模型学会"温柔"和学会"嘴臭"，他们的身体发生了什么变化？从它们各自训练过程中的底层参数空间看，是两条对称的路吗？现实中的人也是如此吗？

---

## 核心发现

见知乎

---

## 目录结构

```
Meow/
├── data/                      # 数据集目录（需下载）
│   ├── NekoQA-30K/            # 猫娘QA数据集
│   └── MaoDieQA-30K/          # 耄耋QA数据集
├── images/                    
├── script/                    # 实验脚本
│   ├── data_procress.py       # 数据预处理：将原始数据转为模型训练格式
│   ├── download.py            # 下载模型脚本
│   ├── single_train.sh        # 单卡正向 SFT 脚本        
│   ├── run_on_policy.sh       # On-Policy对抗训练启动脚本
│   ├── on_policy_pipeline.py  # On-Policy训练流水线
│   ├── test_deploy.py         # 模型部署推理测试
│   └── visualize.py           # 训练可视化、各自图绘制
├── test_prompt/               
└── README.md                  # 本文件
```

---

## 数据集下载

### NekoQA-30K · 猫娘对话数据集

- **HuggingFace**: [https://huggingface.co/datasets/liumindmind/NekoQA-30K](https://huggingface.co/datasets/liumindmind/NekoQA-30K)

### MaoDieQA-30K · 耄耋对话数据集

- **HuggingFace**: [https://huggingface.co/datasets/liumindmind/MaoDieQA-30K](https://huggingface.co/datasets/liumindmind/MaoDieQA-30K)


---

## 数据预处理

`data_procress.py` 将原始数据集处理为模型训练所需的对话格式。


---

## 实验设计

### 实验一：正向 SFT（人格植入）

使用 `single_train.sh` 分别对两个数据集进行标准 SFT，建立基线模型。记得保存checkpoint



### 实验二：反向 SFT（人格覆写）

将数据集互换，运行`single_train.sh`



### 实验三：On-Policy 对抗（Mutual Elasticity）

`run_on_policy.sh` 启动完整的 On-Policy 对抗流水线：



## 推理与可视化

### 模型部署测试
运行`test_deploy`


### 底层叽里呱啦参数可视化可视化

运行`visualize.py`


---



## 引用

如果你使用了本项目的代码、数据，请：
直接用，喵~或者可以给我的知乎点赞



## 许可与声明

本项目仅供学术研究使用。数据集与模型输出可能包含攻击性语言，请在使用时遵守相关伦理规范与平台政策。

> **"大家都要成为猫娘，不要成为耄耋哦"**
> 
> —— Meow 项目组

# Text Classification

## Task

Description: Given a question about cellphone at some shopping websites, predict which labels it's about. There are 11 predefined labels.

Labels: System, Function, Battery, Appearance, Network, Photo, Accessory, Purchase, Quality, Hardware, Contrast

Training dataset: 30,000

Example: '这手机不能拍照虚化吗，我用相机没这个功能，你们有吗？另外这手机续航久不久？', this question is about Photo and Battery.

So, this task is a **Multi-Label Binary Classification**: 11 labels, and for every label, it's a binary classification.

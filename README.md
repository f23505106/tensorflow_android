# tensorflow_android
使用tensorflow lite用mnist数据训练的0-9手写数字识别

![](https://markdown-1251303493.cos.ap-beijing.myqcloud.com/mnist-app.png)

模型的来自 https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

训练模型

```python
python mnist.py
```

生成frozen_model.pb

使用mnist-verify.py 验证生成的模型

使用

```python
tflite_convert --graph_def_file=frozen_model.pb --output_file=model.tflite --input_arrays=Placeholder_3 --output_arrays=probabilities,classes
```
生成tflite文件

使用tflite-test.py验证生成的tflite模型

![](https://markdown-1251303493.cos.ap-beijing.myqcloud.com/mnist-model.png)

# t5-pegasus-chinese
## 关于数据预处理
训练代码没有经过更改，这边需要将训练集经过转换得到模型可以正确识别的形式
```bash
python data_process.py
```
测试集可以直接读取，不需要预处理
## 关于推理过程
！！！测试集需要把CSV文件的表头（第一行）删掉，不然测试代码会将其误认为一行数据
## 包的环境
！！！注意Python要3.8或者3.7，更高版本的无法支持1.8.0的GPU版本的torch
| Package                   | Version                 | Build              | Channel  |
|---------------------------|-------------------------|--------------------|----------|
| _libgcc_mutex              | 0.1                     | main               |          |
| _openmp_mutex              | 5.1                     | 1_gnu              |          |
| bert4torch                 | 0.4.0                   | pypi_0             | pypi     |
| blas                       | 1.0                     | mkl                |          |
| ca-certificates            | 2024.11.26              | h06a4308_0         |          |
| certifi                    | 2024.8.30               | pypi_0             | pypi     |
| charset-normalizer         | 3.4.0                   | pypi_0             | pypi     |
| click                      | 8.1.7                   | pypi_0             | pypi     |
| cuda-cudart                | 12.4.127                | 0                  | nvidia  |
| cuda-cupti                 | 12.4.127                | 0                  | nvidia  |
| cuda-libraries             | 12.4.1                  | 0                  | nvidia  |
| cuda-nvrtc                 | 12.4.127                | 0                  | nvidia  |
| cuda-nvtx                  | 12.4.127                | 0                  | nvidia  |
| cuda-opencl                | 12.4.127                | 0                  | nvidia  |
| cuda-runtime               | 12.4.1                  | 0                  | nvidia  |
| cudatoolkit                | 11.1.74                 | h6bb024c_0         | nvidia  |
| filelock                   | 3.16.1                  | pypi_0             | pypi     |
| fsspec                     | 2024.10.0               | pypi_0             | pypi     |
| huggingface-hub            | 0.26.3                  | pypi_0             | pypi     |
| idna                       | 3.10                    | pypi_0             | pypi     |
| intel-openmp               | 2021.4.0                | h06a4308_3561      |          |
| jieba                      | 0.42.1                  | pypi_0             | pypi     |
| joblib                     | 1.4.2                   | pypi_0             | pypi     |
| libcublas                  | 12.4.5.8                | 0                  | nvidia  |
| libcufft                   | 11.2.1.3                | 0                  | nvidia  |
| libcufile                  | 1.9.1.3                 | 0                  | nvidia  |
| libcurand                  | 10.3.5.147              | 0                  | nvidia  |
| libcusolver                | 11.6.1.9                | 0                  | nvidia  |
| libcusparse                | 12.3.1.170              | 0                  | nvidia  |
| libedit                    | 3.1.20230828            | h5eee18b_0         |          |
| libffi                     | 3.2.1                   | hf484d3e_1007      |          |
| libgcc-ng                  | 11.2.0                  | h1234567_1         |          |
| libgomp                    | 11.2.0                  | h1234567_1         |          |
| libnpp                     | 12.2.5.30               | 0                  | nvidia  |
| libnvfatbin                | 12.4.127                | 0                  | nvidia  |
| libnvjitlink               | 12.4.127                | 0                  | nvidia  |
| libnvjpeg                  | 12.3.1.117              | 0                  | nvidia  |
| libstdcxx-ng               | 11.2.0                  | h1234567_1         |          |
| libuv                      | 1.48.0                  | h5eee18b_0         |          |
| mkl                        
---
基于GOOGLE T5中文生成式模型的摘要生成/指代消解，支持batch批量生成，多进程

**如果你想了解自己是否需要本Git，请看如下几点介绍（重点）：**
1. 模型可部署在CPU/GPU，均测试可用
2. 基于谷歌t5的中文生成式预训练模型
3. 集成了中文摘要生成、指代消解等生成任务语料，开箱即用
4. 基于PyTorch
5. 支持多张显卡DataParallel
6. 支持batch批量推理/生成
7. 支持多进程，推理/生成提速

**本 Git 如何运行：**  
1. 所需Python库  
    - transformers==4.15.0  
    - tokeniziers==0.10.3  
    - torch==1.7.0或1.8.0或1.8.1均可
    - jieba
    - rouge
    - tqdm
    - pandas 
2. 下载t5-pegasus模型放在 t5_pegasus_pretain目录下，目录下三个文件：
   - pytorch_model.bin
   - config.json
   - vocab.txt  

    预训练模型下载地址（追一科技开源的t5-pegasus的pytorch版本，分享自renmada）：
    - Base版本：
      - 百度网盘：https://pan.baidu.com/s/1JIjEEyX-dgmqpQdL7aNbAw 提取码：fd7k
      - Google Drive：https://drive.google.com/file/d/18Y5LVghAGbz7ys0noii1eM1yDtFmW490/view?usp=sharing
    - Small版本：
      - 百度网盘：https://pan.baidu.com/s/1Kc6xFqJZoVxKLBGx924zgQ 提取码：hvq9
      - Google Drive：https://drive.google.com/file/d/1fCL7f_f8I6YuezoYQ3EvT9h-S1WngKZo/view?usp=sharing

    解压后，按上面说的放在对应目录下，文件名称确认无误即可。
3. 命令行执行
   - 训练finetune
        ```bash
        python train_with_finetune.py
        ```
   - 预测generate
        ```bash
        python predict_with_generate.py
        ```
   - 预测generate(多进程，仅支持Linux系统，Windows系统不可用)
        ```bash
        python predict_with_generate.py --use_multiprocess
        ```
**语料介绍：**

**t5-pegasus模型的细节，以便了解它为什么能在摘要任务中有效:**

**实验结果：**


**如对本Git内容存有疑问或建议，欢迎在issue区或者邮箱isguanjing@126.com与我联系。**

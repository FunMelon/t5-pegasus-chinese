import pandas as pd

# 读取同级目录下的 train_dataset.csv 文件，指定分隔符为 |
df = pd.read_csv('train_dataset.csv', sep='|')

# 提取摘要（abstract）和完整文本（content）列
df_processed = df[['abstract', 'content']]

# 计算数据集总行数
total_rows = len(df_processed)
print(f"Total rows in the dataset: {total_rows}")

# 计算验证集的大小，10% 用作验证集
valid_size = total_rows // 10

# 将数据集按照 9:1 划分，前 90% 作为训练集，后 10% 作为验证集
train_df = df_processed.iloc[:-valid_size]
valid_df = df_processed.iloc[-valid_size:]

# 将训练集保存为 train2.tsv 文件，分隔符为制表符
train_df.to_csv('train2.tsv', sep='\t', index=False, header=False)

# 将验证集保存为 valid.tsv 文件，分隔符为制表符
valid_df.to_csv('valid.tsv', sep='\t', index=False, header=False)

print("Data preprocessing complete. 'train.tsv' and 'valid.tsv' have been created.")

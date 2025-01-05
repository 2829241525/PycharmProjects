import pandas as pd
from sklearn.model_selection import train_test_split
import os
import chardet

def analyze_labels(df):
    """
    分析标签分布
    """
    print("\n标签分布统计:")
    label_counts = df['label'].value_counts()
    print(label_counts)
    print("\n标签分布百分比:")
    print(label_counts / len(df) * 100, "%")

def clean_and_analyze_data(df):
    """
    数据清理和分析
    """
    # 显示基本信息
    print("\n原始数据基本信息:")
    print(df.info())
    print("\n原始数据示例:")
    print(df.head())
    print("\n列名:", df.columns.tolist())
    
    # 删除索引列（如果存在）
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # 选择需要的列（spl_desc 和 s_desc）
    if 'spl_desc' in df.columns and 's_desc' in df.columns:
        df = df[['spl_desc', 's_desc']]
    else:
        # 如果列名不匹配，打印所有列名并抛出错误
        print("\n可用的列名:", df.columns.tolist())
        raise ValueError("找不到所需的列 'spl_desc' 和 's_desc'")
    
    # 重命名列
    df.columns = ['text1', 'text2']
    
    # 确保文本列为字符串类型
    df['text1'] = df['text1'].astype(str)
    df['text2'] = df['text2'].astype(str)
    
    # 数据清理
    # 1. 删除重复行
    df_cleaned = df.drop_duplicates()
    print(f"\n删除重复行: {len(df) - len(df_cleaned)} 行被删除")
    
    # 2. 删除空值
    df_cleaned = df_cleaned.dropna()
    print(f"删除空值后的行数: {len(df_cleaned)}")
    
    # 3. 删除文本为空的行（处理空字符串和只包含空白字符的情况）
    df_cleaned = df_cleaned[df_cleaned['text1'].str.strip() != '']
    df_cleaned = df_cleaned[df_cleaned['text2'].str.strip() != '']
    print(f"删除空文本后的行数: {len(df_cleaned)}")
    
    # 4. 清理文本（去除首尾空白字符）
    df_cleaned['text1'] = df_cleaned['text1'].str.strip()
    df_cleaned['text2'] = df_cleaned['text2'].str.strip()
    
    # 5. 添加标签列：如果两个文本完全相同（不区分大小写），则标记为1，否则为0
    df_cleaned['label'] = (df_cleaned['text1'].str.lower() == df_cleaned['text2'].str.lower()).astype(int)
    
    print("\n标签分布:")
    analyze_labels(df_cleaned)
    
    print(f"\n清理后的数据形状: {df_cleaned.shape}")
    
    # 显示一些示例数据
    print("\n数据示例:")
    print(df_cleaned.head())
    
    return df_cleaned

def load_and_split_data(csv_path, output_dir, test_size=0.2, random_state=42):
    """
    读取CSV文件，清理数据，并将数据分割为训练集和测试集，然后保存到指定目录
    """
    # 尝试不同的读取方式
    try:
        print("\n尝试使用 latin1 编码读取...")
        df = pd.read_csv(csv_path,
                         encoding='latin1',
                         on_bad_lines='skip',
                         low_memory=False)
    except Exception as e:
        print(f"第一次尝试失败: {str(e)}")
        try:
            print("\n尝试使用 errors='ignore' 选项读取...")
            df = pd.read_csv(csv_path, 
                           encoding='GB18030', 
                           errors='ignore',
                           on_bad_lines='skip',
                           low_memory=False)
        except Exception as e:
            print(f"第二次尝试失败: {str(e)}")
            try:
                print("尝试使用 errors='replace' 选项读取...")
                df = pd.read_csv(csv_path,
                                 encoding='GB18030',
                                 errors='replace',
                                 on_bad_lines='skip',  # 跳过有问题的行
                                 low_memory=False)  # 避免数据类型推断的警告

            except Exception as e:
                print(f"所有尝试都失败了: {str(e)}")
                raise
    
    print("原始数据形状:", df.shape)
    
    # 数据清理和分析
    #df_cleaned = clean_and_analyze_data(df)
    df_cleaned = df

    # 分割数据
    train_df, test_df = train_test_split(
        df_cleaned, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df_cleaned['label']  # 确保训练集和测试集的标签分布一致
    )
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整的训练集和测试集
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False, encoding='latin1')
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False, encoding='latin1')
    
    # 打印分割后的标签分布
    print("\n训练集标签分布:")
    print(train_df['label'].value_counts())
    print("\n测试集标签分布:")
    print(test_df['label'].value_counts())
    
    print(f"\n数据已保存到目录: {output_dir}")
    
    # 为了保持接口一致，仍然返回分割后的特征和标签
    X_train = train_df[['text1', 'text2']]
    y_train = train_df['label']
    X_test = test_df[['text1', 'text2']]
    y_test = test_df['label']
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # 使用完整的文件路径
    data_dir = r"/week8 文本匹配问题/data"
    csv_path = os.path.join(data_dir, "roomtype.csv")
    output_dir = data_dir
    
    X_train, X_test, y_train, y_test = load_and_split_data(csv_path, output_dir)
    
    # 打印数据集的形状
    print("\n最终数据集形状:")
    print("训练集特征形状:", X_train.shape)
    print("测试集特征形状:", X_test.shape)
    print("训练集标签形状:", y_train.shape)
    print("测试集标签形状:", y_test.shape) 
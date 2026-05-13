"""
实战：用 train.xlsx 训练，填充 missing.xlsx
"""
import warnings
import pandas as pd
import numpy as np
import joblib
import os

warnings.filterwarnings('ignore')

from hyperimpute.plugins.imputers import Imputers


def train_and_fill_custom_files():
    """用指定的训练文件训练，填充指定的目标文件"""
    
    print("=" * 70)
    print("实战：用 train.xlsx 训练，填充 missing.xlsx")
    print("=" * 70)
    
    # ==================== 第1步：检查文件是否存在 ====================
    print("\n【第1步】检查文件")
    print("-" * 70)
    
    train_file = 'new_remain.xlsx'
    target_file = 'bigSet_fixed.xlsx'
    
    if not os.path.exists(train_file):
        print(f"✗ 未找到训练文件: {train_file}")
        return
    
    if not os.path.exists(target_file):
        print(f"✗ 未找到目标文件: {target_file}")
        return
    
    print(f"✓ 找到训练文件: {train_file}")
    print(f"✓ 找到目标文件: {target_file}")
    
    # ==================== 第2步：加载训练数据 ====================
    print("\n【第2步】加载训练数据")
    print("-" * 70)
    
    df_train = pd.read_excel(train_file)
    print(f"\n✓ 成功加载训练数据")
    print(f"  数据形状: {df_train.shape[0]} 行 x {df_train.shape[1]} 列")
    print(f"\n前5行数据:")
    print(df_train.head())
    
    # 检查训练数据的缺失值
    train_missing = df_train.isnull().sum()
    total_train_missing = train_missing.sum()
    
    print(f"\n训练数据的缺失值统计:")
    print(train_missing[train_missing > 0])
    print(f"总缺失值数量: {total_train_missing}")
    
    # 分离数值列和非数值列
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"\n数值列: {numeric_cols}")
    print(f"非数值列: {non_numeric_cols}")
    
    # ==================== 第3步：加载目标数据 ====================
    print("\n\n【第3步】加载需要填充的目标数据")
    print("-" * 70)
    
    df_target = pd.read_excel(target_file)
    print(f"\n✓ 成功加载目标数据")
    print(f"  数据形状: {df_target.shape[0]} 行 x {df_target.shape[1]} 列")
    print(f"\n前5行数据:")
    print(df_target.head())
    
    # 检查目标数据的缺失值
    target_missing = df_target.isnull().sum()
    total_target_missing = target_missing.sum()
    
    print(f"\n目标数据的缺失值统计:")
    print(target_missing[target_missing > 0])
    print(f"总缺失值数量: {total_target_missing}")
    
    if total_target_missing == 0:
        print("\n✓ 目标数据没有缺失值，无需填充！")
        return
    
    # 检查目标数据的列结构
    target_numeric_cols = df_target.select_dtypes(include=[np.number]).columns.tolist()
    target_non_numeric_cols = df_target.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"\n目标数据的数值列: {target_numeric_cols}")
    print(f"目标数据的非数值列: {target_non_numeric_cols}")
    
    # 验证列结构是否匹配
    if set(numeric_cols) != set(target_numeric_cols):
        print(f"\n⚠ 警告：训练数据和目标数据的数值列不完全匹配")
        print(f"  训练数据数值列: {numeric_cols}")
        print(f"  目标数据数值列: {target_numeric_cols}")
        
        # 找出共同的数值列
        common_cols = list(set(numeric_cols) & set(target_numeric_cols))
        if common_cols:
            print(f"  共同列: {common_cols}")
            print(f"  将只填充共同的数值列")
            numeric_cols_to_use = common_cols
        else:
            print("  ✗ 没有共同的数值列，无法填充")
            return
    else:
        print("✓ 列结构完全匹配")
        numeric_cols_to_use = numeric_cols
    
    # ==================== 第4步：训练模型 ====================
    print("\n\n【第4步】使用训练数据训练模型")
    print("-" * 70)
    
    method = 'hyperimpute'  # 可以改为 'missforest', 'median' 等
    print(f"\n使用 '{method}' 方法训练...")
    
    imputer = Imputers().get(method)
    
    # 提取训练数据的数值列
    df_train_numeric = df_train[numeric_cols_to_use].copy()
    
    print("正在训练模型（这可能需要一些时间）...")
    imputer.fit(df_train_numeric)
    print("✓ 模型训练完成！")
    print("  模型已经从训练数据中学习了各列之间的关系")
    
    # ==================== 第5步：用训练好的模型填充目标数据 ====================
    print("\n【第5步】用训练好的模型填充目标数据")
    print("-" * 70)
    
    print("正在填充目标数据...")
    df_target_numeric = df_target[numeric_cols_to_use].copy()
    df_target_numeric_filled = imputer.transform(df_target_numeric)
    print("✓ 目标数据填充完成！")
    
    # 验证填充结果
    remaining_missing = df_target_numeric_filled.isnull().sum().sum()
    if remaining_missing == 0:
        print("✓ 所有数值列的缺失值已成功填充")
    else:
        print(f"⚠ 仍有 {remaining_missing} 个缺失值未填充")
    
    # 合并数值列和非数值列
    df_target_filled = df_target.copy()
    df_target_filled[numeric_cols_to_use] = df_target_numeric_filled
    
    print(f"\n填充后的前5行数据:")
    print(df_target_filled.head())
    
    # ==================== 第7步：保存结果 ====================
    print("\n【第7步】保存填充后的结果")
    print("-" * 70)
    
    output_file = 'new_filled.xlsx'
    df_target_filled.to_excel(output_file, index=False)
    print(f"✓ 填充后的数据已保存到: {output_file}")
    
    # ==================== 总结 ====================
    print("\n\n" + "=" * 70)
    print("✅ 完成！")
    print("=" * 70)
    print(f"""
 处理总结：
- 训练文件: {train_file} ({df_train.shape[0]} 行)
- 目标文件: {target_file} ({df_target.shape[0]} 行)
- 使用方法: {method}
- 填充列: {numeric_cols_to_use}
- 输出文件: {output_file}
    """)


if __name__ == '__main__':
    train_and_fill_custom_files()

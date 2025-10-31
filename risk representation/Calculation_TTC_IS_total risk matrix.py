import pandas as pd 
import numpy as np

# 读取CSV文件，并处理可能的编码问题
def load_data(file_path):
    # 尝试使用不同的编码格式读取文件
    encodings = ['utf-8', 'gbk', 'ISO-8859-1']
    
    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            print(f"文件成功使用 {encoding} 编码读取。")
            return data
        except UnicodeDecodeError:
            print(f"使用 {encoding} 编码读取失败，尝试其他编码。")
    raise ValueError("无法识别文件编码格式。")

# 数据清理：去除单位并转换为数值
def clean_data(data):
    # 去除单位，保留数值部分
    data['xDistance'] = data['xDistance'].str.replace('m', '').astype(float)
    data['yDistance'] = data['yDistance'].str.replace('m', '').astype(float)
    data['xSpeed'] = data['xSpeed'].str.replace('m/s', '').astype(float)
    data['ySpeed'] = data['ySpeed'].str.replace('m/s', '').astype(float)
    data['Vx平均值'] = data['Vx平均值'].astype(float)
    data['Vy平均值'] = data['Vy平均值'].astype(float)
    return data

# 插值处理：正后方没有雷达，使用左右雷达的数据进行插值
def interpolate_missing_data(data):
    # 假设 'type' 为雷达的方向，'正后'需要插值
    for index, row in data.iterrows():
        if row['type'] == '正后':
            left_data = data[(data['type'] == '左后') & (data['time'] == row['time'])]
            right_data = data[(data['type'] == '右后') & (data['time'] == row['time'])]

            if not left_data.empty and not right_data.empty:
                # 对缺失的正后方数据进行插值填充
                data.at[index, 'xDistance'] = (left_data['xDistance'].values[0] + right_data['xDistance'].values[0]) / 2
                data.at[index, 'yDistance'] = (left_data['yDistance'].values[0] + right_data['yDistance'].values[0]) / 2
    return data

# 根据九宫格划分规则分配网格位置
def assign_to_grid(data):
    grid_positions = []
    
    for index, row in data.iterrows():
        xDist = row['xDistance']
        yDist = row['yDistance']
        
        # 通过横向与纵向距离来划分九宫格位置
        if -7.5 <= xDist < -2.5 and 3.37 <= yDist <= 8.37:
            grid_positions.append(1)  # 左前
        elif -2.5 <= xDist < 2.5 and 3.37 <= yDist <= 8.37:
            grid_positions.append(2)  # 前方
        elif 2.5 <= xDist < 7.5 and 3.37 <= yDist <= 8.37:
            grid_positions.append(3)  # 右前
        elif -7.5 <= xDist < -2.5 and -1.26 <= yDist < 3.37:
            grid_positions.append(4)  # 左方
        elif -2.5 <= xDist < 2.5 and -1.26 <= yDist < 3.37:
            grid_positions.append(5)  # 中间
        elif 2.5 <= xDist < 7.5 and -1.26 <= yDist < 3.37:
            grid_positions.append(6)  # 右方
        elif -7.5 <= xDist < -2.5 and -6.25 <= yDist < -1.26:
            grid_positions.append(7)  # 左后
        elif -2.5 <= xDist < 2.5 and -6.25 <= yDist < -1.26:
            grid_positions.append(8)  # 后方
        elif 2.5 <= xDist < 7.5 and -6.25 <= yDist < -1.26:
            grid_positions.append(9)  # 右后
        else:
            grid_positions.append(-1)  # 如果不符合任何条件，设置为-1

    # 确保 grid_positions 的长度和数据的行数一致
    while len(grid_positions) < len(data):
        grid_positions.append(-1)

    # 处理被错误划分到中间网格（网格5）的车辆
    for i, position in enumerate(grid_positions):
        if position == 5:
            xDist = data.at[i, 'xDistance']
            yDist = data.at[i, 'yDistance']
            
            # 将被划分到网格5的车辆重新分配到周围的网格
            if xDist < -2.5:
                grid_positions[i] = 4  # 重新划分为左方
            elif xDist > 2.5:
                grid_positions[i] = 6  # 重新划分为右方
            elif yDist > 3.37:
                grid_positions[i] = 2  # 重新划分为前方
            elif yDist < -6.25:
                grid_positions[i] = 8  # 重新划分为后方

    data['grid_position'] = grid_positions
    return data

# 计算TTC、IS和总风险矩阵
def calculate_ttc_is(data):
    ttc_matrix = []
    is_matrix = []
    total_risk_matrix = []
    
    for index, row in data.iterrows():
        # 获取目标车辆到自车的距离和相对速度
        xDist = row['xDistance']
        yDist = row['yDistance']
        d = np.sqrt(xDist**2 + yDist**2)  # 计算距离
        
        # 计算目标车辆的相对速度（v_rel），取绝对值
        v_self = np.sqrt(abs(row['Vx平均值'])**2 + abs(row['Vy平均值'])**2)  # 自车速度（取绝对值）
        v_rel = abs(v_self - abs(row['xSpeed']))  # 相对速度（取绝对值）
        
        # 计算TTC，避免相对速度为零时除以零的情况
        if v_rel == 0:
            ttc = 0.5  # 设置TTC下限为0.5秒
        else:
            ttc = d / (v_rel + 0.001)  # 计算TTC
        
        # 计算IS，假设预期安全距离（RSS）是自车的速度的平方根
        rss = 2 * np.sqrt(v_self)  # 这里使用2倍的速度平方根作为安全距离
        is_value = d / rss  # 计算IS
        
        # 计算总风险矩阵
        total_risk = 0.5 * ttc + 0.5 * is_value
        
        ttc_matrix.append(ttc)
        is_matrix.append(is_value)
        total_risk_matrix.append(total_risk)
    
    data['TTC_matrix'] = ttc_matrix
    data['IS_matrix'] = is_matrix
    data['Total_Risk_matrix'] = total_risk_matrix
    
    return data

# 输出文件设计：包含TTC、IS、总风险矩阵
def design_output_file(data):
    # 设计输出表头和数据结构
    data_output = data[['time', 'type', 'id', 'xSpeed', 'ySpeed', 'xDistance', 'yDistance', 'Vx平均值', 'Vy平均值', 'grid_position', 'TTC_matrix', 'IS_matrix', 'Total_Risk_matrix']]
    return data_output

# 主程序
def main(file_path):
    # 1. 读取数据
    data = load_data(file_path)
    
    # 2. 清理数据
    data = clean_data(data)
    
    # 3. 填充正后方的缺失数据
    data = interpolate_missing_data(data)
    
    # 4. 将每个车辆分配到九宫格
    data = assign_to_grid(data)
    
    # 5. 计算TTC、IS和总风险矩阵
    data = calculate_ttc_is(data)
    
    # 6. 设计输出文件格式
    data_output = design_output_file(data)
    
    # 7. 输出数据到CSV文件
    data_output.to_csv(r"G:\毕业论文\毕业论文初稿\毕业论文代码\九宫格划分\数据代码实验输出 .csv", index=False)
    print("数据处理完成")

# 调用主程序
if __name__ == "__main__":
    # 请替换为实际的文件路径
    file_path = r"G:\毕业论文\毕业论文初稿\毕业论文代码\九宫格划分\数据代码实验 .csv"  # 数据文件路径
    main(file_path)

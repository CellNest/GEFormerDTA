import csv
import json
# 1. 从.csv文件中读取数据并获取蛋白质序列和行索引
csv_filename = './split/kiba_test_setting_1_origin.csv'
data = []

with open(csv_filename, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        data.append(row)

# 2. 从字典格式的.txt文件中读取键值对
txt_filename = './split/proteins.txt'
protein_dict = {}


with open(txt_filename, 'r') as txt_file:
    txt_data = txt_file.read()
    # 解析JSON数据
    dic = json.loads(txt_data)
    for target_name, seq in dic.items():
        protein_dict[target_name] = seq
# 3. 遍历.csv文件中的每一行并更新数据
        for row in data:
            protein_sequence = row['target_sequence']

            # 4. 检查蛋白质序列是否存在于字典中
            if protein_sequence in protein_dict[target_name]:
                # 5. 如果找到匹配的蛋白质序列，将值插入到.csv文件中
                row['target_name'] = target_name

# 将更新后的数据保存回.csv文件
output_filename = 'kiba_test_setting_1.csv'

with open(output_filename, 'w', newline='') as output_file:
    fieldnames = data[0].keys()
    csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_writer.writerows(data)

import json

# 定义要合并的文件名
file_names = ['/home/qmli/InternLM-XComposer-main/data_eccv/train/vqa_anno/converted_data_general_perception.json', '/home/qmli/InternLM-XComposer-main/data_eccv/train/vqa_anno/converted_data_region_perception.json', '/home/qmli/InternLM-XComposer-main/data_eccv/train/vqa_anno/converted_data_driving_suggestion.json']

# 初始化一个空列表来存储合并后的数据
merged_data = []

# 遍历每个文件并读取数据，然后追加到merged_data列表中
for file_name in file_names:
    with open(file_name, 'r') as file:
        data = json.load(file)  # 假设每个文件的内容是一个列表
        merged_data.extend(data)  # 将当前文件的数据添加到总列表中

# 将合并后的数据写入到一个新的JSON文件
output_file = 'train_val_merge_data.json'
with open(output_file, 'w') as file:
    json.dump(merged_data, file, indent=4)  # 使用indent参数可以使输出的JSON更加易读

print(f"合并完成，结果已保存至 {output_file}")
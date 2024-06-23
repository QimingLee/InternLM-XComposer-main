import json

def convert_jsonl_to_conversations(jsonl_file_path):
    conversations_list = []
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            # 构建新的数据结构
            conversation_entry = {
                "id": str(data["question_id"]),
                "image": [data["image"]],
                "conversations": [
                    {
                        "from": "user",
                        "value": data["question"]
                    },
                    {
                        "from": "assistant",
                        "value": data["answer"]
                    }
                ]
            }
            conversations_list.append(conversation_entry)
    
    return conversations_list

# 使用示例：请将下面的'path_to_your_jsonl_file.jsonl'替换为您的实际文件路径
jsonl_file_path = '/home/qmli/InternLM-XComposer-main/data_eccv/train/vqa_anno/general_perception.jsonl' #task1
# jsonl_file_path = '/home/qmli/InternLM-XComposer-main/data_eccv/train/vqa_anno/region_perception.jsonl' #task2
# jsonl_file_path = '/home/qmli/InternLM-XComposer-main/data_eccv/train/vqa_anno/driving_suggestion.jsonl' #task3
converted_data = convert_jsonl_to_conversations(jsonl_file_path)

# 保存转换后的数据到新的JSON文件（可选）
output_file_path = 'converted_data_general_perception.json'
# output_file_path = 'converted_data_region_perception.json'
# output_file_path = 'converted_data_driving_suggestion.json'
with open(output_file_path, 'w') as output_file:
    json.dump(converted_data, output_file, indent=4)

print("Conversion completed and saved to:", output_file_path)
import torch
import json
from transformers import AutoModel, AutoTokenizer
from peft import AutoPeftModelForCausalLM
torch.set_grad_enabled(False)

# init model and tokenizer
# model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True).cuda().eval()

model = AutoPeftModelForCausalLM.from_pretrained(
    # path to the output directory
    '/home/qmli/InternLM-XComposer-main/finetune/output/finetune',
    device_map="auto",
    trust_remote_code=True
).eval()
tokenizer = AutoTokenizer.from_pretrained('/home/qmli/models/internlm-xcomposer2-vl-7b', trust_remote_code=True)

# text = '<ImageHere>Please describe this image in detail.'
# image = '/home/qmli/InternLM-XComposer-main/examples/image1.webp'
# with torch.cuda.amp.autocast():
#   response, _ = model.chat(tokenizer, query=text, image=image, history=[], do_sample=False)
# print(response)

input_file = '/home/qmli/InternLM-XComposer-main/data_eccv/test/general_perception.jsonl'
output_file = 'general_perception_answer.jsonl'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # 解析当前行的JSON数据
        data = json.loads(line)
        
        # 提取信息
        question_id = data['question_id']
        image_path = data['image']
        question_text = data['question']
        
        # 假设这是从模型获取的响应
        with torch.cuda.amp.autocast():
            model_response, _ = model.chat(tokenizer, query=question_text, image=image_path, history=[], do_sample=False)
        
        # 在数据中增加answer字段
        data['answer'] = model_response
        
        # 将更新后的数据写回新的jsonl文件
        outfile.write(json.dumps(data) + '\n')

print("处理完成，已将答案添加至每个条目并保存至", output_file)
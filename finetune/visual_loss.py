import json
import re
import matplotlib.pyplot as plt

def extract_metrics_from_log(file_path):
    pattern = r"\{.*?'loss':\s*[\d.]+,\s*'learning_rate':\s*[\d.e-]+,\s*'epoch':\s*[\d.]+\}"
    metrics = []

    with open(file_path, 'r') as file:
        content = file.read()
        matches = re.findall(pattern, content)
        for match in matches:
            # 将单引号替换为双引号以符合JSON规范
            json_str = match.replace("'", "\"")
            try:
                data = json.loads(json_str)
                metrics.append({'epoch': float(data['epoch']), 'loss': float(data['loss'])})
            except json.JSONDecodeError as e:
                print(f"Failed to decode after replacement: {json_str}. Error: {e}")
    
    return metrics

def visualize_losses(metrics):
    epochs = [metric['epoch'] for metric in metrics]
    losses = [metric['loss'] for metric in metrics]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, label='Training Loss', color='blue')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')

# 示例使用
file_path = '/home/qmli/InternLM-XComposer-main/finetune/slurm-152806.out'  # 替换为您的日志文件路径
extracted_metrics = extract_metrics_from_log(file_path)
visualize_losses(extracted_metrics)


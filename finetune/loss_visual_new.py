import json
import re
import matplotlib.pyplot as plt

def extract_metrics_from_log(file_path, samples_per_epoch):
    pattern = r"\{.*?'loss':\s*[\d.]+,\s*'learning_rate':\s*[\d.e-]+,\s*'epoch':\s*[\d.]+\}"
    metrics = []

    with open(file_path, 'r') as file:
        content = file.read()
        matches = re.findall(pattern, content)
        for match in matches:
            json_str = match.replace("'", "\"")
            try:
                data = json.loads(json_str)
                epoch_samples = int(data['epoch']) * samples_per_epoch  # 计算每个记录点的样本数
                metrics.append({'samples': epoch_samples, 'loss': float(data['loss'])})
            except json.JSONDecodeError as e:
                print(f"Failed to decode after replacement: {json_str}. Error: {e}")
    
    return metrics

def visualize_losses(metrics):
    samples = [metric['samples'] for metric in metrics]
    losses = [metric['loss'] for metric in metrics]

    plt.figure(figsize=(10, 5))
    plt.plot(samples, losses, label='Training Loss', color='blue')
    plt.title('Loss over Samples')
    plt.xlabel('Samples Seen')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('loss_samples_lower_plot.png', dpi=300, bbox_inches='tight')

# 示例使用
file_path = '/home/qmli/InternLM-XComposer-main/finetune/slurm-153588.out'  # 替换为您的日志文件路径
# 假设每个epoch处理了1000个样本
samples_per_epoch = 4875
extracted_metrics = extract_metrics_from_log(file_path, samples_per_epoch)
visualize_losses(extracted_metrics)
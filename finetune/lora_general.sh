#!/bin/bash
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 5:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH -w gpu09                        # 指定运行作业的节点是 gpu06，若不填写系统自动分配节点
#SBATCH --gres=gpu:a100-sxm4-80gb:1               # 申请 4 卡 A100 80GB
#SBATCH --mem=100G

source ~/.bashrc
# source activate base

sh /home/qmli/InternLM-XComposer-main/finetune/finetune_lora_general.sh
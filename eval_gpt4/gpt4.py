import openai
import json

# 初始化OpenAI API
openai.api_key = 'your-api-key'

# 评测标准prompt，这里简化处理，直接在每次请求中包含评分标准
评测标准_prompt_template =  "You are an impartial judge tasked with evaluating the quality of predicted text provided by autonomous driving AI assistant.You will compare this prediction to a reference text, focusing on the description of objects that influence the driving behavior of ego car, and the explanation of why these objects impact. Your evaluation criteria should include accuracy(checking if the predicted text correctly identifies objects mentioned the reference text), suppression hallucination(ensuring that objects not mentioned in the reference text are not erroneously included in the predicted text), correlation(sessing if the reasons for the objects' impact on the ego car's driving behavior are consistent between the reference and predicted text). Identify and correct any mistakes. Be as objective as possible. Do not allow the length of the predicted text to influence your evaluation. After providing your short explanation, you must rate the response on a scale from 1 to 100 by strictly following this format: [[rating]], for example: Rating: [[10.0]]."


# 评估函数
def evaluate_answer(output, answer):
    prompt = 评测标准_prompt_template.format(output=output, answer=answer)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0,
    )
    try:
        score_str, reason = response.choices[0].text.strip().split(' ', 1)
        score = int(score_str)
        return score, reason
    except ValueError:
        print("未能成功解析评分，返回默认评分为5，原因：'无法解析'")
        return 5, "无法解析"

# 执行评估
scores = []
for item in data:
    score, reason = evaluate_answer(item["answer"], item["answer"])  # 这里假设“output”也是同样的内容，根据实际情况调整
    scores.append(score)
    print(f"回答：'{item['answer']}'，评分：{score}，理由：'{reason}'")

# 计算平均分
if scores:
    average_score = sum(scores) / len(scores)
    print(f"\n所有样例的平均评分是：{average_score}")
else:
    print("没有评分数据")
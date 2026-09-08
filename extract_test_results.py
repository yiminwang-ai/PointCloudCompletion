import re

# 从文本文件中读取日志内容
with open('train_log.txt', 'r') as file:
    log_text = file.read()

# 使用正则表达式匹配 "test result: tensor(数字)"
test_results = re.findall(r'test result: tensor\((.*?)\)', log_text)

# 将匹配结果转换为浮点数
test_results_float = [float(result) for result in test_results]

# 打印提取的测试结果
print(test_results_float)

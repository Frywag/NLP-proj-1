from Lexicon import Lexicon
from TextAnalyzer import TextAnalyzer
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from Lexicon import Lexicon
from TextAnalyzer import TextAnalyzer
from Comparator import Comparator
import matplotlib.pyplot as plt


# # 假设你已经创建了 Lexicon 对象 lexicon

lexicon = Lexicon("D:/UserDesktop/55/GJCXSJ/concrete_dictionary.xlsx")
#
# 创建 TextAnalyzer 对象
analyzer = TextAnalyzer(lexicon)
#
# # # 加载文本
# # analyzer.load_text("This is the second text.")
# 或者从文件加载
analyzer.load_text_from_file("D:/UserDesktop/55/Text1.txt")
#
#
# 预处理
analyzer.preprocess()
#
# 测试手动计算mattr
analyzer.calculate_mattr(window_size=50)
#
# # 计算具体性
# analyzer.calculate_concreteness()

# 评估难度
difficulty_metrics = analyzer.calculate_difficulty_metrics()
print(f"Difficulty metrics: {difficulty_metrics}")
#
# # 提取关键词
# keywords = analyzer.extract_keywords()
# print(f"Keywords: {keywords}")
# #
# # 生成词云图
# analyzer.generate_wordcloud()
#
# # 抽取摘要
# summary = analyzer.extract_summary_sentences()
# print(f"Summary: {summary}")
#
# # 绘制具体性分布图
# analyzer.plot_concreteness_distribution()
#
# # 导出未登录词
# analyzer.export_unlisted_words("./unlisted_words.xlsx")

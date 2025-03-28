from Lexicon import Lexicon
# 创建 Lexicon 对象，传入词典文件路径
lexicon = Lexicon("D:/UserDesktop/55/GJCXSJ/concrete_dictionary.xlsx")  # 替换为你的词典文件路径

# 加载自定义词典
lexicon.load_custom_lexicon("D:/UserDesktop/55/GJCXSJ/custom_dictionary.xlsx")
# 查询词汇的具体性评分
score = lexicon.lookup("example")
if score is not None:
    print(f"The concreteness score of 'example' is: {score}")
else:
    print("'example' not found in the lexicon.")

# 添加自定义词汇
lexicon.add_custom_word("d_word", 6.5)

# 保存自定义词典
lexicon.save_custom_lexicon("D:/UserDesktop/55/GJCXSJ/custom_dictionary.xlsx")

# # 清空自定义词典
# lexicon.clear_custom_lexicon("D:/UserDesktop/55/GJCXSJ/custom_dictionary.xlsx")


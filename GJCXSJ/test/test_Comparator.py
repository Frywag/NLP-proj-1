from Comparator import Comparator
from TextAnalyzer import TextAnalyzer
from Lexicon import Lexicon


lexicon = Lexicon("D:/UserDesktop/55/GJCXSJ/concrete_dictionary.xlsx")
# 假设你已经创建了两个 TextAnalyzer 对象 analyzer1 和 analyzer2，并加载了文本、进行了分析
analyzer1 = TextAnalyzer(lexicon)
analyzer1.load_text_from_file("D:/UserDesktop/55/Text1.txt")
analyzer1.preprocess()
analyzer1.calculate_concreteness()

analyzer2 = TextAnalyzer(lexicon)
analyzer2.load_text_from_file("D:/UserDesktop/55/Text2.txt")
analyzer2.preprocess()
analyzer2.calculate_concreteness()

# 创建 Comparator 对象
comparator = Comparator(analyzer1, analyzer2)

# 比较指标
comparison_results = comparator.compare_metrics()
print(comparison_results)

# 生成柱状图
comparator.plot_comparison()
# 如果要生成雷达图（目前不支持，会打印提示信息）
# comparator.plot_comparison(type='radar')
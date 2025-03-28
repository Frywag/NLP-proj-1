import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from TextAnalyzer import TextAnalyzer
from Lexicon import Lexicon

class Comparator:
    """
    文本比较类，用于比较两个 TextAnalyzer 对象的分析结果。

    Attributes:
        analyzer1 (TextAnalyzer): 第一个文本的 TextAnalyzer 对象。
        analyzer2 (TextAnalyzer): 第二个文本的 TextAnalyzer 对象。
    """

    def __init__(self, analyzer1, analyzer2):
        """
        构造函数，初始化 Comparator 对象。

        Args:
            analyzer1 (TextAnalyzer): 第一个文本的 TextAnalyzer 对象。
            analyzer2 (TextAnalyzer): 第二个文本的 TextAnalyzer 对象。
        """
        self.analyzer1 = analyzer1
        self.analyzer2 = analyzer2

    def compare_metrics(self):
        """
        计算并比较两个文本的各项指标。

        Returns:
            dict: 包含比较结果的字典。
        """

        # 确保两个 TextAnalyzer 对象都已经进行了分析
        if self.analyzer1.avg_concreteness is None:
            self.analyzer1.calculate_avg_concreteness()
        if self.analyzer2.avg_concreteness is None:
            self.analyzer2.calculate_avg_concreteness()


        metrics1 = self.analyzer1.calculate_difficulty_metrics()
        metrics2 = self.analyzer2.calculate_difficulty_metrics()

        comparison = {
            'metric': ['Average Concreteness', 'Word Density', 'MATTR'],
            'Text 1': [metrics1['avg_concreteness'], metrics1['word_density'], metrics1['mattr']],
            'Text 2': [metrics2['avg_concreteness'], metrics2['word_density'], metrics2['mattr']],
        }
        # 创建 DataFrame
        df = pd.DataFrame(comparison)
        # 设置 metric 列为索引
        df = df.set_index('metric')

        # 处理 None 值, 转换为字符串 "N/A", 为了后续能正常显示表格
        df = df.applymap(lambda x: 'N/A' if x is None else x)

        return df

    def plot_comparison(self, type='bar'):
        """
        生成对比图表。

        Args:
            type (str): 图表类型，'bar' (柱状图) 或 'radar' (雷达图)。
                          目前仅支持 bar 图, 后续可扩展.
        """
        metrics1 = self.analyzer1.calculate_difficulty_metrics()
        metrics2 = self.analyzer2.calculate_difficulty_metrics()

        # 提取指标和值
        metrics = ['avg_concreteness', 'word_density', 'mattr']
        values1 = [metrics1[m] for m in metrics]
        values2 = [metrics2[m] for m in metrics]

        # 处理 None 值, 替换为 0 或一个小的非零值，避免绘图错误
        values1 = [0 if v is None else v for v in values1]
        values2 = [0 if v is None else v for v in values2]

        if type == 'bar':
            x = np.arange(len(metrics))
            width = 0.35

            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width/2, values1, width, label='Text 1')
            rects2 = ax.bar(x + width/2, values2, width, label='Text 2')

            ax.set_ylabel('Scores')
            ax.set_title('Comparison of Text Metrics')
            ax.set_xticks(x)
            ax.set_xticklabels(['Avg. Conc.', 'Word Density', 'MATTR'])
            ax.legend()

            # 添加数值标签
            self.autolabel(rects1,ax)
            self.autolabel(rects2,ax)


            fig.tight_layout()
            plt.show()

        elif type == 'radar': # 暂时不支持, 会报错
            print("雷达图暂不支持")
            return
            # # 雷达图代码 (需要更多处理，例如角度、标签等)
            # fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            #
            # angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
            # angles = np.concatenate((angles, [angles[0]]))
            #
            # values1 = np.concatenate((values1, [values1[0]]))
            # values2 = np.concatenate((values2, [values2[0]]))
            #
            # ax.plot(angles, values1, 'o-', linewidth=2, label='Text 1')
            # ax.fill(angles, values1, alpha=0.25)
            # ax.plot(angles, values2, 'o-', linewidth=2, label='Text 2')
            # ax.fill(angles, values2, alpha=0.25)
            #
            # ax.set_thetagrids(angles * 180/np.pi, metrics)
            # ax.set_title('Comparison of Text Metrics')
            # ax.grid(True)
            # ax.legend()
            # plt.show()

        else:
            print("不支持的图表类型。")

    def autolabel(self,rects,ax):
        """
        在每个柱形条上方添加数值标签。
        """
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
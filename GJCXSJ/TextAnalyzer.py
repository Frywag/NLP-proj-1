import spacy
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


class TextAnalyzer:
    """
    文本处理与分析类。

    Attributes:
        lexicon (Lexicon): 词典对象。
        text (str): 待分析文本。
        tokens (list): 分词后的 token 列表。
        pos_tags (list): 词性标注结果 (token, POS tag) 的列表。
        concreteness_scores (list): 每个 token 的具体性评分列表。
        unlisted_words (list): 未登录词列表。
        avg_concreteness (float): 平均具体性得分。
    """

    def __init__(self, lexicon):
        """
        构造函数，初始化 TextAnalyzer 对象。

        Args:
            lexicon (Lexicon): Lexicon 对象。
        """
        self.lexicon = lexicon
        self.text = ""
        self.tokens = []
        self.pos_tags = []
        self.concreteness_scores = []
        self.unlisted_words = []
        self.avg_concreteness = None
        self.nlp = spacy.load("en_core_web_sm")

    def load_text(self, text):
        """
        直接加载文本字符串。

        Args:
            text (str): 待分析的文本。
        """
        self.text = text
        # 重置分析结果
        self.reset_analysis()


    def load_text_from_file(self, filepath):
        """
        从文件加载文本。

        Args:
            filepath (str): 文本文件的路径。
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:  # 指定编码为 utf-8
                self.text = file.read()
            # 重置分析结果
            self.reset_analysis()

        except FileNotFoundError:
            print(f"错误：文件未找到：{filepath}")
            raise
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            raise

    def reset_analysis(self):
        """
        重置分析结果（除了 lexicon）。
        """
        self.tokens = []
        self.pos_tags = []
        self.concreteness_scores = []
        self.unlisted_words = []
        self.avg_concreteness = None

    def preprocess(self):
        """
        文本预处理：分词、词形还原、词性标注（使用 spaCy），
        去除自定义停用词、标点、多余空格和数字，以及重复的 token。
        """
        doc = self.nlp(self.text)
        custom_stop_words = ['this', 'be', 'the', 'a', 'of', 'at', 'to']  # 自定义停用词列表

        filtered_tokens = []
        filtered_pos_tags = []

        for token in doc:
            # 过滤条件：
            # 1. 不是标点符号
            # 2. 词形还原后的小写形式不在自定义停用词列表中
            # 3. 不是纯空白字符
            # 4. 不是纯数字 (根据需要选择是否注释掉)
            lemma = token.lemma_.lower()  # 获取 lemma 并转换为小写

            if (not token.is_punct and
                    lemma not in custom_stop_words and
                    not lemma.isspace() and  # 过滤空白字符
                    not lemma.isdigit()):  # 过滤纯数字 (如果需要)
                filtered_tokens.append(lemma)
                filtered_pos_tags.append((lemma, token.pos_))

                # 去除重复的 token
            self.tokens = list(dict.fromkeys(filtered_tokens))  # 使用字典保持原始顺序
            self.pos_tags = [(token, pos) for token, pos in filtered_pos_tags if token in self.tokens]


    def calculate_concreteness(self):
        """
        计算每个 token 的具体性评分（调用 lexicon.lookup()）。
        """
        self.concreteness_scores = []  # 清空之前的评分
        self.unlisted_words = []  # 清空之前的未登录词列表

        for token in self.tokens:
            score = self.lexicon.lookup(token)
            if score is not None:
                self.concreteness_scores.append(score)
            else:
                self.concreteness_scores.append(None)  # 未找到的词，评分设为 None
                self.unlisted_words.append(token)


    def calculate_avg_concreteness(self):
        """
        计算平均具体性得分。
        """
        valid_scores = [score for score in self.concreteness_scores if score is not None]
        if valid_scores:
            self.avg_concreteness = sum(valid_scores) / len(valid_scores)
        else:
            self.avg_concreteness = None  # 如果没有有效的评分，平均值设为 None


    def get_unlisted_words(self):
        """
        获取未登录词列表。

        Returns:
            list: 未登录词列表。
        """
        return self.unlisted_words

    def add_temp_concreteness(self, word, score):
        """
        为未登录词临时添加评分（仅影响本次分析）。

        Args:
            word (str): 要添加评分的词。
            score (float): 具体性评分。
        """
        if word in self.unlisted_words:
            index = self.unlisted_words.index(word)
            if index < len(self.concreteness_scores):
                self.concreteness_scores[index] = score
                print(f"已为未登录词 '{word}' 临时添加评分：{score}")
        else:
            print(f"错误：未找到未登录词 '{word}'")
        # 重新计算平均具体性
        self.calculate_avg_concreteness()

    def calculate_word_density(self):
        """
        计算词汇密度。

        公式：(实词总数 / 总词数) × 100%
        实词包括：名词、动词、形容词、副词

        Returns:
            float: 词汇密度（百分比）。
        """
        if not self.tokens: # 避免除以0
            return 0.0

        content_word_tags = ['NOUN', 'VERB', 'ADJ', 'ADV']
        content_word_count = sum(1 for _, pos in self.pos_tags if pos in content_word_tags)
        print(f"content_word_count: {content_word_count},  total tokens: {len(self.tokens)}")  # 调试输出

        return (content_word_count / len(self.tokens)) * 100

    # def calculate_mattr(self):
    #   """
    #   计算文本的平均移动类型-标记比率（MATTR），这是一种词汇多样性的度量。
    #
    #   返回值：
    #       float：MATTR 值。如果文本没有足够的标记来计算 MATTR，则返回 None。
    #   """
    #   # 确保文本已被处理为 spaCy 文档
    #   doc = self.nlp(self.text)
    #   # 使用 textdescriptives 计算 MATTR
    #   try:
    #       metrics = td.extract_metrics(doc, metrics=["diversity"])
    #       return metrics["diversity_mattr"]
    #   except ValueError as e:
    #     print("错误：文本可能太短而无法计算MATTR")
    #     return None

    def calculate_mattr(self, window_size=50):
        """
        手动计算 MATTR。
        """
        if len(self.tokens) < window_size:
            print("错误：文本太短，无法使用手动计算和指定窗口大小计算 MATTR")
            return None

        ttrs = []
        for i in range(0, len(self.tokens) - window_size + 1):
            window = self.tokens[i:i + window_size]
            types = len(Counter(window))  # 使用 Counter 计算不同类型的数量
            ttr = types / window_size
            ttrs.append(ttr)

        return sum(ttrs) / len(ttrs)

    def analyze_mattr(self, window_sizes=[50, 100]):
        """分析不同窗口大小下MATTR的计算情况"""
        self.preprocess()  # 先进行预处理

        print("文本长度（预处理后）:", len(self.tokens), "个词")

        print("\n手动计算 MATTR:")
        for size in window_sizes:
            mattr_manual = self.calculate_mattr(window_size=size)
            if mattr_manual is not None:
                print(f"  窗口大小 {size}: MATTR = {mattr_manual:.4f}")

    def calculate_difficulty_metrics(self):
        """
        整合所有难度指标。

        Returns:
            dict: 包含各项难度指标的字典。
        """
        return {
            'avg_concreteness': self.avg_concreteness,
            'word_density': self.calculate_word_density(),
            'mattr': self.calculate_mattr(),
        }

    def extract_keywords(self, n=10):
        """
        提取关键词（按具体性评分排序）。

        Args:
            n (int): 要提取的关键词数量。

        Returns:
            list: 包含关键词及其具体性评分的列表，按评分降序排列。
        """
        # 创建一个字典，将 token 映射到它们的具体性评分
        token_score_map = dict(zip(self.tokens, self.concreteness_scores))

        # 过滤掉评分为 None 的 token
        valid_tokens = {token: score for token, score in token_score_map.items() if score is not None}

        # 按照具体性评分对 token 进行降序排序
        sorted_tokens = sorted(valid_tokens.items(), key=lambda item: item[1], reverse=True)

        # 返回前 n 个关键词及其评分
        return sorted_tokens[:n]

    def generate_wordcloud(self, width=800, height=400,
                           background_color='white',
                           font_path='D:/UserDesktop/55/GJCXSJ/SimHei.ttf'):
        """
        生成词云图。

        Args:
            width (int): 词云图的宽度。
            height (int): 词云图的高度。
            background_color (str): 词云图的背景颜色。
            font_path (str): 字体文件路径。
        """

        # 1. 计算 TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')  # 去除英文停用词, 因为已经在预处理中去除了,所以这里可以不用
        tfidf_matrix = vectorizer.fit_transform([self.text])  # 使用原始文本
        feature_names = vectorizer.vocabulary_
        feature_names = list(feature_names.keys())
        tfidf_scores = tfidf_matrix.toarray().flatten()

        # 2. 创建一个字典，将 token 映射到它们的 TF-IDF 评分
        tfidf_dict = dict(zip(feature_names, tfidf_scores))

        # 3. 将 TF-IDF 评分与具体性评分结合
        word_frequencies = {}
        for token in self.tokens:
            concreteness_score = self.lexicon.lookup(token)
            if concreteness_score is not None:
                # 获取 TF-IDF 评分，如果 token 不在 TF-IDF 字典中，则使用一个默认值（如 0）
                tfidf_score = tfidf_dict.get(token, 0)
                # 结合 TF-IDF 和具体性评分（例如，相乘或加权平均, 这里采用相乘）
                combined_score = concreteness_score * (tfidf_score + 1)  # +1 避免乘以0
                word_frequencies[token] = combined_score

        # 4. 生成词云
        if not word_frequencies:
            print("没有足够的数据生成词云。")
            return

        wordcloud = WordCloud(width=width, height=height, background_color=background_color,
                              font_path=font_path).generate_from_frequencies(word_frequencies)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')  # 不显示坐标轴
        plt.show()


    def extract_summary_sentences(self, n=3):
      """
      抽取式摘要（选择包含高具体性词汇的句子）。

      Args:
          n (int): 要提取的句子数量。

      Returns:
          list: 包含摘要句子的列表。
      """
      doc = self.nlp(self.text)
      sentences = list(doc.sents)

      # 为每个句子计算具体性评分
      sentence_scores = []
      for sentence in sentences:
          sentence_score = 0
          token_count = 0
          for token in sentence:
              score = self.lexicon.lookup(token.lemma_)  # 使用词形还原后的形式
              if score is not None:
                  sentence_score += score
                  token_count += 1
          # 避免除以0
          if token_count > 0:
              sentence_scores.append((sentence, sentence_score / token_count))
          else:
              sentence_scores.append((sentence, 0))  # 如果句子中没有已知具体性评分的词，则评分设为0

      # 按具体性评分对句子进行降序排序
      sorted_sentences = sorted(sentence_scores, key=lambda item: item[1], reverse=True)

      # 返回前 n 个句子
      return [sentence.text for sentence, _ in sorted_sentences[:n]]


    def plot_concreteness_distribution(self, pos_filter=None):
      """
      生成具体性得分分布图表（直方图）。
      """
      # 过滤掉 None 值，只保留有效的具体性评分
      valid_scores = []
      if pos_filter is None or pos_filter == "All":
          # 没有词性过滤，使用所有有效的具体性评分
          valid_scores = [score for score in self.concreteness_scores if score is not None]
      else:
          # 有词性过滤
          for token, score in zip(self.tokens, self.concreteness_scores):
              if score is not None:
                  # 查找 token 对应的词性
                  for word, pos in self.pos_tags:
                      if word == token:
                          if pos == pos_filter:
                              valid_scores.append(score)
                          break  # 找到匹配的 token 和词性后，跳出内层循环

      if not valid_scores:
          print(f"没有足够的数据来绘制 {pos_filter or '所有词性'} 的分布图。")
          return

      plt.figure(figsize=(8, 6))
      plt.hist(valid_scores, bins=10, color='skyblue', edgecolor='black')
      plt.xlabel('Concreteness Score')
      plt.ylabel('Frequency')
      plt.title(f'Distribution of Concreteness Scores ({pos_filter or "All POS"})')
      plt.grid(axis='y', alpha=0.75)
      plt.show()


    def export_unlisted_words(self, filepath):
      """
      将未登录词列表导出为 .xlsx 或 .csv 文件。

      Args:
          filepath (str): 导出文件的路径。
      """
      if not self.unlisted_words:
        print("没有未登录词需要导出。")
        return

      df = pd.DataFrame({'Unlisted Words': self.unlisted_words})

      try:
        if filepath.endswith('.xlsx'):
          df.to_excel(filepath, index=False)
        elif filepath.endswith('.csv'):
          df.to_csv(filepath, index=False)
        else:
          print("错误：不支持的文件格式。请使用 .xlsx 或 .csv。")
          return
        print(f"未登录词已导出到: {filepath}")
      except Exception as e:
        print(f"导出未登录词时发生错误: {e}")


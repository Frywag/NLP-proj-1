import pandas as pd

class Lexicon:
    """
    词典加载与管理类。

    Attributes:
        filepath (str): 词典文件路径 (Excel 文件)。
        data (pd.DataFrame): 加载后的词典数据。
        custom_data (pd.DataFrame): 用户自定义词典数据（初始为空）。
    """

    def __init__(self, filepath):
        """
        构造函数，初始化 Lexicon 对象。

        Args:
            filepath (str): 词典文件的路径。
        """
        self.filepath = filepath
        self.data = None  # 初始时，data 为 None
        self.custom_data = pd.DataFrame(columns=['Word', 'Bigram', 'Conc.M', 'Unknown', 'Total', 'SUBTLEX'])  # 初始化一个空的 DataFrame
        self.load_lexicon()  # 在初始化时加载词典


    def load_lexicon(self):
        """
        从文件加载词典数据到 self.data。
        """
        try:
            self.data = pd.read_excel(self.filepath)
            # 检查列名是否符合预期
            expected_columns = ['Word', 'Bigram', 'Conc.M', 'Unknown', 'Total', 'SUBTLEX']
            if not all(col in self.data.columns for col in expected_columns):
                raise ValueError("词典文件列名不符合预期")

        except FileNotFoundError:
            print(f"错误：词典文件未找到：{self.filepath}")
            # 可以选择抛出异常，或者给 self.data 一个默认值（例如一个空的 DataFrame）
            raise
        except ValueError as ve:
            print(f"错误：{ve}")
            raise
        except Exception as e:
            print(f"加载词典时发生其他错误: {e}")
            raise


    def load_custom_lexicon(self, filepath):
      """
      从文件加载自定义词典

      Args:
        filepath (str):
      """
      try:
        loaded_data = pd.read_excel(filepath)
        # 检查列名
        expected_columns = ['Word', 'Bigram', 'Conc.M', 'Unknown', 'Total', 'SUBTLEX']
        if not all(col in loaded_data.columns for col in expected_columns):
          raise ValueError("自定义词典文件列名不符合预期")

        self.custom_data = loaded_data # 更新整个 custom_data
        print("自定义词典加载成功")

      except FileNotFoundError:
        print("自定义词典文件不存在，已创建一个空的自定义词典")
        self.custom_data = pd.DataFrame(columns=expected_columns)  # 创建一个空的 DataFrame

      except ValueError as ve:
            print(f"错误：{ve}")

      except Exception as e:
        print(f"加载自定义词典时发生其他错误: {e}")
        raise

    def lookup(self, word):
        """
        查询词汇的具体性评分。

        优先查询自定义词典，其次查询主词典。

        Args:
            word (str): 要查询的词汇。

        Returns:
            float 或 None: 如果找到词汇，返回具体性评分 (Conc.M)；否则返回 None。
        """
        # 优先查询自定义词典
        if not self.custom_data.empty:  # 检查 custom_data 是否为空
            result = self.custom_data[self.custom_data['Word'] == word]
            if not result.empty:
                return result['Conc.M'].iloc[0]  # 返回 Conc.M 列的值

        # 如果自定义词典中没有，查询主词典
        if self.data is not None:
            result = self.data[self.data['Word'] == word]
            if not result.empty:
                return result['Conc.M'].iloc[0]

        return None  # 如果在两个词典中都没有找到，返回 None

    def add_custom_word(self, word, concreteness):
        """
        添加自定义词汇及其评分到 custom_data。

        Args:
            word (str): 要添加的词汇。
            concreteness (float): 词汇的具体性评分。
        """
        # 检查评分是否有效
        if not isinstance(concreteness, (int, float)):
            print("错误：具体性评分必须是数字。")
            return

        # 检查是否已存在
        if not self.custom_data.empty and (self.custom_data['Word'] == word).any():
            return False
            # print(f"警告：词汇 '{word}' 已存在于自定义词典中。")
            # while True:  # 使用循环，直到用户输入有效的 y 或 n
            #     choice = input("是否覆盖现有评分？(y/n): ").lower()  # 获取用户输入并转为小写
            #     if choice == 'y':
            #         self.custom_data.loc[self.custom_data['Word'] == word, 'Conc.M'] = concreteness
            #         print(f"词汇 '{word}' 的评分已更新。")
            #         break  # 更新后退出循环
            #     elif choice == 'n':
            #         print("操作已取消，保留原有评分。")
            #         return  # 取消操作，直接返回
            #     else:
            #         print("无效输入，请输入 'y' 或 'n'。")
        else:
            # 创建一个新的 DataFrame 来存储新词汇
            new_word_df = pd.DataFrame({'Word': [word], 'Bigram': [0], 'Conc.M': [concreteness],
                                        'Unknown': [0], 'Total': [0], 'SUBTLEX': [0]})  # 假设其他列默认为0

            # 使用 concat 来添加新词汇, ignore_index=True 确保索引重置
            self.custom_data = pd.concat([self.custom_data, new_word_df], ignore_index=True)
            print(f"词汇‘{word}’已成功添加")
            return True


    def save_custom_lexicon(self, filepath):
        """
        将自定义词典保存到文件（Excel 文件）。

        Args:
            filepath (str): 保存文件的路径。
        Returns:
            bool:  True 保存成功, False 保存失败
        """
        try:
            self.custom_data.to_excel(filepath, index=False)
            print(f"自定义词典已保存到: {filepath}")
            return True  # 保存成功
        except Exception as e:
            print(f"保存自定义词典时发生错误: {e}")
            return False  # 保存失败

    def clear_custom_lexicon(self, filepath):
        """
        清空自定义词典的内容（但保留文件和列结构）。
        """
        # 可以选择直接创建一个新的空 DataFrame:
        self.custom_data = pd.DataFrame(columns=['Word', 'Bigram', 'Conc.M', 'Unknown', 'Total', 'SUBTLEX'])
        try:
            self.custom_data.to_excel(filepath, index=False)
            print("自定义词典内容已清空。")
            return True  # 清空成功
        except Exception as e:
            print(f"保存自定义词典时发生错误: {e}")
            return False  # 保存失败



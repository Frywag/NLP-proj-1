import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from Lexicon import Lexicon
from TextAnalyzer import TextAnalyzer
from Comparator import Comparator
import matplotlib.pyplot as plt

# 加载词典
lexicon = Lexicon("./concrete_dictionary.xlsx")  # 替换为你的词典文件路径
custom_lexicon_path = "./custom_dictionary.xlsx"  # 自定义词典的默认文件名

# ------------------ Helper Functions ------------------


def get_colored_text(text, concreteness_scores):
    """
    根据具体性评分，生成带有颜色高亮的文本（Tkinter Text 组件版本）。

    Args:
        text (str): 原始文本
        concreteness_scores (list):  具体性评分列表
    Return:
       list: 一个包含文本和颜色标签的列表，可以直接插入到 Tkinter Text 组件

    """
    if not concreteness_scores:
        return [text]

    colored_text_parts = []
    words = text.split()
    for word, score in zip(words, concreteness_scores):
        if score is not None:
            if score >= 4.0:
                color = "red"  # 高具体性：红色
            elif score >= 3.1:
                color = "orange"  # 中高等具体性：橙色
            elif score >= 2.4:
                color = "green"  # 中等具体性：绿色
            else:
                color = "blue"  # 低具体性：蓝色
        else:
            color = "gray"  # 未登录词

        colored_text_parts.append((word, color))  # 将单词和颜色作为元组

    return colored_text_parts


# ------------------ GUI Functions ------------------

def analyze_text():
    """单文本分析"""
    text = single_text_entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showerror("错误", "请输入或选择文本！")
        return

    analyzer = TextAnalyzer(lexicon)
    analyzer.load_text(text)
    analyzer.preprocess()
    analyzer.calculate_concreteness()
    analyzer.calculate_avg_concreteness()

    # 更新结果显示
    avg_conc_label.config(text=f"{analyzer.avg_concreteness:.4f}" if analyzer.avg_concreteness is not None else "N/A")
    unlisted_count_label.config(text=str(len(analyzer.unlisted_words)))
    unlisted_listbox.delete(0,tk.END) # 先清空
    for word in analyzer.unlisted_words:
        unlisted_listbox.insert(tk.END, word)

    # 更新带颜色高亮的文本
    colored_text_text.delete("1.0", tk.END)  # 清空文本框
    colored_text_parts = get_colored_text(text,analyzer.concreteness_scores)

    # 交替插入文本和颜色标签
    # 插入文本和颜色标签
    for word, color in colored_text_parts:
        colored_text_text.insert(tk.END, word + " ", color)  # 在每个单词后面添加空格


    status_bar.config(text="单文本分析完成！")


def load_single_file():
    """加载单文本文件"""
    filepath = filedialog.askopenfilename(filetypes=(("Text Files", "*.txt"),))
    if filepath:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                single_text_entry.delete("1.0", tk.END)  # 清空文本框
                single_text_entry.insert("1.0", f.read())
            status_bar.config(text=f"已从文件加载文本: {filepath}")
        except Exception as e:
            messagebox.showerror("错误", f"读取文件错误: {e}")

def compare_texts():
    """比较文本"""
    text1 = text1_entry.get("1.0", tk.END).strip()
    text2 = text2_entry.get("1.0", tk.END).strip()

    if not text1 or not text2:
        messagebox.showerror("错误", "请输入或选择两个文本进行比较！")
        return

    analyzer1 = TextAnalyzer(lexicon)
    analyzer1.load_text(text1)
    analyzer1.preprocess()
    analyzer1.calculate_concreteness()

    analyzer2 = TextAnalyzer(lexicon)
    analyzer2.load_text(text2)
    analyzer2.preprocess()
    analyzer2.calculate_concreteness()

    comparator = Comparator(analyzer1, analyzer2)
    comparison_results = comparator.compare_metrics()

    # 更新表格
    for i in comparison_table.get_children(): # 清空表格
        comparison_table.delete(i)

    for index, row in comparison_results.iterrows():
        comparison_table.insert("", tk.END, values=(index, row['Text 1'], row['Text 2']))
    status_bar.config(text="文本比较完成！")
def load_file1():
    filepath = filedialog.askopenfilename(filetypes=(("Text Files", "*.txt"),))
    if filepath:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text1_entry.delete("1.0", tk.END)
                text1_entry.insert("1.0", f.read())
        except Exception as e:
             messagebox.showerror("错误", f"读取文件错误: {e}")

def load_file2():
    filepath = filedialog.askopenfilename(filetypes=(("Text Files", "*.txt"),))
    if filepath:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text2_entry.delete("1.0", tk.END)
                text2_entry.insert("1.0", f.read())
        except Exception as e:
            messagebox.showerror("错误", f"读取文件错误: {e}")

def show_wordcloud():
    text = single_text_entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showerror("错误","请先进行单文本分析！")
        return
    analyzer = TextAnalyzer(lexicon)
    analyzer.load_text(text)
    analyzer.preprocess()
    analyzer.calculate_concreteness()

    try:
        analyzer.generate_wordcloud(font_path='./SimHei.ttf')  # 确保这里字体路径正确

        status_bar.config(text="词云图生成完成！")

    except Exception as e:
        messagebox.showerror("错误", f"生成词云图时出错：{e}")


def show_concreteness_distribution():
    text = single_text_entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showerror("错误","请先进行单文本分析")
        return
    analyzer = TextAnalyzer(lexicon)
    analyzer.load_text(text)
    analyzer.preprocess()
    analyzer.calculate_concreteness()

    try:
        # 获取可用的词性列表，并添加 "All" 选项
        available_pos = list(set(pos for _, pos in analyzer.pos_tags))
        available_pos.sort()  # 排序，使选项更有序
        available_pos.insert(0, "All")  # 在开头插入 "All"

        # 创建词性选择下拉框
        pos_var = tk.StringVar(value="All")  # 默认选中 "All"
        pos_combobox = ttk.Combobox(textvariable=pos_var, values=available_pos, state="readonly")
        pos_combobox.pack(pady=5)

        def update_plot(*args):  # 当选择改变时，重新绘制图形
            selected_pos = pos_var.get()
            plt.clf()  # 清空当前的图形
            if selected_pos == "All":
                analyzer.plot_concreteness_distribution()  # 传入 None，表示不过滤

            else:
                analyzer.plot_concreteness_distribution(pos_filter=selected_pos)
        pos_var.trace("w", update_plot)  # 绑定选择事件

        analyzer.plot_concreteness_distribution()  # 初始绘制(不过滤)
        status_bar.config(text="具体性分布图生成完成！")

    except Exception as e:
        messagebox.showerror("错误", f"生成具体性分布图时出错: {e}")


def save_custom_lexicon():
    """保存自定义词典"""
    filepath = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=(("Excel Files", "*.xlsx"), ("All Files", "*.*"))
    )
    if filepath:
        if lexicon.save_custom_lexicon(filepath):
            messagebox.showinfo("成功", f"自定义词典已保存到: {filepath}")
            status_bar.config(text=f"自定义词典已保存到: {filepath}")
        else:
            messagebox.showerror("错误", "保存自定义词典失败！")


def clear_custom_lexicon():
    """清空自定义词典"""
    if messagebox.askyesno("确认", "确定要清空自定义词典吗？"):
        if lexicon.clear_custom_lexicon(custom_lexicon_path):
            messagebox.showinfo("成功", "自定义词典已清空。")
            status_bar.config(text="自定义词典已清空")
        else:  # 通常是保存出了问题
             messagebox.showerror("错误", "清空自定义词典失败！")


def export_unlisted_words():
    """导出未登录词"""
    # 确保已经进行了文本分析
    text = single_text_entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showerror("错误", "请先进行单文本分析！")
        return

    analyzer = TextAnalyzer(lexicon)
    analyzer.load_text(text)
    analyzer.preprocess()
    analyzer.calculate_concreteness()
    if not hasattr(analyzer, 'unlisted_words') or not analyzer.unlisted_words:
        messagebox.showerror("错误", "请先进行单文本分析，并确保存在未登录词！")
        return

    filepath = filedialog.asksaveasfilename(
        defaultextension=".xlsx",  # 默认扩展名
        filetypes=(("Excel Files", "*.xlsx"), ("CSV Files", "*.csv"), ("All Files", "*.*"))
    )
    if filepath:
        analyzer.export_unlisted_words(filepath) # 导出
        messagebox.showinfo("成功", f"未登录词已导出到: {filepath}")

def extract_summary(summary_widget):
    """提取摘要"""
    text = single_text_entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showerror("错误", "请输入或选择文本！")
        return

    analyzer = TextAnalyzer(lexicon)
    analyzer.load_text(text)
    analyzer.preprocess()
    analyzer.calculate_concreteness() # 确保计算了具体性

    try:
        # 获取摘要
        summary = analyzer.extract_summary_sentences(n=3)  # 提取3个句子
        summary_text = "\n".join(summary)  # 用换行符连接句子

        # 显示摘要
        summary_widget.config(state='normal')  # 设置为可编辑
        summary_widget.delete("1.0", tk.END)
        summary_widget.insert("1.0", summary_text)
        summary_widget.config(state='disabled')  # 设置为只读
        status_bar.config(text="摘要提取完成！")

    except Exception as e:
        messagebox.showerror("错误", f"提取摘要时发生错误：{e}")


def add_custom_word_gui():
    """
    通过 GUI 添加自定义词汇及其评分。
    """
    # 创建新窗口
    add_window = tk.Toplevel(root)
    add_window.title("添加自定义词汇")

    # 词汇输入框
    ttk.Label(add_window, text="词汇：").pack(pady=5)
    word_entry = ttk.Entry(add_window)
    word_entry.pack(pady=5)

    # 具体性评分输入框
    ttk.Label(add_window, text="具体性评分 (1.0-5.0)：").pack(pady=5)
    score_entry = ttk.Entry(add_window)
    score_entry.pack(pady=5)

    # 添加按钮 (注意这里的 command)
    def add_word():  # 将 add_word 定义为内部函数
        """
        内部函数，处理添加词汇的逻辑。
        """
        word = word_entry.get().strip()
        concreteness_str = score_entry.get().strip()

        if not word:
            messagebox.showerror("错误", "请输入词汇！")
            return
        if not concreteness_str:
            messagebox.showerror("错误", "请输入具体性评分！")
            return

        try:
            concreteness = float(concreteness_str)
            if concreteness < 1.0 or concreteness > 5.0:
                messagebox.showerror("错误", "具体性评分应在 1.0 到 5.0 之间！")
                return
        except ValueError:
            messagebox.showerror("错误", "具体性评分必须是数字！")
            return

        # 检查词汇是否已经存在
        if not lexicon.custom_data.empty and (lexicon.custom_data['Word'] == word).any():
            if messagebox.askyesno("覆盖确认", f"词汇 '{word}' 已存在，是否覆盖？"):
                lexicon.custom_data.loc[lexicon.custom_data['Word'] == word, 'Conc.M'] = concreteness
                messagebox.showinfo("成功", f"词汇‘{word}’已更新")
                lexicon.save_custom_lexicon(custom_lexicon_path)  # 保存
                add_window.destroy()  # 关闭窗口
                status_bar.config(text=f"词汇‘{word}’已更新, 自定义词典已保存")

            else:
                messagebox.showinfo("提示", "操作取消")
                status_bar.config(text="操作取消")
                return

        else:  # 不存在
            if lexicon.add_custom_word(word, concreteness):  # 调用 lexicon的方法
                messagebox.showinfo("成功", f"词汇 '{word}' 已添加到自定义词典！")
                lexicon.save_custom_lexicon(custom_lexicon_path)  # 保存
                add_window.destroy()  # 关闭窗口
                status_bar.config(text=f"词汇‘{word}’已添加, 自定义词典已保存")

            else:
                messagebox.showerror("错误", f"添加词汇 '{word}' 失败！")

    add_button = ttk.Button(add_window, text="添加", command=add_word)  # command 现在调用内部函数
    add_button.pack(pady=10)


# ------------------ GUI Layout (Tkinter) ------------------

root = tk.Tk()
root.title("文本分析系统")
root.geometry("1100x1000")  # 设置初始窗口大小

# 主框架
main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# 标题
title_label = ttk.Label(main_frame, text="基于词汇具体性的文本分析系统", font=("Helvetica", 20))
title_label.pack(pady=10)
description_label = ttk.Label(main_frame, text="本系统可以对英文文本进行具体性分析、难度评估、关键词/摘要提取，并支持文本比较功能。")
description_label.pack()
# 单文本分析框架
single_text_frame = ttk.LabelFrame(main_frame, text="单文本分析", padding=10)
single_text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

single_text_entry = tk.Text(single_text_frame, wrap=tk.WORD, height=10, width=40)
single_text_entry.pack(pady=5)

load_button = ttk.Button(single_text_frame, text="选择文件", command=load_single_file)
load_button.pack(pady=5)

analyze_button = ttk.Button(single_text_frame, text="分析", command=analyze_text)
analyze_button.pack(pady=5)

ttk.Label(single_text_frame, text="平均具体性得分：").pack(anchor=tk.W)
avg_conc_label = ttk.Label(single_text_frame, text="")
avg_conc_label.pack(anchor=tk.W)

ttk.Label(single_text_frame, text="未登录词数量：").pack(anchor=tk.W)
unlisted_count_label = ttk.Label(single_text_frame, text="")
unlisted_count_label.pack(anchor=tk.W)

ttk.Label(single_text_frame, text="未登录词列表：").pack(anchor=tk.W)
unlisted_listbox = tk.Listbox(single_text_frame, height=3,width=30)
unlisted_listbox.pack(anchor=tk.W)

ttk.Label(single_text_frame, text="带颜色高亮的文本：").pack(anchor=tk.W)
colored_text_text = tk.Text(single_text_frame, wrap=tk.WORD, height=5, width=40)
colored_text_text.pack(pady=5)
# 设置颜色标签
colored_text_text.tag_configure("red", foreground="red")
colored_text_text.tag_configure("orange", foreground="orange")
colored_text_text.tag_configure("green", foreground="green")
colored_text_text.tag_configure("blue", foreground="blue")
colored_text_text.tag_configure("gray", foreground="gray")


# 文本比较框架
compare_text_frame = ttk.LabelFrame(main_frame, text="文本比较", padding=10)
compare_text_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

ttk.Label(compare_text_frame, text="文本 1：").pack(anchor=tk.W)
text1_entry = tk.Text(compare_text_frame, wrap=tk.WORD, height=5, width=40)
text1_entry.pack(pady=5)
load_button1 = ttk.Button(compare_text_frame, text="选择文件", command=load_file1)
load_button1.pack(pady=5)

ttk.Label(compare_text_frame, text="文本 2：").pack(anchor=tk.W)
text2_entry = tk.Text(compare_text_frame, wrap=tk.WORD, height=5, width=40)
text2_entry.pack(pady=5)
load_button2 = ttk.Button(compare_text_frame, text="选择文件", command=load_file2)
load_button2.pack(pady=5)


compare_button = ttk.Button(compare_text_frame, text="比较", command=compare_texts)
compare_button.pack(pady=5)

# 表格
columns = ('指标', '文本 1', '文本 2')
comparison_table = ttk.Treeview(compare_text_frame, columns=columns, show='headings')
for col in columns:
    comparison_table.heading(col, text=col)
    comparison_table.column(col, width=100, anchor=tk.CENTER)  # 调整列宽
comparison_table.pack(pady=5)

#可视化按钮框架
visulization_frame = ttk.Frame(main_frame)
visulization_frame.pack(pady=10)

wordcloud_button = ttk.Button(visulization_frame,text="词云图",command=show_wordcloud)
wordcloud_button.pack(side=tk.LEFT, padx=5)
con_dist_button = ttk.Button(visulization_frame,text='具体性分布图',command=show_concreteness_distribution)
con_dist_button.pack(side=tk.LEFT, padx=5)

# 词典管理按钮 (添加到单文本分析框架下方)
lexicon_frame = ttk.Frame(single_text_frame)
lexicon_frame.pack(pady=10)

# 在 GUI 中添加按钮 (例如，在 lexicon_frame 中):
export_button = ttk.Button(lexicon_frame, text="导出未登录词", command=export_unlisted_words)
export_button.pack(side=tk.LEFT, padx=5)

add_word_button = ttk.Button(lexicon_frame, text="添加自定义词汇", command=add_custom_word_gui)  # 添加按钮
add_word_button.pack(side=tk.LEFT, padx=5)

save_custom_button = ttk.Button(lexicon_frame, text="保存自定义词典", command=save_custom_lexicon)
save_custom_button.pack(side=tk.LEFT, padx=5)

clear_custom_button = ttk.Button(lexicon_frame, text="清空自定义词典", command=clear_custom_lexicon)
clear_custom_button.pack(side=tk.LEFT, padx=5)



# 摘要提取
ttk.Label(single_text_frame, text="摘要：").pack(anchor=tk.W)
summary_text_widget = tk.Text(single_text_frame, wrap=tk.WORD, height=5, width=40, state="disabled") # 设置为只读
summary_text_widget.pack(pady=5)

extract_summary_button = ttk.Button(single_text_frame, text="提取摘要", command=lambda: extract_summary(summary_text_widget)) # 传递参数
extract_summary_button.pack(pady=5)



# 状态栏
# 创建一个样式
style = ttk.Style()
style.configure("Sunken.TLabel", borderwidth=1, relief="sunken")

status_bar = ttk.Label(root, text="", style="Sunken.TLabel", anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

root.mainloop()
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import joblib
import threading
import sys  # 添加导入sys
from PIL import Image, ImageTk
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk, ImageDraw
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# 全局颜色方案 - 高级科技感配色
COLORS = {
    "background": "#7e779d",
    "card": "#9d96be",
    "accent": "#F2A7C7",
    "text": "#FFFFFF",
    "text_muted": "#FFFFFF",
    "success": "#a1bae7",
    "button": "#a1bae7",
    "grey": "#484554",
    "button_hover": "#7e779d"
}

class ModelPredictorApp:
    def __init__(self):
        # 创建主窗口并支持拖放
        self.root = TkinterDnD.Tk()
        self.root.title("Meta Cassiterite_1.0")
        self.root.geometry("1200x900")
        self.root.tk.call('tk', 'scaling', 1.5)  # 适应高清显示器
        self.root.configure(bg=COLORS["background"])

        # 设置应用状态和数据
        self.models = {
            "Random Forest": None,
            "XGBoost": None,
            "TabNet": None,
            "Stacking": None
        }
        self.models_loaded = False  # 跟踪模型是否已加载
        self.data = None
        self.processed_data = None
        self.prediction_results = None
        self.feature_names = None
        self.sample_ids = None

        # 元素特征列表
        self.element_features = ['Al', 'Sc', 'Ti', 'V', 'Fe', 'Ga', 'W', 'Sb', 'Zr', 'Hf', 'Nb', 'Ta', 'U']
        # 派生特征列表
        self.derived_features = ['ZrHf', 'NbTa', 'SbW', 'FeAl', 'UHf', 'UZr']

        # 创建界面布局
        self.create_layout()

        # 创建加载遮罩（初始隐藏）
        self.create_loading_overlay()

    def resource_path(self, relative_path):
        """ 获取资源的绝对路径 """
        try:
            # PyInstaller 创建临时文件夹并将路径存储在 _MEIPASS 中
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        
        return os.path.join(base_path, relative_path)

    def create_layout(self):
        """创建主界面布局"""
        # 顶部标题栏
        header = tk.Frame(self.root, bg=COLORS["background"], height=60)
        header.pack(fill=tk.X)

        title_label = tk.Label(header, text="Meta Cassiterite",
                              font=("Arial", 30, "bold"),
                              fg="#2b2848", bg=COLORS["background"])
        # 使用 place() 精确居中
        title_label.place(relx=0.5, rely=0.5, anchor="center")

        # 主界面分为左右两栏
        main_frame = tk.Frame(self.root, bg=COLORS["background"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # 左侧控制面板
        left_frame = tk.Frame(main_frame, bg=COLORS["card"], width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # 上传区域
        upload_label = tk.Label(left_frame, text="💎 Upload",
                               font=("Arial", 20, "bold"),
                               fg=COLORS["text"], bg=COLORS["card"])
        upload_label.pack(anchor=tk.W, padx=50, pady=(15, 10))

        # 拖放区域
        self.drop_area = tk.Frame(left_frame, bg=COLORS["text"],
                                 height=150, width=270)
        self.drop_area.pack(fill=tk.X, padx=12, pady=10)
        self.drop_area.pack_propagate(False)

        self.drop_text = tk.Label(self.drop_area,
                                 text="Drop file here (xls, csv, xlsx)\n or \nClick to upload",
                                 font=("Arial", 12, "bold"),
                                 fg=COLORS["grey"], bg=COLORS["text"])
        self.drop_text.pack(expand=True)

        # 注册拖放区域
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', self.on_drop)

        # 浏览按钮
        browse_btn = tk.Button(left_frame, text="Open file", bg=COLORS["button"],
                              fg=COLORS["text"], font=("Arial", 15, "bold"),
                              relief=tk.FLAT, padx=10, pady=5,
                              command=self.browse_file)
        browse_btn.pack(fill=tk.X, padx=15, pady=10)

        # 文件信息显示
        self.file_var = tk.StringVar(value="No File Selected")
        file_info = tk.Label(left_frame, textvariable=self.file_var,
                            fg=COLORS["text_muted"], bg=COLORS["card"],
                            font=("Arial", 15, "bold"), wraplength=270)
        file_info.pack(fill=tk.X, padx=15, pady=5)

        # 分隔线
        separator = ttk.Separator(left_frame, orient="horizontal")
        separator.pack(fill=tk.X, padx=15, pady=15)

        # 模型控制
        model_label = tk.Label(left_frame, text="💎 Process",
                              font=("Arial", 20, "bold"),
                              fg=COLORS["text"], bg=COLORS["card"])
        model_label.pack(anchor=tk.W, padx=50, pady=(5, 10))

        # 预测按钮
        predict_btn = tk.Button(left_frame, text="Prediction", bg=COLORS["success"],
                               fg=COLORS["text"], font=("Arial", 15, "bold"),
                               relief=tk.FLAT, padx=10, pady=8,
                               command=self.run_prediction)
        predict_btn.pack(fill=tk.X, padx=15, pady=10)

        # 导出按钮
        export_btn = tk.Button(left_frame, text="Export Results", bg=COLORS["accent"],
                              fg=COLORS["text"], font=("Arial", 15, "bold"),
                              relief=tk.FLAT, padx=10, pady=5,
                              command=self.export_results)
        export_btn.pack(fill=tk.X, padx=15, pady=5)

        # 添加导出处理后数据集按钮
        export_processed_btn = tk.Button(left_frame, text="Export Processed Data",
                                        bg=COLORS["grey"],
                                        fg=COLORS["text"], font=("Arial", 15, "bold"),
                                        relief=tk.FLAT, padx=10, pady=5,
                                        command=self.export_processed_data)
        export_processed_btn.pack(fill=tk.X, padx=15, pady=5)

        # 模型状态指示
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(left_frame, textvariable=self.status_var,
                               font=("Arial", 10, "bold"),
                               fg=COLORS["text_muted"], bg=COLORS["card"])
        status_label.pack(anchor=tk.W, padx=15, pady=(20, 5))

        # 右侧结果显示面板
        right_frame = tk.Frame(main_frame, bg=COLORS["background"])
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 结果标题
        results_label = tk.Label(right_frame, text="Results",
                                font=("Arial", 20, "bold"),
                                fg=COLORS["text"], bg=COLORS["background"])
        results_label.pack(anchor=tk.W, padx=350, pady=10)

        # 结果表格框架
        table_frame = tk.Frame(right_frame, bg=COLORS["card"])
        table_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        # 创建滚动条
        y_scroll = ttk.Scrollbar(table_frame)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # 创建结果显示的树形视图
        self.result_tree = ttk.Treeview(table_frame,
                                       yscrollcommand=y_scroll.set,
                                       xscrollcommand=x_scroll.set)
        self.result_tree.pack(fill=tk.BOTH, expand=True)

        # 配置滚动条
        y_scroll.config(command=self.result_tree.yview)
        x_scroll.config(command=self.result_tree.xview)

        # 设置树形视图样式
        style = ttk.Style()
        style.configure("Treeview",
                       background=COLORS["card"],
                       foreground=COLORS["text"],
                       fieldbackground=COLORS["card"],
                       borderwidth=0)
        style.configure("Treeview.Heading",
                       background=COLORS["background"],
                       foreground=COLORS["accent"],
                       font=("Arial", 10, "bold"))

        # 底部状态栏
        self.status_bar = tk.Label(self.root, text="Ready",
                                  font=("Arial", 10, "bold"),
                                  bg="#7e779d", fg=COLORS["text_muted"],
                                  anchor=tk.W, padx=15)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_loading_overlay(self):
        """创建加载遮罩"""
        self.loading_frame = tk.Frame(self.root, bg=COLORS["background"])
        self.loading_frame.place(x=0, y=0, relwidth=1, relheight=1)

        loading_text = tk.Label(self.loading_frame, text="In processing, please wait...",
                               font=("Arial", 16, "bold"),
                               fg=COLORS["accent"], bg=COLORS["background"])
        loading_text.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

        progress = ttk.Progressbar(self.loading_frame, mode="indeterminate",
                                  length=400)
        progress.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        progress.start()

        # 初始隐藏
        self.loading_frame.place_forget()

    def on_drop(self, event):
        """处理文件拖放"""
        file_path = event.data

        # 处理Windows路径格式
        if file_path.startswith("{") and file_path.endswith("}"):
            file_path = file_path[1:-1]
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]

        # 多文件只取第一个
        if " " in file_path:
            file_path = file_path.split(" ")[0]

        # 检查文件类型
        if file_path.lower().endswith(('.csv', '.xlsx', '.xls')):
            self.load_data(file_path)
        else:
            messagebox.showerror("Error", "Unsupported file format. Please upload a CSV or Excel file.")

    def browse_file(self):
        """通过文件对话框选择文件"""
        file_path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[
                ("All files", "*.*"),
                ("Excel file", "*.xlsx *.xls"),
                ("CSV file", "*.csv")
            ]
        )

        if file_path:
            self.load_data(file_path)

    def apply_log10_transform(self, df):
        """对元素特征进行Log10变换，保留缺失值"""
        transformed_df = df.copy()
        for col in self.element_features:
            if col in transformed_df.columns:
                # 仅对非缺失值应用log10变换
                mask = transformed_df[col].notna() & (transformed_df[col] > 0)
                transformed_df.loc[mask, col] = np.log10(transformed_df.loc[mask, col])
        return transformed_df

    def impute_missing_values(self, df):
        """使用随机森林多次插补缺失值并平均"""
        # 提取要插补的列
        cols_to_impute = [col for col in self.element_features if col in df.columns]
        if not cols_to_impute:
            return df

        # 提取需要插补的数据
        data_to_impute = df[cols_to_impute].copy()

        # 如果没有缺失值，直接返回原始数据
        if not data_to_impute.isna().any().any():
            return df

        # 多重插补
        imputed_data_list = []
        for i in range(10):  # 执行10次插补
            estimator = RandomForestRegressor(
                n_estimators=50,
                max_depth=7,
                n_jobs=-1,
                random_state=2025+i
            )

            imp = IterativeImputer(
                estimator=estimator,
                max_iter=15,
                random_state=2025+i,
                tol=0.01
            )

            # 执行插补
            imputed_data = imp.fit_transform(data_to_impute)
            imputed_data_list.append(imputed_data)

        # 计算平均值
        imputed_avg = np.mean(imputed_data_list, axis=0)

        # 创建插补后的DataFrame
        imputed_df = df.copy()
        imputed_df[cols_to_impute] = imputed_avg

        return imputed_df

    def create_derived_features(self, df):
        """创建派生特征"""
        df_with_derived = df.copy()

        # 确保所有元素特征都存在
        for col in self.element_features:
            if col not in df_with_derived.columns:
                self.status_bar.config(text=f"Warning: {col} column not found. Derived features may be incomplete.")

        # 创建派生特征
        if 'Zr' in df.columns and 'Hf' in df.columns:
            df_with_derived['ZrHf'] = df['Zr'] - df['Hf']

        if 'Nb' in df.columns and 'Ta' in df.columns:
            df_with_derived['NbTa'] = df['Nb'] - df['Ta']

        if 'Sb' in df.columns and 'W' in df.columns:
            df_with_derived['SbW'] = df['Sb'] - df['W']

        if 'Fe' in df.columns and 'Al' in df.columns:
            df_with_derived['FeAl'] = df['Fe'] - df['Al']

        if 'U' in df.columns and 'Hf' in df.columns:
            df_with_derived['UHf'] = df['U'] - df['Hf']

        if 'U' in df.columns and 'Zr' in df.columns:
            df_with_derived['UZr'] = df['U'] - df['Zr']

        return df_with_derived

    def load_data(self, file_path):
        """加载并处理数据文件"""
        try:
            # 显示加载遮罩
            self.loading_frame.place(x=0, y=0, relwidth=1, relheight=1)
            self.root.update_idletasks()

            # 根据文件类型加载数据
            if file_path.lower().endswith('.csv'):
                self.data = pd.read_csv(file_path)
            else:  # Excel文件
                self.data = pd.read_excel(file_path)

            # 复制原始数据用于保留所有列
            self.original_data = self.data.copy()

            # 检查数据格式
            # 提取ID列，如果存在的话
            self.sample_id_column = None
            self.sample_ids = None

            # 检查可能的ID列名
            possible_id_columns = ['ID', 'Sample_ID', 'SampleID', 'Sample', 'id', 'sample_id', 'sampleid', 'sample']
            for col in possible_id_columns:
                if col in self.data.columns:
                    self.sample_id_column = col
                    self.sample_ids = self.data[col].astype(str).tolist()  # 确保转换为字符串
                    break

            # 如果没有找到ID列，生成默认ID
            if self.sample_ids is None:
                self.sample_ids = [f"样品{i+1}" for i in range(len(self.data))]
                # 添加到原始数据中
                self.original_data['Sample_ID'] = self.sample_ids
                self.sample_id_column = 'Sample_ID'

            # 显示原始数据中元素特征的缺失值统计
            print("\n原始数据缺失值统计:")
            for col in self.element_features:
                if col in self.data.columns:
                    missing = self.data[col].isna().sum()
                    total = len(self.data)
                    print(f"{col}: {missing}/{total} ({missing/total*100:.1f}%)")

            # 1. 对元素特征进行Log10变换
            self.status_bar.config(text="Applying LOG10 transform...")
            log10_data = self.apply_log10_transform(self.data)
            print("\nLOG10变换后的数据样例：")
            element_cols_in_data = [col for col in self.element_features if col in log10_data.columns]
            if element_cols_in_data:
                print(log10_data[element_cols_in_data].head())

            # 显示LOG10变换后缺失值统计
            print("\nLOG10变换后缺失值统计:")
            for col in self.element_features:
                if col in log10_data.columns:
                    missing = log10_data[col].isna().sum()
                    total = len(log10_data)
                    print(f"{col}: {missing}/{total} ({missing/total*100:.1f}%)")

            # 2. 对Log10变换后的数据进行缺失值插补
            self.status_bar.config(text="Imputing missing values with Random Forest...")
            imputed_data = self.impute_missing_values(log10_data)
            print("\n随机森林插补后的数据样例：")
            if element_cols_in_data:
                print(imputed_data[element_cols_in_data].head())

            # 显示插补后缺失值统计
            print("\n随机森林插补后缺失值统计:")
            for col in self.element_features:
                if col in imputed_data.columns:
                    missing = imputed_data[col].isna().sum()
                    total = len(imputed_data)
                    print(f"{col}: {missing}/{total} ({missing/total*100:.1f}%)")

            # 3. 创建派生特征
            self.status_bar.config(text="Creating derived features...")
            processed_data = self.create_derived_features(imputed_data)
            print("\n派生特征计算后的数据样例：")
            derived_cols_in_data = [col for col in self.derived_features if col in processed_data.columns]
            if derived_cols_in_data:
                print(processed_data[derived_cols_in_data].head())

            # 处理预测特征
            self.processed_df = processed_data.copy()  # 保存处理后的完整DataFrame
            X = processed_data.copy()

            # 如果找到了ID列，从特征中移除
            if self.sample_id_column and self.sample_id_column in X.columns:
                X = X.drop(columns=[self.sample_id_column])

            # 如果有Group列，也需要删除
            if 'Group' in X.columns:
                X = X.drop(columns=['Group'])

            # 更新特征名列表
            self.feature_names = X.columns.tolist()
            self.processed_data = X.values

            # 更新文件信息
            self.file_var.set(f"Loading data: {os.path.basename(file_path)}\n{len(self.data)} Row × {len(self.feature_names)} Column")
            self.status_bar.config(text=f"Data has been loaded and processed successfully: {os.path.basename(file_path)}")

            # 显示处理流程摘要
            print("\n数据处理流程摘要:")
            print(f"原始数据: {len(self.data)} 行 × {len(self.data.columns)} 列")
            print(f"处理后数据: {len(self.processed_data)} 行 × {len(self.feature_names)} 列")
            print(f"处理后特征: {', '.join(self.feature_names)}")

            # 隐藏加载遮罩
            self.loading_frame.place_forget()

        except Exception as e:
            self.loading_frame.place_forget()
            messagebox.showerror("Error", f"Loading failed: {str(e)}")
            self.status_bar.config(text="Fails in loading data")
            print(f"Error during data loading: {str(e)}")

    def export_processed_data(self):
        """导出处理后的数据集"""
        if self.processed_data is None or self.feature_names is None:
            messagebox.showwarning("警告", "没有可导出的已处理数据")
            return

        file_path = filedialog.asksaveasfilename(
            title="导出处理后数据",
            defaultextension=".xlsx",
            filetypes=[
                ("Excel file", "*.xlsx"),
                ("CSV file", "*.csv")
            ]
        )

        if not file_path:
            return

        try:
            # 创建包含处理后数据的DataFrame
            processed_df = pd.DataFrame(self.processed_data, columns=self.feature_names)

            # 添加样品ID列
            processed_df.insert(0, "Sample_ID", self.sample_ids)

            # 导出
            if file_path.endswith('.xlsx'):
                processed_df.to_excel(file_path, index=False)
            else:
                processed_df.to_csv(file_path, index=False)

            self.status_bar.config(text=f"处理后数据已成功导出: {os.path.basename(file_path)}")
            messagebox.showinfo("成功", "已成功导出处理后的数据集")

            # 打印处理步骤信息
            print("数据处理流程:")
            print("1. 对元素浓度 (Al, Sc, Ti, V, Fe, Ga, W, Sb, Zr, Hf, Nb, Ta, U) 进行LOG10变换，保留缺失值")
            print("2. 使用RandomForest进行10次插补，取平均值填充缺失数据")
            print("3. 创建派生特征: ZrHf, NbTa, SbW, FeAl, UHf, UZr")
            print(f"导出的处理后数据集包含 {len(processed_df)} 行和 {len(processed_df.columns)} 列")
            print(f"特征列表: {', '.join(processed_df.columns.tolist())}")

        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")

    def load_models(self):
        """加载预训练模型"""
        try:
            # 显示加载遮罩
            self.loading_frame.place(x=0, y=0, relwidth=1, relheight=1)
            self.root.update_idletasks()

            # 使用线程加载模型以防止UI冻结
            thread = threading.Thread(target=self._load_models_thread)
            thread.daemon = True
            thread.start()

            return True
        except Exception as e:
            self.loading_frame.place_forget()
            messagebox.showerror("错误", f"加载模型失败: {str(e)}")
            return False

    def _load_models_thread(self):
        """在线程中加载模型"""
        error_message = None

        try:
            # 加载模型
            self.root.after(0, lambda: self.status_var.set("Loading RF model... (1/4)"))
            self.models["Random Forest"] = joblib.load(self.resource_path('best_rf_model.pkl'))

            self.root.after(0, lambda: self.status_var.set("Loading XGB model... (2/4)"))
            self.models["XGBoost"] = joblib.load(self.resource_path('best_xgb_model.pkl'))

            self.root.after(0, lambda: self.status_var.set("Loading TN model... (3/4)"))
            self.models["TabNet"] = joblib.load(self.resource_path('best_tabnet_model.pkl'))

            self.root.after(0, lambda: self.status_var.set("Loading Ensemble model... (4/4)"))
            self.models["Stacking"] = joblib.load(self.resource_path('best_ensem_model.pkl'))

            # 设置模型已加载标志
            self.models_loaded = True

            # 更新状态
            self.root.after(0, lambda: self.status_var.set("Model loaded"))

            # 如果是从预测函数调用的，继续执行预测
            if hasattr(self, '_continue_prediction') and self._continue_prediction:
                self.root.after(100, self._run_prediction_after_loading)
                self._continue_prediction = False

        except Exception as e:
            error_message = str(e)
            self.root.after(0, lambda: self.status_var.set("Failed to loading model"))
            self.models_loaded = False

        finally:
            # 隐藏加载遮罩
            self.root.after(0, self.loading_frame.place_forget)

            # 如果有错误，显示错误消息
            if error_message:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to loading model: {error_message}"))

    def run_prediction(self):
        """执行预测"""
        if self.processed_data is None:
            messagebox.showwarning("Warning", "Please load original data")
            return

        # 如果模型未加载，先加载模型
        if not self.models_loaded:
            self.status_var.set("To load model...")
            self._continue_prediction = True
            if not self.load_models():
                return
        else:
            # 如果模型已加载，直接运行预测
            self._run_prediction_after_loading()

    def _run_prediction_after_loading(self):
        """模型加载完成后执行预测"""
        try:
            # 显示加载遮罩
            self.loading_frame.place(x=0, y=0, relwidth=1, relheight=1)
            self.root.update_idletasks()

            # 在线程中执行预测
            thread = threading.Thread(target=self._predict_thread)
            thread.daemon = True
            thread.start()

        except Exception as e:
            self.loading_frame.place_forget()
            messagebox.showerror("Error", f"Error in predicition processes: {str(e)}")

    def _predict_thread(self):
        """在线程中执行预测"""
        error_message = None

        try:
            # 创建一个新的结果字典
            results = {}

            # 使用从数据文件加载的样品ID
            results["Sample_ID"] = self.sample_ids

            # 获取类别映射（字母表示）
            try:
                class_names = self.models["Random Forest"].classes_
                class_map = {i: chr(65+i) for i in range(len(class_names))}  # A, B, C...
            except:
                class_map = None

            # 随机森林预测
            self.root.after(0, lambda: self.status_bar.config(text="正在使用随机森林模型预测..."))
            rf_pred = self.models["Random Forest"].predict(self.processed_data)
            rf_proba = self.models["Random Forest"].predict_proba(self.processed_data)

            if class_map:
                # 将类别数值转换为字母
                results["RF_Class"] = [class_map[p] for p in rf_pred]
            else:
                results["RF_Class"] = rf_pred

            results["RF_Confidence"] = np.max(rf_proba, axis=1)

            # XGBoost预测
            self.root.after(0, lambda: self.status_bar.config(text="正在使用XGBoost模型预测..."))
            xgb_pred = self.models["XGBoost"].predict(self.processed_data)
            xgb_proba = self.models["XGBoost"].predict_proba(self.processed_data)

            if class_map:
                results["XGB_Class"] = [class_map[p] for p in xgb_pred]
            else:
                results["XGB_Class"] = xgb_pred

            results["XGB_Confidence"] = np.max(xgb_proba, axis=1)

            # TabNet预测
            self.root.after(0, lambda: self.status_bar.config(text="正在使用TabNet模型预测..."))
            tabnet_pred = self.models["TabNet"].predict(self.processed_data)
            tabnet_proba = self.models["TabNet"].predict_proba(self.processed_data)

            if class_map:
                results["TabNet_Class"] = [class_map[p] for p in tabnet_pred]
            else:
                results["TabNet_Class"] = tabnet_pred

            results["TabNet_Confidence"] = np.max(tabnet_proba, axis=1)

            # 集成模型预测
            self.root.after(0, lambda: self.status_bar.config(text="正在使用集成模型预测..."))
            stack_pred = self.models["Stacking"].predict(self.processed_data)
            stack_proba = self.models["Stacking"].predict_proba(self.processed_data)

            if class_map:
                results["Stacking_Class"] = [class_map[p] for p in stack_pred]
            else:
                results["Stacking_Class"] = stack_pred

            results["Stacking_Confidence"] = np.max(stack_proba, axis=1)

            # 创建结果DataFrame
            self.prediction_results = pd.DataFrame(results)

            # 将原始数据特征添加到结果中（用于导出）
            # 首先添加原始元素特征（经过LOG10和插补）
            element_cols = [col for col in self.element_features if col in self.feature_names]
            for i, col in enumerate(element_cols):
                col_index = self.feature_names.index(col)
                self.prediction_results[col] = self.processed_data[:, col_index]

            # 然后添加派生特征
            derived_cols = [col for col in self.derived_features if col in self.feature_names]
            for i, col in enumerate(derived_cols):
                col_index = self.feature_names.index(col)
                self.prediction_results[col] = self.processed_data[:, col_index]

            # 更新UI显示结果
            self.root.after(0, self.display_results)
            self.root.after(0, lambda: self.status_bar.config(text="Prediction Finished"))

        except Exception as e:
            error_message = str(e)
            self.root.after(0, lambda: self.status_bar.config(text="Prediction Failed"))
            print(f"预测错误: {str(e)}")

        finally:
            # 隐藏加载遮罩
            self.root.after(0, self.loading_frame.place_forget)

            # 如果有错误，显示错误消息
            if error_message:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Prediction Failed: {error_message}"))

    def display_results(self):
        """在表格中显示预测结果"""
        if self.prediction_results is None:
            return

        # 清除现有数据
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)

        # 设置要显示的列
        display_cols = [
            "Sample_ID",
            "RF_Class", "RF_Confidence",
            "XGB_Class", "XGB_Confidence",
            "TabNet_Class", "TabNet_Confidence",
            "Stacking_Class", "Stacking_Confidence"
        ]

        # 配置树形视图
        self.result_tree['columns'] = display_cols
        self.result_tree['show'] = 'headings'

        # 设置列标题
        column_labels = {
            "Sample_ID": "Sample ID",
            "RF_Class": "RF Class",
            "RF_Confidence": "RF Prob",
            "XGB_Class": "XGB Class",
            "XGB_Confidence": "XGB Prob",
            "TabNet_Class": "TabNet Class",
            "TabNet_Confidence": "TabNet Prob",
            "Stacking_Class": "Ensemble Class",
            "Stacking_Confidence": "Ensemble Prob"
        }

        for col in display_cols:
            self.result_tree.heading(col, text=column_labels.get(col, col))

            # 设置列宽
            if col == "Sample_ID":
                self.result_tree.column(col, width=100, anchor=tk.W)  # 调整宽度并左对齐
            elif "Confidence" in col:
                self.result_tree.column(col, width=80, anchor=tk.CENTER)
            else:
                self.result_tree.column(col, width=80, anchor=tk.CENTER)

        # 插入数据
        for i, row in self.prediction_results.iterrows():
            values = []
            for col in display_cols:
                if "Confidence" in col:
                    # 格式化置信度为 XX.XX%
                    values.append(f"{row[col]*100:.2f}%")
                else:
                    values.append(str(row[col]))

            self.result_tree.insert('', 'end', values=values)

        # 显示记录数量
        print(f"显示预测结果: {len(self.prediction_results)} 条记录")

    def export_results(self):
        """导出预测结果"""
        if self.prediction_results is None:
            messagebox.showwarning("Warning", "No results for exporting")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Prediction",
            defaultextension=".xlsx",
            filetypes=[
                ("Excel file", "*.xlsx"),
                ("CSV file", "*.csv")
            ]
        )

        if not file_path:
            return

        try:
            # 创建导出数据的副本
            export_df = self.prediction_results.copy()

            # 将置信度列转换为百分比格式
            for col in export_df.columns:
                if "Confidence" in col:
                    export_df[col] = export_df[col].apply(lambda x: f"{x*100:.2f}%")

            if file_path.endswith('.xlsx'):
                export_df.to_excel(file_path, index=False)
            else:
                export_df.to_csv(file_path, index=False)

            self.status_bar.config(text=f"Exported successfully: {os.path.basename(file_path)}")
            messagebox.showinfo("Success", "Successful to export")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")

    def run(self):
        """运行应用程序"""
        self.root.mainloop()

if __name__ == "__main__":
    app = ModelPredictorApp()
    app.run()
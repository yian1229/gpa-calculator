import streamlit as st
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
import os
import shutil
import pytesseract
from openai import OpenAI
import json
import io

# ==========================================
# 模块合并：OCR Helper (原 ocr_helper.py)
# ==========================================

# 默认不指定路径，依赖系统 PATH (适用于 Linux/Cloud)
# 仅在 Windows 本地测试时可能需要指定路径
DEFAULT_TESSERACT_PATH = None

def preprocess_image(image):
    """
    图像预处理：针对深色模式优化
    1. 转换为灰度图
    2. 反相 (如果是黑底白字)
    3. 增强对比度
    """
    # 转换为 RGB (防止 PNG 透明通道问题)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 转换为灰度
    gray_image = ImageOps.grayscale(image)
    
    # 检测是否为深色模式 (计算平均像素亮度，<128 认为是深色)
    # 简单采样中间区域
    width, height = gray_image.size
    crop = gray_image.crop((width//4, height//4, width*3//4, height*3//4))
    extrema = crop.getextrema()
    # 如果大部分像素比较暗，可能是黑底白字
    # 这里我们直接做个“反相”副本，两个都让 OCR 跑一遍，谁字多用谁？
    # 或者直接暴力点，假设用户提供的截图大部分是黑底（手机截图常见），尝试反相。
    
    # 为了保险，我们生成一个“反相”版本（变成白底黑字）
    inverted_image = ImageOps.invert(gray_image)
    
    # 增强对比度
    enhancer = ImageEnhance.Contrast(inverted_image)
    enhanced_image = enhancer.enhance(2.0)
    
    return enhanced_image

def perform_ocr(image, tesseract_cmd=None):
    """
    使用 Tesseract 对图像进行 OCR 识别
    """
    # 1. 优先使用用户传入的路径 (如果非空)
    if tesseract_cmd and tesseract_cmd.strip():
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd.strip()
    else:
        # 2. 尝试自动检测系统中的 tesseract
        if shutil.which("tesseract"):
            # 在 Linux/Cloud 环境下通常能直接找到
            pytesseract.pytesseract.tesseract_cmd = "tesseract"
        else:
            # 3. Windows 本地回退逻辑
            win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(win_path):
                pytesseract.pytesseract.tesseract_cmd = win_path
    
    try:
        # --- 增强版 OCR 逻辑 ---
        # 1. 对原图进行预处理 (针对深色模式反相)
        processed_image = preprocess_image(image)
        
        # 2. 识别 (同时保留原图识别结果，防止反相错误)
        # 这里我们只用处理后的图，因为 Tesseract 极其讨厌黑底
        text = pytesseract.image_to_string(processed_image, lang='chi_sim+eng')
        
        return text
    except pytesseract.TesseractError as e:
        if "lang" in str(e):
             return "Error: 请确保 Tesseract 安装了中文语言包 (chi_sim)。\n或者您可以尝试只识别数字和英文。"
        return f"OCR Error: {e}"
    except Exception as e:
        # Fallback provided in UI if tesseract is missing
        return f"Error: 无法运行 Tesseract OCR。请确保已安装 Tesseract 并配置路径。\n详细错误: {e}"

def parse_with_deepseek(ocr_text, api_key):
    """
    使用 DeepSeek API 清洗和结构化数据
    """
    if not ocr_text or "Error" in ocr_text:
        return []

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # 针对用户截图特点（黑底卡片式）优化 Prompt
    prompt = f"""
    你是一个数据提取助手。请从下面的 OCR 识别文本中提取“科目名称”、“成绩”和“学分”。
    
    OCR 文本内容：
    {ocr_text}
    
    核心提取规则 (CRITICAL):
    1. **成绩 (Score)**: 
       - 重点寻找位于行尾或独立的**大数值**（通常是 60-100 之间的整数）。
       - **忽略**标记为“平时成绩”、“期中成绩”的小数值（通常 < 50）。
       - 如果一行有多个数字，例如 "平时成绩: 29 绩点: 4.5 95"，取那个最大的 **95** 作为最终成绩。
       
    2. **学分 (Credit)**:
       - 学分通常紧跟在课程名称下方或旁边。
       - 寻找类似 "限选 - 3 学分"、"必修 - 2.0 学分"、"Credit: 3" 的模式。
       - 如果找不到明确的“学分”字样，尝试寻找 0.5 到 6.0 之间的小数（通常是 1, 2, 3, 4, 0.5）。
       
    3. **科目名称 (Subject)**:
       - 提取中文课程名。
       
    4. **去噪**: 
       - 忽略 "学期 2025-2026-1"、"考试成绩" 等无关表头。

    输出格式:
    标准的 JSON 列表，无 Markdown。
    [
        {{"subject": "ERP原理", "score": 95, "credit": 3.0}},
        {{"subject": "就业指导", "score": 85, "credit": 0.5}}
    ]
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a smart data extraction assistant. Output raw JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        # 清理可能存在的 markdown 标记
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        data = json.loads(content)
        return data
    except Exception as e:
        print(f"DeepSeek API Error: {e}")
        return []

# ==========================================
# 模块合并：GPA Calculator (原 gpa_calculator.py)
# ==========================================

def calculate_gpa(data_list):
    """
    计算绩点
    输入: [{"subject": "Math", "score": 80, "credit": 2}, ...]
    输出: (平均绩点, 详细数据的 DataFrame)
    """
    if not data_list:
        return 0.0, pd.DataFrame()

    # 转换为 DataFrame 方便处理
    df = pd.DataFrame(data_list)
    
    # 数据清洗：确保数值类型正确
    try:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df['credit'] = pd.to_numeric(df['credit'], errors='coerce')
    except KeyError:
        return 0.0, df # 缺少列

    df = df.dropna(subset=['score', 'credit']) # 去除无效数据

    # 去重：保留第一个出现的科目
    # 假设 'subject' 列存在
    if 'subject' in df.columns:
        df = df.drop_duplicates(subset=['subject'], keep='first')
    
    if df.empty:
        return 0.0, df

    # 计算单科绩点
    # 公式：(成绩 - 50) / 10
    df['gpa_point'] = df['score'].apply(lambda x: (x - 50) / 10)
    
    # 乘以学分
    df['weighted_point'] = df['gpa_point'] * df['credit']
    
    total_weighted_point = df['weighted_point'].sum()
    total_credit = df['credit'].sum()
    
    if total_credit == 0:
        final_gpa = 0.0
    else:
        final_gpa = total_weighted_point / total_credit
        
    return final_gpa, df

# ==========================================
# 主程序：Streamlit App
# ==========================================

# 设置页面配置
st.set_page_config(page_title="智能绩点计算器", page_icon="🎓")

st.title("🎓 智能绩点计算器")
st.markdown("""
这是一个专为新手设计的绩点计算工具。
**流程**：上传成绩截图 -> 自动识别 -> 自动计算平均绩点。
""")

# --- 侧边栏配置 ---
st.sidebar.header("⚙️ 设置")

# 1. DeepSeek API Key
api_key = st.sidebar.text_input("DeepSeek API Key", type="password", help="请输入你的 DeepSeek API Key")
if not api_key:
    st.sidebar.warning("⚠️ 请先输入 API Key 才能使用智能识别功能")

# 2. Tesseract 路径配置 (仅在非云端环境显示，或折叠显示)
# 在云端 (Linux) 通常不需要手动设置，除非是本地 Windows 用户
is_windows = os.name == 'nt'
tesseract_cmd = None

if is_windows:
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔧 本地 OCR 设置 (Windows)", expanded=False):
        default_val = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        tesseract_cmd = st.text_input("Tesseract 路径", value=default_val)
        st.info("如果是云端部署，请忽略此项。")

# --- 主界面 ---

# 1. 上传图片
uploaded_files = st.file_uploader("请上传成绩单截图（支持多张）", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    st.success(f"已上传 {len(uploaded_files)} 张图片")
    
    # 预览图片
    with st.expander("查看上传的图片"):
        cols = st.columns(len(uploaded_files))
        for i, file in enumerate(uploaded_files):
            image = Image.open(file)
            cols[i].image(image, caption=f"图片 {i+1}", use_column_width=True)

    # 2. 开始处理按钮
    if st.button("🚀 开始识别并计算"):
        if not api_key:
            st.error("❌ 请先在左侧侧边栏输入 DeepSeek API Key")
        else:
            all_extracted_data = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"正在处理第 {i+1}/{len(uploaded_files)} 张图片...")
                
                # 读取图片
                image = Image.open(file)
                
                # 步骤 A: OCR 识别
                ocr_text = perform_ocr(image, tesseract_cmd)
                if "Error" in ocr_text and "Tesseract" in ocr_text:
                    st.error(f"图片 {i+1} OCR 失败: {ocr_text}")
                    st.stop()
                
                # 步骤 B: DeepSeek 解析
                status_text.text(f"正在智能解析第 {i+1} 张图片的内容...")
                parsed_data = parse_with_deepseek(ocr_text, api_key)
                
                if parsed_data:
                    all_extracted_data.extend(parsed_data)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("处理完成！正在计算...")
            
            # 3. 计算与展示
            if not all_extracted_data:
                st.warning("未能从图片中识别出任何有效数据，请检查图片清晰度或 OCR 设置。")
            else:
                final_gpa, df_result = calculate_gpa(all_extracted_data)
                
                st.divider()
                st.subheader("📊 计算结果")
                
                # 展示总绩点
                st.metric(label="平均绩点 (GPA)", value=f"{final_gpa:.4f}")
                
                # 展示详细表格
                st.markdown("### 详细清单")
                st.dataframe(df_result)
                
                # 下载结果
                csv = df_result.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下载计算结果 (CSV)",
                    data=csv,
                    file_name='gpa_result.csv',
                    mime='text/csv',
                )

# --- 底部帮助 ---
st.divider()
with st.expander("❓ 新手使用指南"):
    st.markdown("""
    ### 1. 获取 DeepSeek API Key
    1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)。
    2. 注册/登录并创建一个 API Key。
    3. 复制 Key 粘贴到左侧输入框。
    
    ### 2. 关于 OCR 识别
    本工具在云端会自动调用 Tesseract OCR。
    如果遇到识别错误，请尝试裁剪图片，仅保留成绩表格部分。
    """)

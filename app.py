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
    使用 Tesseract 对图像进行 OCR 识别 (双重策略：原图 + 反相图)
    """
    # 1. 设置 Tesseract 路径
    if tesseract_cmd and tesseract_cmd.strip():
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd.strip()
    else:
        if shutil.which("tesseract"):
            pytesseract.pytesseract.tesseract_cmd = "tesseract"
        else:
            win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(win_path):
                pytesseract.pytesseract.tesseract_cmd = win_path
    
    try:
        results = []
        
        # --- 策略 A: 原图识别 ---
        # 针对部分非深色区域或正常文字
        text_original = pytesseract.image_to_string(image, lang='chi_sim+eng')
        results.append(f"--- Source A (Original) ---\n{text_original}")
        
        # --- 策略 B: 反相增强识别 ---
        # 针对深色模式 (黑底白字)
        processed_image = preprocess_image(image)
        text_inverted = pytesseract.image_to_string(processed_image, lang='chi_sim+eng')
        results.append(f"--- Source B (Inverted) ---\n{text_inverted}")
        
        # 合并所有文本
        return "\n".join(results)
        
    except pytesseract.TesseractError as e:
        if "lang" in str(e):
             return "Error: 请确保 Tesseract 安装了中文语言包 (chi_sim)。"
        return f"OCR Error: {e}"
    except Exception as e:
        return f"Error: 无法运行 Tesseract OCR。详细错误: {e}"

def parse_with_deepseek(ocr_text, api_key):
    """
    使用 DeepSeek API 清洗和结构化数据 (终极优化版)
    """
    if not ocr_text or "Error" in ocr_text:
        return []

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # 针对用户反馈的痛点进行 Prompt 终极深度优化
    prompt = f"""
    你是一个拥有高级纠错能力的教务数据提取专家。请处理以下包含噪音和重复内容的 OCR 文本。
    
    OCR 文本内容 (包含原图和处理后图像的识别结果):
    {ocr_text}
    
    === 核心任务 ===
    提取所有课程的：1. 科目名称 (subject)  2. 成绩 (score)  3. 学分 (credit)
    
    === 强制纠错规则 (HIGHEST PRIORITY) ===
    1. **特定课程修正**:
       - **“习近平新时代中国特色社会主义思想概论”**: 这门课的标准学分是 **3.0**。OCR 经常将“3”误识别为“5”或“8”。如果你看到 5 学分，**必须强制修正为 3 学分**。
       - **“形势与政策”**: 这门课通常没有显示学分，或者学分很少。如果找不到学分，**必须保留该课程并设学分为 0**。不要因为缺学分而丢弃它。
    
    2. **学分 (Credit) 识别逻辑**:
       - 常见学分值：0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0。
       - **异常值警惕**：如果识别出 5.0, 6.0, 8.0 等非常规学分，请结合上下文（如“限选 - X 学分”）仔细辨别。如果看起来像 OCR 错误（如 3 变成 5），请修正为最可能的常见值（通常是 3 或 4）。
       - 格式提取：优先寻找 "限选 - 3 学分", "必修 2 学分", "3 学分" 等明确字样。
    
    3. **成绩 (Score) 识别逻辑**:
       - 目标：**最终总成绩**。
       - 特征：通常是行尾的、字号较大的、蓝色的（在原图中）、介于 0-100 之间的整数。
       - **排除干扰**：绝对忽略“平时成绩: 90”、“期中: 88”、“绩点: 4.5”等干扰项。如果一行有 [38, 4.5, 95]，取 **95**。
    
    4. **去重与合并**:
       - 输入包含两次识别结果（Source A/B）。请整合两者的信息，输出一份干净的、不重复的课程列表。
    
    === 输出格式 ===
    仅输出标准的 JSON 列表，严禁包含 ```json 代码块标记或其他废话。
    示例:
    [
        {{"subject": "习近平新时代中国特色社会主义思想概论", "score": 94, "credit": 3.0}},
        {{"subject": "形势与政策", "score": 86, "credit": 0.0}}
    ]
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a precise data extraction engine. Correct OCR errors based on logic. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        # 强力清洗 markdown 标记
        content = content.replace("```json", "").replace("```", "").strip()
            
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

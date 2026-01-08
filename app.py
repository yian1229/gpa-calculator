import streamlit as st
import pandas as pd
from PIL import Image
import os
import shutil
from ocr_helper import perform_ocr, parse_with_deepseek, DEFAULT_TESSERACT_PATH
from gpa_calculator import calculate_gpa

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


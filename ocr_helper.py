import pytesseract
from PIL import Image
from openai import OpenAI
import json
import io
import shutil
import os

# 默认不指定路径，依赖系统 PATH (适用于 Linux/Cloud)
# 仅在 Windows 本地测试时可能需要指定路径
DEFAULT_TESSERACT_PATH = None

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
        # 识别中文和英文
        # 注意：packages.txt 确保了云端环境安装了 chi_sim
        text = pytesseract.image_to_string(image, lang='chi_sim+eng')
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

    prompt = f"""
    你是一个数据提取助手。请从下面的 OCR 识别文本中提取“科目名称”、“成绩”和“学分”。
    
    OCR 文本内容：
    {ocr_text}
    
    规则：
    1. 识别并提取所有科目的名称、成绩（分数）和学分。
    2. 如果文本中有噪音或乱码，请利用上下文修正。
    3. 输出格式必须是标准的 JSON 列表，不要包含 Markdown 格式（如 ```json ... ```）。
    4. 每个列表项包含 keys: "subject" (string), "score" (float), "credit" (float)。
    5. 如果没有识别到有效数据，返回空列表 []。
    
    例如：
    [
        {{"subject": "高等数学", "score": 85, "credit": 4.0}},
        {{"subject": "大学英语", "score": 90, "credit": 2.0}}
    ]
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs raw JSON."},
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

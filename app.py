import streamlit as st
import requests
import json
import os
import tempfile
from pydub import AudioSegment
from openai import OpenAI
import docx
import PyPDF2
import unicodedata
import re

# --- Configuration & Constants ---
OPENAI_API_KEY = os.getenv("GEMINI_API_KEY")
MINIMAX_GROUP_ID = os.getenv("MINIMAX_GROUP_ID")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")

VOICE_OPTIONS = {
    "青涩青年音色": "male-qn-qingse", "精英青年音色": "male-qn-jingying", "霸道青年音色": "male-qn-badao",
    "青年大学生音色": "male-qn-daxuesheng", "少女音色": "female-shaonv", "御姐音色": "female-yujie",
    "成熟女性音色": "female-chengshu", "甜美女性音色": "female-tianmei", "男性主持人": "presenter_male",
    "女性主持人": "presenter_female", "男性有声书1": "audiobook_male_1", "男性有声书2": "audiobook_male_2",
    "女性有声书1": "audiobook_female_1", "女性有声书2": "audiobook_female_2",
    "青涩青年音色-beta": "male-qn-qingse-jingpin", "精英青年音色-beta": "male-qn-jingying-jingpin",
    "霸道青年音色-beta": "male-qn-badao-jingpin", "青年大学生音色-beta": "male-qn-daxuesheng-jingpin",
    "少女音色-beta": "female-shaonv-jingpin", "御姐音色-beta": "female-yujie-jingpin",
    "成熟女性音色-beta": "female-chengshu-jingpin", "甜美女性音色-beta": "female-tianmei-jingpin",
    "聪明男童": "clever_boy", "可爱男童": "cute_boy", "萌萌女童": "lovely_girl", "卡通猪小琪": "cartoon_pig",
    "病娇弟弟": "bingjiao_didi", "俊朗男友": "junlang_nanyou", "纯真学弟": "chunzhen_xuedi",
    "冷淡学长": "lengdan_xiongzhang", "霸道少爷": "badao_shaoye", "甜心小玲": "tianxin_xiaoling",
    "俏皮萌妹": "qiaopi_mengmei", "妩媚御姐": "wumei_yujie", "嗲嗲学妹": "diadia_xuemei",
    "淡雅学姐": "danya_xuejie"
}

DIALOGUE_STYLES = {
    "轻松幽默": "以轻松、幽默的方式进行对话，加入适当的笑话和轻松的语气。",
    "专业深入": "以专业、深入的方式探讨主题，注重细节和逻辑分析。",
    "生动叙事": "以故事化、形象化的方式叙述，增强听众的沉浸感。",
    "激烈争辩": "以对立、争辩的风格展开对话，突出不同观点的碰撞。",
    "温暖治愈": "用温柔、暖心的语言传递情感，注重共情与安抚，适合分享心事、给予鼓励的场景。",
    "简洁干练": "语言直截了当、聚焦核心信息，适合快速沟通、传达指令或高效解决问题的场景。",
    "文艺诗意": "运用优美修辞与意象化表达，营造浪漫或哲思氛围，适合文学交流、情感抒发的场景。",
    "萌系软萌": "用俏皮可爱的语气和词汇，搭配撒娇或萌系表达，适合轻松娱乐、互动调侃的场景。",
    "悬疑探秘": "用神秘悬疑的语调设置悬念，引导好奇，适合故事讲述、推理讨论或剧情分析的场景。",
    "励志燃情": "充满激情与力量感，用短句、排比激发斗志，适合演讲、激励他人或自我打气的场景。",
    "吐槽犀利": "用幽默讽刺、一针见血的语言点评现象，带点毒舌趣味，适合调侃热点、分享观点的场景。"
}

MINIMAX_API_URL_TEMPLATE = "https://api.minimax.chat/v1/t2a_v2?GroupId={group_id}"

# --- Initialize Session State ---
if 'extracted_content' not in st.session_state:
    st.session_state.extracted_content = None
if 'dialogue_script' not in st.session_state:
    st.session_state.dialogue_script = None
if 'final_audio_path' not in st.session_state:
    st.session_state.final_audio_path = None
if 'json_script_data' not in st.session_state:
    st.session_state.json_script_data = None
if 'edited_dialogue' not in st.session_state:
    st.session_state.edited_dialogue = None
if 'character_recommendations' not in st.session_state:
    st.session_state.character_recommendations = None

# --- Helper Functions ---
def sanitize_filename(filename):
    """
    Sanitize a filename to prevent encoding issues:
    1. Normalize Unicode characters
    2. Replace spaces with underscores
    3. Remove characters that might cause issues
    4. Ensure the filename has a valid extension
    """
    # Normalize unicode characters
    clean_name = unicodedata.normalize('NFKD', filename)
    # Replace spaces and remove problematic characters
    clean_name = re.sub(r'[^\w\s.-]', '', clean_name).replace(' ', '_')
    
    # Ensure we keep the original extension
    original_ext = os.path.splitext(filename)[1].lower()
    if original_ext in ['.txt', '.pdf', '.docx']:
        # Make sure the extension is preserved correctly
        base = os.path.splitext(clean_name)[0]
        clean_name = f"{base}{original_ext}"
        
    return clean_name

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file (.txt, .pdf, .docx)."""
    try:
        # Sanitize the filename
        original_filename = sanitize_filename(uploaded_file.name)
        file_extension = os.path.splitext(original_filename)[1].lower()
        
        # Check if extension is supported
        if file_extension not in ['.txt', '.pdf', '.docx']:
            st.error("不支持的文件类型。请上传 .txt, .pdf 或 .docx 文件")
            return None
            
        # Create a temporary file and save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_filepath = temp_file.name
        
        # Process the file based on its extension
        if file_extension == '.txt':
            with open(temp_filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_extension == '.pdf':
            text = ""
            pdf_reader = PyPDF2.PdfReader(temp_filepath)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text() or ""
                text += page_text
        elif file_extension == '.docx':
            doc = docx.Document(temp_filepath)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
        # Clean up the temporary file
        os.unlink(temp_filepath)
        
        # Log extracted text for debugging
        st.write(f"提取的文本 (前100字符): {text[:100]}...")
        
        # Check if text is empty or invalid
        if not text.strip():
            st.error("从文件中提取的文本为空或无效。")
            return None
            
        return text
    except Exception as e:
        st.error(f"处理文件时出错：{e}")
        return None

def generate_dialogue_openai(content, char1_name, char2_name, dialogue_style, model="gemini-2.0-flash"):
    """Generate dialogue using OpenAI API."""
    client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/")
    style_instruction = DIALOGUE_STYLES[dialogue_style]
    prompt = f"""
    根据以下内容，为两个角色（{char1_name} 和 {char2_name}）生成一段引人入胜的播客对话。
    对话应以{style_instruction}的方式，深入探讨内容的主题和信息，可以适当展开话题，以对话形式呈现,可适当增加一些口语，增加真实性。
    基础分析结构
    简明概述：以简单术语解释核心概念
    重要案例研究：历史与现代应用实例
    当前实践方法：科学有效性与方法论
    未来影响：预测发展方向与应用可能
    分析方法（费曼技巧）
    将复杂概念简化为通俗易懂的解释
    识别知识盲点并深入研究
    通过类比和实例连接相关概念
    为每个发现提供实际应用案例
    设计引导思考的练习问题
    建立清晰的心智模型
    每个关键点用3句话总结
    战略思维框架
    预测分析：探究不同背景下的多种方法与基本假设
    监控观察：识别模式并与最佳实践比较
    选择应用：从熟悉案例过渡到创新框架
    顺序安排：策略性地构建学习路径
    联系整合：通过分解、测试和迭代，将表面差异与核心原则连接起来
    文字里不要包含动作或者表情细节，因为播客是文字形式。最多15轮对话。
    输出格式为 JSON 列表，每个对象包含 "speaker" 键（值为 "{char1_name}" 或 "{char2_name}"）和 "line" 键（角色所说的话）。
    不要在 JSON 列表之外包含任何其他文本或说明。

    内容：
    ---
    {content}
    ---

    JSON 输出：
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个富有创意的播客脚本作者。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        dialogue_json_str = response.choices[0].message.content.strip()
        
        if dialogue_json_str.startswith("```json"):
            dialogue_json_str = dialogue_json_str[7:]
        if dialogue_json_str.endswith("```"):
            dialogue_json_str = dialogue_json_str[:-3]
        dialogue_json_str = dialogue_json_str.strip()

        dialogue = json.loads(dialogue_json_str)
        if not isinstance(dialogue, list) or not all(
            isinstance(item, dict) and "speaker" in item and "line" in item for item in dialogue
        ):
            st.error(f"AI 返回的对话格式不正确。原始输出：{dialogue_json_str}")
            return None
        return dialogue
    except json.JSONDecodeError:
        st.error(f"无法解析 AI 响应为 JSON。原始输出：{dialogue_json_str}")
        return None
    except Exception as e:
        st.error(f"使用 OpenAI 生成对话时出错：{e}")
        return None

def recommend_characters_and_voices(content):
    """Analyze content and recommend character names, voices, and dialogue style using OpenAI."""
    client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/")
    prompt = f"""
    根据以下内容，推荐两个适合进行播客对话的角色名字（例如，名字应反映内容主题或角色背景），
    每个角色的音色（从以下音色列表中选择：{', '.join(VOICE_OPTIONS.keys())}），
    以及一个适合内容的对话风格（从以下风格中选择：{', '.join(DIALOGUE_STYLES.keys())}）。
    输出 JSON 格式，包含两个角色，每个角色有 "name" 和 "voice" 键，以及一个 "dialogue_style" 键表示推荐的对话风格。
    
    内容：
    ---
    {content}
    ---

    JSON 输出：
    {{
        "characters": [
            {{"name": "角色1名字", "voice": "音色名称"}},
            {{"name": "角色2名字", "voice": "音色名称"}}
        ],
        "dialogue_style": "对话风格名称"
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": "你是一个擅长分析文本并推荐播客角色和对话风格的 AI。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        response_json_str = response.choices[0].message.content.strip()
        
        if response_json_str.startswith("```json"):
            response_json_str = response_json_str[7:]
        if response_json_str.endswith("```"):
            response_json_str = response_json_str[:-3]
        response_json_str = response_json_str.strip()

        recommendations = json.loads(response_json_str)
        if (
            not isinstance(recommendations, dict) or
            "characters" not in recommendations or
            "dialogue_style" not in recommendations or
            not isinstance(recommendations["characters"], list) or
            len(recommendations["characters"]) != 2 or
            not all(
                isinstance(item, dict) and "name" in item and "voice" in item and item["voice"] in VOICE_OPTIONS
                for item in recommendations["characters"]
            ) or
            recommendations["dialogue_style"] not in DIALOGUE_STYLES
        ):
            st.warning(f"AI 推荐的格式不正确，将使用默认值。原始输出：{response_json_str}")
            return {
                "characters": [
                    {"name": "Alice", "voice": "少女音色"},
                    {"name": "Bob", "voice": "青涩青年音色"}
                ],
                "dialogue_style": "轻松幽默"
            }
        return recommendations
    except Exception as e:
        st.warning(f"推荐角色和对话风格时出错，将使用默认值：{e}")
        return {
            "characters": [
                {"name": "Alice", "voice": "少女音色"},
                {"name": "Bob", "voice": "青涩青年音色"}
            ],
            "dialogue_style": "轻松幽默"
        }

def text_to_speech_minimax(text_to_speak, voice_id):
    """Generate speech using Minimax API and return audio content."""
    url = MINIMAX_API_URL_TEMPLATE.format(group_id=MINIMAX_GROUP_ID)
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "speech-02-turbo",
        "text": text_to_speak,
        "timber_weights": [
            {
                "voice_id": voice_id,
                "weight": 100
            }
        ],
        "voice_setting": {
            "voice_id": "",
            "speed": 1.0,
            "pitch": 0,
            "vol": 1.0,
            "latex_read": False
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3"
        },
        "language_boost": "auto"
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        parsed_json = response.json()
        
        if 'data' not in parsed_json or 'audio' not in parsed_json['data']:
            st.error(f"Minimax API 未返回音频数据。响应：{response.text}")
            return None
        
        audio_hex = parsed_json['data']['audio']
        try:
            audio_content = bytes.fromhex(audio_hex)
            return audio_content
        except ValueError as ve:
            st.error(f"无法将音频数据从十六进制解码：{ve}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"调用 Minimax API 时出错：{e}")
        return None
    except json.JSONDecodeError as je:
        st.error(f"无法解析 Minimax API 响应为 JSON：{response.text}")
        return None

def concatenate_audio_files(audio_files_paths, output_path):
    """Concatenate multiple MP3 files into one."""
    if not audio_files_paths:
        return None
    
    combined = AudioSegment.empty()
    try:
        for file_path in audio_files_paths:
            segment = AudioSegment.from_mp3(file_path)
            combined += segment
        combined.export(output_path, format="mp3")
        return output_path
    except Exception as e:
        st.error(f"音频合并时出错：{e}")
        st.error("请确保已安装 ffmpeg 并将其添加到系统 PATH 中。")
        return None

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="AI 播客生成器")
st.title("🎙️ AI 播客生成器")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ 配置")
    
    st.subheader("📜 内容输入")
    uploaded_file = st.file_uploader("上传内容（txt, pdf, docx）", type=["txt", "pdf", "docx"])
    st.caption("提示：如果上传包含中文或特殊字符的文件名时遇到错误，请尝试将文件名修改为仅包含英文字母和数字，然后重新上传。")
    raw_text_input = st.text_area("或在此粘贴文本内容", height=150)

    st.divider()

    st.subheader("🗣️ 播客角色(点击下方生成对话脚本会自动选择合适的persona)")

    if st.session_state.extracted_content and not st.session_state.character_recommendations:
        with st.spinner("正在分析内容并推荐角色和对话风格..."):
            st.session_state.character_recommendations = recommend_characters_and_voices(st.session_state.extracted_content)
    elif not st.session_state.extracted_content:
        st.info("请先上传文件或输入文本内容以生成角色和对话风格推荐。")

    default_char1 = (
        st.session_state.character_recommendations["characters"][0]
        if st.session_state.character_recommendations
        else {"name": "Alice", "voice": "少女音色"}
    )
    default_char2 = (
        st.session_state.character_recommendations["characters"][1]
        if st.session_state.character_recommendations
        else {"name": "Bob", "voice": "青涩青年音色"}
    )
    default_dialogue_style = (
        st.session_state.character_recommendations["dialogue_style"]
        if st.session_state.character_recommendations
        else "轻松幽默"
    )

    st.write(f"**角色 1（推荐：{default_char1['name']}, 音色：{default_char1['voice']}）**")
    char1_name = st.text_input("角色 1 姓名", value=default_char1['name'], key="char1_name")
    char1_voice_name = st.selectbox(
        f"{char1_name} 的音色",
        options=list(VOICE_OPTIONS.keys()),
        index=list(VOICE_OPTIONS.keys()).index(default_char1['voice']),
        key="char1_voice"
    )
    char1_voice_id = VOICE_OPTIONS[char1_voice_name]

    st.write(f"**角色 2（推荐：{default_char2['name']}, 音色：{default_char2['voice']}）**")
    char2_name = st.text_input("角色 2 姓名", value=default_char2['name'], key="char2_name")
    char2_voice_name = st.selectbox(
        f"{char2_name} 的音色",
        options=list(VOICE_OPTIONS.keys()),
        index=list(VOICE_OPTIONS.keys()).index(default_char2['voice']),
        key="char2_voice"
    )
    char2_voice_id = VOICE_OPTIONS[char2_voice_name]

    st.subheader("💬 对话风格")
    st.write(f"**推荐对话风格：{default_dialogue_style}**")
    dialogue_style = st.selectbox(
        "选择对话风格",
        options=list(DIALOGUE_STYLES.keys()),
        index=list(DIALOGUE_STYLES.keys()).index(default_dialogue_style),
        key="dialogue_style"
    )
    
    st.divider()
    generate_dialogue_button = st.button("📝 生成对话脚本", type="primary", use_container_width=True)

# --- Main Area for Output ---
# 1. Display extracted content if file uploaded
if uploaded_file:
    with st.spinner("正在从文件中提取文本..."):
        content = extract_text_from_file(uploaded_file)
        if content:
            st.session_state.extracted_content = content
            st.subheader("📄 提取的内容")
            st.text_area("提取的文本", content, height=200, disabled=True)
        else:
            st.error("无法从上传的文件中提取内容。")
            st.stop()
elif raw_text_input:
    st.session_state.extracted_content = raw_text_input

# 2. Generate dialogue
if generate_dialogue_button:
    st.session_state.dialogue_script = None
    st.session_state.final_audio_path = None
    st.session_state.json_script_data = None
    st.session_state.edited_dialogue = None

    content = st.session_state.extracted_content
    if not content:
        st.warning("请上传文件或粘贴文本内容。")
        st.stop()

    with st.spinner(f"AI 正在为 {char1_name} 和 {char2_name} 编写对话...（这可能需要一些时间）"):
        dialogue = generate_dialogue_openai(content, char1_name, char2_name, dialogue_style)
    
    if not dialogue:
        st.error("生成对话失败。")
        st.stop()

    st.session_state.dialogue_script = dialogue
    st.session_state.edited_dialogue = json.dumps(dialogue, indent=2, ensure_ascii=False)
    st.session_state.json_script_data = st.session_state.edited_dialogue.encode('utf-8')

# 3. Display and edit dialogue
if st.session_state.dialogue_script:
    st.subheader("💬 生成的对话脚本（可编辑）")
    edited_text = st.text_area(
        "编辑对话脚本（JSON 格式）",
        st.session_state.edited_dialogue,
        height=400
    )
    
    try:
        edited_dialogue = json.loads(edited_text)
        if not isinstance(edited_dialogue, list) or not all(
            isinstance(item, dict) and "speaker" in item and "line" in item for item in edited_dialogue
        ):
            st.error("编辑后的对话格式不正确。请确保为有效的 JSON 列表，每个对象包含 'speaker' 和 'line' 键。")
        else:
            st.session_state.edited_dialogue = edited_text
            st.session_state.json_script_data = edited_text.encode('utf-8')
            
            st.download_button(
                label="📥 下载对话脚本 (JSON)",
                data=st.session_state.json_script_data,
                file_name="podcast_script.json",
                mime="application/json"
            )
            
            if st.button("🚀 生成播客", type="primary"):
                st.session_state.final_audio_path = None
                
                dialogue = edited_dialogue
                individual_audio_files = []
                temp_dir = tempfile.mkdtemp()

                generation_errors = False
                with st.spinner("正在将对话转换为语音并组装播客..."):
                    progress_bar = st.progress(0)
                    total_lines = len(dialogue)

                    for i, turn in enumerate(dialogue):
                        speaker_name = turn.get("speaker")
                        line_text = turn.get("line")

                        if not speaker_name or not line_text:
                            st.warning(f"跳过无效对话片段：{turn}")
                            continue

                        voice_id_to_use = char1_voice_id if speaker_name == char1_name else char2_voice_id
                        
                        status_placeholder = st.empty()
                        status_placeholder.info(f"正在为 {speaker_name} 生成音频：\"{line_text[:30]}...\"")
                        
                        audio_content = text_to_speech_minimax(line_text, voice_id_to_use)
                        
                        if audio_content:
                            temp_audio_path = os.path.join(temp_dir, f"line_{i+1}.mp3")
                            with open(temp_audio_path, "wb") as f:
                                f.write(audio_content)
                            individual_audio_files.append(temp_audio_path)
                        else:
                            st.error(f"无法为以下内容生成音频：{speaker_name} - \"{line_text}\"")
                            generation_errors = True
                        
                        progress_bar.progress((i + 1) / total_lines)
                        status_placeholder.empty()

                    if generation_errors and not individual_audio_files:
                        st.error("未生成任何音频文件，无法创建播客。")
                        st.stop()
                    elif generation_errors:
                        st.warning("部分音频生成失败，将使用已生成的音频继续。")

                    if individual_audio_files:
                        final_podcast_path = os.path.join(temp_dir, "final_podcast.mp3")
                        concatenated_audio = concatenate_audio_files(individual_audio_files, final_podcast_path)
                        if concatenated_audio:
                            st.session_state.final_audio_path = concatenated_audio
                            st.success("播客生成成功！")
                        else:
                            st.error("音频文件合并失败。")
                    else:
                        st.error("未成功生成任何音频片段，无法创建播客。")
    except json.JSONDecodeError:
        st.error("编辑的对话不是有效的 JSON 格式。请检查格式并重试。")

# 4. Display final podcast
if st.session_state.final_audio_path:
    st.subheader("🎧 收听播客")
    try:
        with open(st.session_state.final_audio_path, "rb") as ap:
            st.audio(ap.read(), format="audio/mp3")
        
        with open(st.session_state.final_audio_path, "rb") as fp:
            st.download_button(
                label="📥 下载播客 (MP3)",
                data=fp,
                file_name="generated_podcast.mp3",
                mime="audio/mpeg"
            )
    except FileNotFoundError:
        st.error("未找到最终音频文件。生成或清理过程中可能出现问题。")
    except Exception as e:
        st.error(f"显示或下载音频时出错：{e}")

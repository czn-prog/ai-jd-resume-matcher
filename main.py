import gradio as gr
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# 加载密钥
load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    model=os.getenv("LLM_MODEL"),
    temperature=0.3
)

# Prompt模板1：JD解析
jd_prompt = PromptTemplate(
    input_variables=["jd_text"],
    template="""
你是资深AI招聘产品，分析下面岗位JD，输出结构化内容：
1. 核心硬性要求（学历、工作年限、专业、技能）
2. 岗位核心业务能力
3. AI相关能力要求
4. 加分项
JD内容：{jd_text}
只输出清晰结构化文本，不要多余话术。
"""
)

# Prompt模板2：简历匹配打分
match_prompt = PromptTemplate(
    input_variables=["jd_info", "resume_text"],
    template="""
基于JD要求，对简历进行匹配评估：
JD信息：{jd_info}
简历内容：{resume_text}
输出内容：
1. 综合匹配分数（0-100）
2. 简历优势点
3. 缺失能力/关键词
4. 简历优化建议（量化成果、补充AI相关经历）
5. 模拟3道针对性面试提问
"""
)

# Prompt模板3：批量评测打分
eval_prompt = PromptTemplate(
    input_variables=["jd_info", "resume_text"],
    template="""
仅输出JSON格式，字段：match_score(0-100), missing_count(缺失能力数量), ai_match(0-10 AI匹配度)
JD：{jd_info}
简历：{resume_text}
"""
)

# 业务函数
def parse_jd(jd_text):
    chain = jd_prompt | llm
    res = chain.invoke({"jd_text": jd_text})
    return res.content

def match_resume(jd_info, resume_text):
    if not jd_info or not resume_text:
        return "请先解析JD，再输入简历内容"
    chain = match_prompt | llm
    res = chain.invoke({"jd_info": jd_info, "resume_text": resume_text})
    return res.content

def batch_eval(jd_info, csv_file):
    if not jd_info:
        return "请先解析JD", None
    df = pd.read_csv(csv_file)
    result_list = []
    for row in df.to_dict("records"):
        resume = row["resume"]
        chain = eval_prompt | llm
        eval_res = chain.invoke({"jd_info": jd_info, "resume_text": resume})
        result_list.append({"resume": resume, "eval": eval_res.content})
    out_df = pd.DataFrame(result_list)
    out_path = "output/test_result.csv"
    os.makedirs("output", exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return "批量评测完成，文件已导出", out_path

# 页面搭建
with gr.Blocks(title="AI简历JD匹配助手") as demo:
    gr.Markdown("# AI简历-JD智能匹配工具（AI产品求职Demo）")
    gr.Markdown("## 普通求职者端：JD解析 + 简历匹配优化")
    with gr.Row():
        with gr.Column(scale=1):
            jd_input = gr.Textbox(label="粘贴岗位JD", lines=8)
            parse_btn = gr.Button("1.解析JD需求")
            jd_out = gr.Textbox(label="JD结构化拆解结果", lines=10)
            parse_btn.click(parse_jd, inputs=[jd_input], outputs=[jd_out])
        with gr.Column(scale=1):
            resume_input = gr.Textbox(label="粘贴你的简历全文", lines=8)
            match_btn = gr.Button("2.简历匹配打分&优化")
            match_out = gr.Textbox(label="匹配报告+面试题", lines=12)
            match_btn.click(match_resume, inputs=[jd_out, resume_input], outputs=[match_out])

    gr.Markdown("## 产品后台评测模块（AI产品核心能力）")
    with gr.Row():
        csv_upload = gr.File(label="批量简历CSV文件", file_types=[".csv"])
        eval_btn = gr.Button("批量AI评测，输出报表")
    eval_msg = gr.Textbox(label="评测状态")
    eval_file = gr.File(label="下载评测结果")
    eval_btn.click(batch_eval, inputs=[jd_out, csv_upload], outputs=[eval_msg, eval_file])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

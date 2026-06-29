import os
import time
import json
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 加载.env密钥
load_dotenv()
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL_NAME = os.getenv("LLM_MODEL")

# 自动创建输出文件夹
if not os.path.exists("output"):
    os.makedirs("output")

# 初始化智谱GLM大模型
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.3
)

# 匹配打分Prompt模板
match_prompt = ChatPromptTemplate.from_template("""
你是资深招聘HR，根据岗位JD与简历进行匹配打分，满分100分。
打分维度：
1. 核心技能匹配（40分）
2. 工作年限&行业经验（30分）
3. 学历专业匹配（20分）
4. 项目综合能力（10分）

岗位JD：{jd_text}
简历内容：{resume_text}

仅输出标准JSON，无多余文字，格式如下：
{{
    "score": 数字分数,
    "analysis": "分段说明匹配优势与短板"
}}
""")

# 单份简历AI打分函数
def single_resume_match(jd_text, resume_text):
    if not jd_text.strip() or not resume_text.strip():
        return "请输入完整JD和简历内容", ""
    try:
        resp = llm.invoke(match_prompt.format(jd_text=jd_text, resume_text=resume_text))
        data = json.loads(resp.content)
        score = data["score"]
        analysis = f"匹配总分：{score}分\n{data['analysis']}"
        return analysis, f"{score} 分"
    except Exception as e:
        return f"AI调用失败：{str(e)}", "0 分"

# 批量CSV评测核心函数
def batch_evaluate(csv_file, jd_text):
    if not csv_file:
        return "错误：请上传简历CSV文件", None, None
    if not jd_text.strip():
        return "错误：请填写目标岗位JD", None, None

    # 读取CSV，兼容utf-8/gbk
    try:
        df = pd.read_csv(csv_file, encoding="utf-8")
    except:
        df = pd.read_csv(csv_file, encoding="gbk")

    # 校验必填表头
    need_cols = ["id", "name", "work_year", "education", "major", "skills", "work_experience"]
    lack = [c for c in need_cols if c not in df.columns]
    if lack:
        return f"CSV缺失字段：{','.join(lack)}", None, None

    # 空值填充
    df = df.fillna("无")
    total = len(df)
    result_rows = []

    for idx, row in df.iterrows():
        try:
            resume_full = f"""
姓名：{row['name']}
工作年限：{row['work_year']}
学历：{row['education']}
专业：{row['major']}
技能：{row['skills']}
工作经历：{row['work_experience']}
            """
            resp = llm.invoke(match_prompt.format(jd_text=jd_text, resume_text=resume_full))
            res = json.loads(resp.content)
            result_rows.append({
                "id": row["id"],
                "name": row["name"],
                "match_score": res["score"],
                "match_detail": res["analysis"],
                "status": "成功"
            })
            print(f"进度 {idx+1}/{total} {row['name']} 完成")
            time.sleep(0.6)
        except Exception as e:
            result_rows.append({
                "id": row["id"],
                "name": row["name"],
                "match_score": 0,
                "match_detail": f"处理异常：{str(e)}",
                "status": "失败"
            })
            continue

    # 导出结果CSV
    res_df = pd.DataFrame(result_rows)
    save_path = "output/批量评测结果.csv"
    res_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    # 统计汇总
    succ = len(res_df[res_df["status"] == "成功"])
    fail = len(res_df[res_df["status"] == "失败"])
    avg_score = round(res_df[res_df[res_df["match_score"] > 0]["match_score"].mean(), 2]) if succ > 0 else 0
    summary = f"""
# 批量评测完成
- 总简历数量：{total} 份
- 成功处理：{succ} 份
- 失败条目：{fail} 份
- 平均匹配分数：{avg_score} 分
结果文件已保存至 output/批量评测结果.csv
"""
    return summary, res_df, save_path

# 搭建Gradio页面（修复Tabs拆包报错）
with gr.Blocks(title="AI简历-JD智能匹配系统") as demo:
    gr.Markdown("# AI简历JD智能匹配工具")
    with gr.Tabs():
        # 标签1：单条匹配
        with gr.Tab("单份简历匹配"):
            jd_input = gr.Textbox(label="岗位JD内容", lines=8, placeholder="粘贴招聘岗位要求")
            resume_input = gr.Textbox(label="简历文本", lines=10, placeholder="粘贴简历全文")
            run_btn = gr.Button("一键匹配打分", variant="primary")
            score_out = gr.Textbox(label="匹配得分")
            analysis_out = gr.Textbox(label="匹配分析详情", lines=12)
            run_btn.click(single_resume_match, inputs=[jd_input, resume_input], outputs=[analysis_out, score_out])

        # 标签2：批量评测模块
        with gr.Tab("后台批量评测模块"):
            gr.Markdown("## CSV批量简历评测")
            batch_jd = gr.Textbox(label="统一岗位JD", lines=6)
            upload_csv = gr.File(label="上传简历CSV文件", file_types=[".csv"])
            start_batch = gr.Button("启动批量评测", variant="primary")
            summary_md = gr.Markdown(label="评测汇总信息")
            result_table = gr.DataFrame(label="匹配结果明细")
            file_download = gr.File(label="下载评测报表")
            start_batch.click(batch_evaluate, inputs=[upload_csv, batch_jd], outputs=[summary_md, result_table, file_download])

# 启动服务
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

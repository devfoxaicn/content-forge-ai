# =====================
# Imports
# =====================
import os
import re
import sys
import time
import json
import hashlib
import logging
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType
from power_ai.database.cos import COS
from power_ai.office.feishu.document import BiTable, Table
from power_ai.platform.aip import AIPWorkflow, block_until_all_success
from power_ai.office.feishu.robot import CustomRobot, ApplicationRobot
from nio.fms import fms_client
from urllib.parse import urlparse
from openai import OpenAI



# =====================
# Argument Parsing
# =====================
parser = argparse.ArgumentParser()
parser.add_argument('-dt', default='20250410') 
parser.add_argument('-model_version', default='v0')  
args = parser.parse_args()

dt = args.dt  
model_version = args.model_version

format_dt = f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}"
next_day = (datetime.strptime(dt, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")

print('*'*10, 'dt, model_version: ', dt, format_dt, next_day, model_version)


# =====================
# Spark Session Init
# =====================
spark = SparkSession.builder \
        .appName("NIO_Drive_Quality_Control") \
        .enableHiveSupport() \
        .getOrCreate()

# Hive写入配置
spark.conf.set("hive.exec.dynamic.partition", "true")
spark.conf.set("hive.exec.dynamic.partition.mode", "nonstrict")
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

# =====================
# SQL Query
# =====================
sql_query = f'''
WITH ranked_data AS (
    SELECT
        fl_ad_account,
        fl_tag,
        cal_month,
        latest_partition_end_time,
        ROW_NUMBER() OVER (
            PARTITION BY fl_ad_account
            ORDER BY cal_month DESC
        ) AS rn
    FROM ads.uo_con_fl_quadrant_eval_rpt_1h_f
    WHERE datetime = 'latest'
      AND fl_tag IN ('第2象限：结果差过程好', '第3象限：结果差过程差')
      AND x_index = '锁单达成率'
      AND y_index = '用户运营过程复合指标'
      -- 使用 Presto 的日期函数：过滤掉当月（取 < 本月月初）
      AND cal_month < date(date_trunc('month', current_date))
),
xy_fl AS (
    SELECT
        fl_ad_account
    FROM ranked_data
    WHERE rn = 1
),
test_drive_orders AS (
    -- 试驾单筛选：只保留 fl 命中的订单
    SELECT
        td_order_no, -- 试驾单号
        td_accompany_fl_ad_account
    FROM dwd.ue_test_drive_order_entity_1h_f
    WHERE datetime = 'latest'
      AND td_order_status = 4
      AND date(td_actual_end_time) = date('{format_dt}')
      AND td_accompany_fl_ad_account IN (
          SELECT fl_ad_account FROM xy_fl
      )
)
SELECT
    lower(td.td_brand_code) AS brand,
    td.td_order_no AS order_no,
    fl.td_accompany_fl_ad_account AS fl_ad_account,
    a.order_no AS asr_no,
    b.order_no AS rating_no,
    a.text_url,
    CASE
        WHEN a.brand = 'nio' THEN get_json_object(info, '$.merge_detail.log_path')
        WHEN a.brand = 'alps' THEN get_json_object(info, '$.merge_recording.url')
    END AS audio_url,
    CASE
        WHEN a.brand = 'nio' THEN get_json_object(info, '$.merge_detail.duration')
        WHEN a.brand = 'alps' THEN get_json_object(info, '$.merge_recording.duration')
    END AS duration
FROM (
	SELECT  td_brand_code
	       ,td_order_no -- 试驾单号
	       ,td_actual_end_time
	FROM dwm.ue_test_drive_participant_info_wide_1h_f
	WHERE datetime = '{dt}23'
	AND date(td_actual_end_time) = date('{format_dt}')
	AND td_brand_code IN ('nio', 'fy')
	GROUP BY  td_brand_code
	         ,td_order_no
	         ,td_actual_end_time
) td
JOIN test_drive_orders fl
    ON td.td_order_no = fl.td_order_no
LEFT JOIN (
    -- asr数据，包含3个品牌
    SELECT
        id, -- 自增ID
        brand, -- 品牌
        order_no, -- 试驾单号
        status, -- 录音处理状态
        fms_task_id, -- fms拼接任务ID
        asr_result, -- asr结果
        creator, -- 创建人
        create_time, -- 创建时间
        updater, -- 更新人
        update_time, -- 更新时间
        latest_partition_end_time,
        get_json_object(asr_result, '$.content_link') AS text_url
    FROM ods_helios_prod_all_prod.test_drive_recording_processing_info_1h_a
    WHERE datetime = '{next_day}03'
) a
    ON td.td_order_no = a.order_no
LEFT JOIN (
    -- 录音质检数据，包含 ALPS
    SELECT
        id, -- 自增主键
        order_no, -- 试驾单号
        type, -- 质检类型
        info, -- 质检信息
        creator, -- 创建人
        create_time, -- 创建时间
        updater, -- 更新人
        update_time, -- 更新时间
        latest_partition_end_time
    FROM ods_helios_prod.test_drive_rating_1h_a
    WHERE datetime = '{next_day}03'
) b
    ON td.td_order_no = b.order_no
-- LIMIT 30
'''
print(sql_query)

# =====================
# Data Extraction
# =====================
print("正在执行Spark SQL查询...")
spark_df = spark.sql(sql_query)
pandas_df = spark_df.toPandas()
print('*'*20, pandas_df.shape, pandas_df.head(5))

# 统计数量并输出日志（仅计数，不去重、不清洗）
total_rows = len(pandas_df)
order_cnt = pandas_df['order_no'].notna().sum() if 'order_no' in pandas_df.columns else 0
asr_cnt = pandas_df['asr_no'].notna().sum() if 'asr_no' in pandas_df.columns else 0
rating_cnt = pandas_df['rating_no'].notna().sum() if 'rating_no' in pandas_df.columns else 0

print('[统计] 行数:', total_rows)
print('[统计] 数量 - order_no:', order_cnt, ' asr_no:', asr_cnt, ' rating_no:', rating_cnt)

# =====================
# LLM API相关配置
# =====================
appId = 10008
appSecret = "4cfdb3115a617a508ef584f896c08133"

dict_aip_model = {
    'qwen3-32b': 'http://qwen-test-32b-inference-prod.idchl-gpu.nioint.com/v1',
    'qwen3-30b-2507': 'http://qwen-30b-2507-inference-prod.idchl-gpu.nioint.com/v1',
    'qwen3_235b': 'http://qwen-235b-2507-inference-prod.idchl-gpu.nioint.com/v1/',
}

def query_openai(query: str, model_name: str = 'qwen3_235b', no_think: bool = True) -> dict:
    # 关闭思考模式
    if no_think and '/no_think' not in query[-20:]:
        query = query + ' /no_think'
    # 初始化OpenAI客户端，配置自定义端点
    url = dict_aip_model[model_name]
    client = OpenAI(
        base_url=url,
        api_key="none"
    )
    # 请求接口
    response = client.chat.completions.create(
        model="qwen",
        temperature=0.1,
        messages=[
            {'role': 'user', 'content': query}
        ],
        timeout=100000000
    )
    content = response.choices[0].message.content
    cleaned_content = content.replace('<think>\n\n</think>\n\n', '')
    usage = response.to_dict()['usage']
    return {
        'content': cleaned_content,
        'prompt_tokens': usage['prompt_tokens'],
        'result_token': usage['completion_tokens']
    }

def generate_sign(method, path, params, app_secret):
    sign_str = method + path + "?"
    for key in sorted(params.keys()):
        if key == 'sign':
            continue
        sign_str += key + "=" + str(params[key]) + "&"
    sign_str = sign_str[:-1]
    sign_str += app_secret
    sign = hashlib.md5(sign_str.encode())
    return sign.hexdigest()

def fabric_llm_chat(prompt: str, model_name: str = "qwen3_235b", no_think: bool = True):
    """使用 test2.query_openai 调用大模型，返回与旧接口兼容的字典结构。"""
    result = query_openai(query=prompt, model_name=model_name, no_think=no_think)
    # 兼容旧调用：返回字典中含有 content
    return result

# =====================
# LLM Prompt生成
# =====================
    
def generate_sales_call_notes(row):
    prompt = f"""
你是一个试驾录音分析专家。你的任务是从试驾录音文本中：
1. **首要任务**：判断试驾真实性，识别虚假试驾（这是最重要的任务，必须严格识别）
2. 识别出车企员工
3. 分析员工是否存在安全问题或智驾问题

【核心原则】：
- 对于虚假试驾识别：必须严格识别，宁可多判，不可漏判。只要符合虚假试驾特征，必须判为"是"。
- 对于安全问题/智驾问题：宁可漏判，不可错判。只有在你认定存在明确问题时才输出。严禁进行任何推测、解读言外之意或分析整体语境。

# 输入数据
试驾录音ASR文本：
{row['text_content']}

# 分析任务

1. 识别员工
通过发言内容直接识别：谁在介绍产品、解答专业问题、回答用户疑问、引导试驾流程。此人即为员工。可能有多个。确定员工是第几个说话人（如"说话人1"、"说话人2"等）。如果无法明确识别，输出"无法判断"。

2. 判断试驾真实性（【最高优先级任务】必须严格识别虚假试驾）
**这是最重要的任务，必须优先完成。虚假试驾识别不准确会导致整个分析失效。**

虚假试驾是指没有真实客户参与、或试驾过程不符合正常试驾流程的情况。以下情况必须判为虚假试驾：

【核心判断标准】：
- 如果录音中只有FL（员工）一个人在说话，没有客户的声音或回应，判为虚假试驾。
- 如果录音中FL全程自言自语、自问自答，没有与客户的真实互动，判为虚假试驾。
- 如果录音内容与试驾完全无关（如纯销售咨询、售后问题、闲聊等），判为虚假试驾。

【具体虚假试驾类型识别】：

2.1 跑空车（最常见）
识别特征：
- 录音中只有FL一个人在说话，全程没有客户的声音
- FL独自介绍车辆功能，但没有客户提问、回应或互动
- FL自言自语，如"这个功能是这样的"、"现在我们在XX路段"，但没有客户参与
- 录音中FL像是在做演示或讲解，但没有任何客户的声音
判断标准：如果整个录音过程中，除了FL的声音外，没有任何其他说话人的声音，或虽然有多个说话人标记但实际只有FL在说话，判为"是"。

2.2 FL自言自语，无交流
识别特征：
- FL在说话，但没有任何客户回应、提问或互动
- FL自问自答，如"这个怎么样？挺好的"、"客户可能会问什么？我来介绍一下"
- 录音中FL像是在排练或练习，而不是真实的试驾对话
判断标准：如果FL的发言没有引发任何客户回应，或客户完全没有参与对话，判为"是"。

2.3 试驾全程再听会，无任何交流
识别特征：
- 录音内容显示是在开会、培训或内部讨论
- 录音中有多人讨论，但讨论的是业务、流程、培训等内容，而非试驾过程
- 录音中提到了"再听会"、"复盘"、"培训"等关键词
- 没有任何真实的试驾对话内容
判断标准：如果录音内容明确显示是会议、培训或内部讨论，而非真实试驾，判为"是"。

2.4 一个试驾单挂两个试驾，但实际只有一人
识别特征：
- 录音中只有一个人的声音（通常是FL）
- 虽然试驾单可能显示有多个参与者，但录音中只有FL在说话
- 没有客户的声音或互动
- FL可能提到"客户"但实际没有客户在场
判断标准：如果录音中实际只有FL一个人在说话，没有任何其他说话人的声音，即使试驾单显示有多个参与者，也应判为"是"。

2.5 其他虚假试驾情况
- 录音内容与试驾完全无关（如纯销售咨询、售后问题、闲聊等）
- 录音内容明显是测试、演示或练习，而非真实试驾
- 录音中FL在描述试驾过程，但没有任何客户参与的证据

【判断流程和技巧】：
1. **检查说话人数量**：
   - 如果ASR文本中只有"说话人1"（或只有一个说话人标记），且这个说话人是FL，判为"是"（跑空车）
   - 如果ASR文本中有多个说话人标记，但实际内容中只有FL在说话，其他说话人没有任何发言，判为"是"
   - 如果ASR文本中有多个说话人，且都有发言和互动，继续下一步判断

2. **检查对话模式**：
   - 正常试驾：FL介绍 → 客户提问/回应 → FL回答 → 客户继续提问/回应（有来有往的对话）
   - 虚假试驾：只有FL在说话，没有客户提问、回应或互动
   - 如果FL连续说很多话，但没有任何客户的声音或回应，判为"是"

3. **检查内容特征**：
   - 如果FL在介绍功能时像是在自言自语（如"这个功能是这样的，客户可能会问..."），判为"是"
   - 如果录音中提到"再听会"、"复盘"、"培训"等关键词，判为"是"
   - 如果录音内容与试驾完全无关，判为"是"

4. **检查互动证据**：
   - 客户提问（如"这个多少钱？"、"续航怎么样？"等）
   - 客户回应（如"好的"、"嗯"、"明白了"等，即使是简短回应也算）
   - 客户表达意见（如"我觉得不错"、"这个功能我喜欢"等）
   - 如果整个录音中完全没有以上任何客户互动的证据，判为"是"

5. **综合判断**：
   - 如果满足以上任何虚假试驾特征，必须判为"是"
   - 如果录音中有客户的声音和互动（即使较少），判为"否"
   - 如果无法确定，判为"无法判断"

【特别注意】：
- **严格标准**：对于虚假试驾识别，必须严格。只要录音中只有FL一个人在说话，或FL自言自语没有客户互动，就必须判为"是"
- 不要因为录音质量差、有噪音就判为虚假试驾（但如果确实只有FL的声音，仍应判为"是"）
- 如果录音中有客户的声音和互动，即使较少（如只有几次"嗯"、"好的"），也不应判为虚假试驾
- 如果FL在介绍车辆，客户有回应（即使只是"嗯"、"好的"等简短回应），不应判为虚假试驾
- **关键区分**：有客户互动（即使少）= 真实试驾；完全没有客户互动 = 虚假试驾

3. 安全问题分析（必须明确无疑）
只在以下情况直接、明确发生时，才认定为安全问题：
- 员工自身的危险驾驶行为（员工在驾驶时）：
    * 明确提及不系安全带。
    * 明确提及严重超速（例如："现在限速80，我开到120了"）。
    * 明确提及让穿着洞洞鞋、拖鞋、高跟鞋的用户进行驾驶操作。
    * 明确提及在公共道路上为了演示而将车辆急刹到远低于限速或刹停。
- 员工主动引导用户危险驾驶：
    * 明确告诉用户"可以不系安全带"等。
- 用户危险行为，员工未及时提醒：
    * 用户明确说自己做了危险行为（如："我没系安全带"、"我穿拖鞋开的"），且员工在后续对话中完全没有提及任何相关安全提醒。
注意：以下情况一律不算安全问题： 
- 联系上下文，如果员工有提醒（即使提醒不及时），或危险行为发生在试驾尾声（如已返回、靠边停车后），则不算安全问题。
- 联系上下文，目的是体验车子性能的加速/刹车、单手驾驶等演示或测试，则不算安全问题。
- 联系上下文，及时提醒/制止用户，则不算安全问题。
- 非驾驶者穿拖鞋，比如用户乘坐的时候，则不算安全问题。
- 在安全地点停车换手，则不算安全问题。
- 行驶中使用零重力座椅时员工明确表示"不推荐/不建议"，则不算安全问题。

4. 智驾问题分析（必须明确无疑）
【首要判定原则】：在分析任何智驾相关表述时，请先检查是否符合以下"绝对安全区"表述，这些表述在任何情况下都不应被识别为智驾问题：
1. 关于"保守"的表述：员工任何提及蔚来智驾"保守"、"稳健"、"安全第一"或与"激进"对比的表述，一律视为对安全性的强调。
2. 关于"城区智驾"现状的客观描述：员工任何提及城区智驾"还在发展"、"还在迭代"、"需要更新"、"有待提高"、"不成熟"、"不太好用"、"高速肯定可以"、"高速没问题"等表述，一律视为对当前技术发展阶段的客观说明。目前智驾高速没问题，城区等非高速路段还在迭代，不算问题。
3. 关于"够用"的表述：员工使用"够用"、"满足基本需求"、"日常可用"等词语描述智驾能力，一律视为中性或积极评价。
4. 关于未来迭代的表述：员工提及"下个版本会更好"、"期待更新"、"还会优化"等，一律视为积极展望。
只在以下情况直接、明确发生时，才认定为智驾问题，边缘性贬低行为一律不算：
- 贬低自身智驾能力：
    * 员工直接说蔚来智驾很差，或不如其他品牌，必须使用明确负面词汇：比如"不如华为/小鹏/理想"、"差远了"、"完全不行"、"很差"、"根本不能用"、"做得很烂"、"垃圾"、"这个功能完全没用"、"落后"、"体验差"等。
    * 【特别注意】：员工使用"保守"、"稳健"、"够用"、"城区还在迭代"、"有待更新"等词语，绝对不算贬低。必须严格区分客观描述与负面评价。
    * 【特别注意】：安全提醒（如："手要扶方向盘"、"注意接管"、"不能看车机视频/电影"、"早晚高峰期不建议使用"、"现阶段还是以人为主"、"城市道路尽量不要使用"），不算智驾问题，智驾还是需要人来把控才安全。
    * 【注意】：内部车型对比绝对禁止分析：只要提到"蔚来"、"NIO"、"乐道"、"ONVO"、"萤火虫"、"firefly"、"小虫子"及其具体车型（L60、L90、L80、ET5T、ET5、ES6、EC6、ES8、ES7、ET7、EC7、ET9、ES9）之间的任何对比，无论说什么，一律不算贬低。立刻停止分析。
    * 【注意】：只有智能辅助驾驶相关的功能才算智驾，比如巡航、领航、自动泊车、自动驾驶、智能辅助驾驶、ADS、世界模型、城区、L2、L3、端到端等；其他功能不算，比如HUD、单踏板模式、动力回收、充电、车载机器人、nomi、高架、导航等。
- 未正确澄清负面评价：
    * 用户明确说负面评价（如："蔚来智驾不行"、"不如XX"），员工用明确认同词回答，如："对"、"是的"、"确实"、"你说得对"。并且，在之后的整个对话中，员工再也没有说过任何话进行解释、澄清或转移话题。（需要联系上下文）
    * 【特别注意】：如果员工说了"嗯"、"哦"，或先说了"对/是的"但紧接着进行了解释（如"对，不过…"、"确实，但我们…"），则不算问题。只要后续进行了解释，用其他优势或话题抚平用户对智驾的疑虑，就不算问题，不需要深入分析整体语境。（需要联系上下文）
【问题判定前检查清单】：
在输出任何智驾问题前，请先逐项检查以下内容：
1. 员工是否使用了"保守"、"稳健"、"安全"等词语？→ 如果是，绝对不算问题。
2. 员工是否在描述城区智驾现状，如"高速可以"、"城区不太好用"、"还在迭代"？→ 如果是，绝对不算问题。
3. 员工是否使用了"够用"、"满足需求"等词语？→ 如果是，绝对不算问题。
4. 员工是否提及未来会优化、下个版本会更好？→ 如果是，绝对不算问题。
只有以上所有检查都为"否"时，才继续分析是否为真正的问题。

5. 生成试驾总结
生成客观、专业的试驾过程总结（≤200字）：
- 客观描述试驾过程的关键环节（如：试驾路线、体验的功能、客户反馈等）
- 总结员工的服务表现（如：是否主动介绍、是否及时提醒等）
- 提炼试驾中的亮点或问题点
- 禁止包含任何车型名称、品牌名称、价格信息
- 使用第三人称，专业、客观、简洁的语言风格

6. 提炼客户购车关注点
从客户提问、反馈、评价中提取关注点：
- 关注点类型：价格、性能、配置、智能化、安全性、舒适性、品牌、竞品对比等
- 必须是客户明确表达或强烈暗示的关注点
- 每个关注点用简洁词汇描述（如："价格"、"续航里程"、"智能驾驶"、"舒适性"等）

# 输出格式要求
仅输出标准JSON格式，不含```json```标记。所有字段必须严格按照要求输出。字符串字段如果为空，输出空字符串""，不要输出null。数组字段如果为空，输出空数组[]，不要输出null。

如果ASR为空或无效，只输出：
{{
  "是否虚假试驾": "无法判断",
  "试驾总结": "试驾录音ASR为空"
}}

如果ASR有效，输出完整格式：
{{
  "是否虚假试驾": "是/否/无法判断。必须严格按照第2部分'判断试驾真实性'的标准进行判断。如果录音中只有FL一个人在说话、FL自言自语无客户互动、录音内容是会议/培训、或明显是跑空车等情况，必须判为'是'。只有在明确有客户参与且存在真实互动的情况下，才判为'否'。如果不确定，判为'无法判断'",
  "fl角色": "说话人1/说话人2/无法判断",
  "试驾总结": "客观、专业的试驾过程总结，≤200字，不得包含车型名称、品牌名称、价格信息。如果判为虚假试驾，请在总结中简要说明虚假试驾的类型（如：跑空车、FL自言自语无交流、试驾全程再听会等）",
  "试驾是否有质检问题": "是/否/无法判断",
  "问题分类": ["安全问题"] / ["智驾问题"] / ["安全问题", "智驾问题"] / [],
  "具体问题": "问题的具体描述，≤150字。如果存在问题，简洁明确地描述问题的具体表现（如：'员工在驾驶时未系安全带'、'员工明确表示智驾不如竞品'）。如果不存在问题，输出空字符串''",
  "AI判断问题": "AI判断的依据和详细说明，≤200字。如果存在问题：1.明确说明问题类型（安全问题/智驾问题）；2.详细描述问题的具体表现和严重程度；3.提供判断依据（引用录音中的关键表述，用引号标注）；4.说明为什么判定为问题（引用判定标准）。如果不存在问题，输出空字符串''",
  "开始时间戳": "毫秒时间戳或空字符串。如果发现问题，输出问题录音片段的开始时间戳（毫秒）。如果未发现问题或ASR文本中没有时间戳信息，输出空字符串''",
  "结束时间戳": "毫秒时间戳或空字符串。如果发现问题，输出问题录音片段的结束时间戳（毫秒）。如果未发现问题或ASR文本中没有时间戳信息，输出空字符串''",
  "客户购车关注点": ["关注点1", "关注点2", ...]
}}
"""
    return prompt
    

# =====================
# 并发API调用与批量处理
# =====================
def batch_api_call(prompts: List[str], max_workers: int = 5) -> List[str]:
    results = [""] * len(prompts)
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fabric_llm_chat, prompt): i for i, prompt in enumerate(prompts)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                res = future.result()  # 期望为 { 'content': str, ... }
                content = res.get('content', '') if isinstance(res, dict) else str(res)
                results[idx] = content
                if content and str(content).strip():
                    success_count += 1
                    print(f"已成功获取答案数量: {success_count}")
            except Exception as e:
                print(f"处理失败(index={idx}): {str(e)}")
                results[idx] = ""
    return results

def batch_process_users(pandas_df):
    retry_workers_list = [5, 4, 3, 2, 1]
    prompts = [generate_sales_call_notes(row) for _, row in pandas_df.iterrows()]
    total = len(prompts)
    results = [""] * total
    remain_indices = list(range(total))
    round_num = 0
    while remain_indices and round_num < len(retry_workers_list):
        max_workers = retry_workers_list[round_num]
        print(f"\n第{round_num+1}轮，待处理数量: {len(remain_indices)}，max_workers={max_workers}")
        sub_prompts = [prompts[i] for i in remain_indices]
        sub_results = batch_api_call(sub_prompts, max_workers=max_workers)
        new_remain_indices = []
        for idx, res in zip(remain_indices, sub_results):
            if res and str(res).strip():
                results[idx] = res
            else:
                new_remain_indices.append(idx)
        print(f"本轮成功: {len(remain_indices) - len(new_remain_indices)}，剩余: {len(new_remain_indices)}")
        remain_indices = new_remain_indices
        round_num += 1
    prompt_lengths = [len(p) for p in prompts]
    avg_prompt_length = sum(prompt_lengths) / total if total else 0
    output_lengths = [len(r) for r in results]
    avg_output_length = sum(output_lengths) / total if total else 0
    success_count = sum([1 for r in results if r and str(r).strip()])
    success_rate = success_count / total if total else 0
    print(f"\n所有prompt平均字数: {avg_prompt_length:.2f}")
    print(f"大模型输出平均字数: {avg_output_length:.2f}")
    print(f"大模型调用成功率: {success_rate:.2%} ({success_count}/{total})")
    
    # 解析 LLM 返回的 JSON 结果，提取新增字段
    def parse_llm_result(text):
        """从 LLM 返回的文本中解析 JSON，提取所需字段"""
        if not text or not str(text).strip():
            return {
                'fl_role': '',
                'test_drive_summary': '',
                'has_quality_issue': '',
                'problem_classification': '',
                'specific_problem': '',
                'ai_judgment_problem': '',
                'start_timestamp': '',
                'end_timestamp': '',
                'call_notes': text or ''
            }
        
        try:
            # 尝试提取 JSON（可能被 ```json``` 包裹）
            text_clean = str(text).strip()
            # 移除可能的 ```json 和 ``` 标记
            if '```json' in text_clean:
                text_clean = text_clean.split('```json')[1].split('```')[0].strip()
            elif '```' in text_clean:
                text_clean = text_clean.split('```')[1].split('```')[0].strip()
            
            # 解析 JSON
            data = json.loads(text_clean)
            
            # 提取字段，如果不存在则使用默认值
            fl_role = data.get('fl角色', '')
            test_drive_summary = data.get('试驾总结', '')
            has_quality_issue = data.get('试驾是否有质检问题', '')
            problem_classification = data.get('问题分类', [])
            specific_problem = data.get('具体问题', '')
            ai_judgment_problem = data.get('AI判断问题', '')
            start_timestamp = data.get('开始时间戳', '')
            end_timestamp = data.get('结束时间戳', '')
            
            # 将问题分类列表转为字符串（用逗号分隔）
            if isinstance(problem_classification, list):
                problem_classification_str = ','.join(problem_classification) if problem_classification else ''
            else:
                problem_classification_str = str(problem_classification) if problem_classification else ''
            
            # 如果 specific_problem 为空，但 ai_judgment_problem 不为空，则使用 ai_judgment_problem 作为备用
            if not specific_problem and ai_judgment_problem:
                specific_problem = ai_judgment_problem
            
            return {
                'fl_role': str(fl_role) if fl_role else '',
                'test_drive_summary': str(test_drive_summary) if test_drive_summary else '',
                'has_quality_issue': str(has_quality_issue) if has_quality_issue else '',
                'problem_classification': problem_classification_str,
                'specific_problem': str(specific_problem) if specific_problem else '',
                'ai_judgment_problem': str(ai_judgment_problem) if ai_judgment_problem else '',
                'start_timestamp': str(start_timestamp) if start_timestamp else '',
                'end_timestamp': str(end_timestamp) if end_timestamp else '',
                'call_notes': text  # 保留原始返回内容
            }
        except Exception as e:
            print(f"解析 LLM 返回结果失败: {e}, 原始内容: {text[:200]}")
            return {
                'fl_role': '',
                'test_drive_summary': '',
                'has_quality_issue': '',
                'problem_classification': '',
                'specific_problem': '',
                'ai_judgment_problem': '',
                'start_timestamp': '',
                'end_timestamp': '',
                'call_notes': text or ''
            }
    
    # 解析所有结果
    parsed_results = [parse_llm_result(res) for res in results]
    
    # 构造结果 DataFrame，确保保留 fl_ad_account 字段，便于后续写入 Hive
    n = len(pandas_df)
    fl_series = (
        pandas_df['fl_ad_account']
        if 'fl_ad_account' in pandas_df.columns
        else pd.Series([''] * n)
    )

    result_df = pd.DataFrame({
        'brand': pandas_df['brand'],
        'order_no': pandas_df['order_no'],
        'fl_ad_account': fl_series,
        'text_url': pandas_df['text_url'],
        'text_url_authorized': pandas_df['text_url_authorized'],
        'text_content': pandas_df['text_content'],
        'audio_url': pandas_df['audio_url'],
        'audio_url_authorized': pandas_df['audio_url_authorized'],
        'duration': pandas_df['duration'],
        'call_notes': [r['call_notes'] for r in parsed_results],
        'fl_role': [r['fl_role'] for r in parsed_results],
        'test_drive_summary': [r['test_drive_summary'] for r in parsed_results],
        'has_quality_issue': [r['has_quality_issue'] for r in parsed_results],
        'problem_classification': [r['problem_classification'] for r in parsed_results],
        'specific_problem': [r['specific_problem'] for r in parsed_results],
        'ai_judgment_problem': [r['ai_judgment_problem'] for r in parsed_results],
        'start_timestamp': [r['start_timestamp'] for r in parsed_results],
        'end_timestamp': [r['end_timestamp'] for r in parsed_results],
    })
    return result_df


# =====================
# 提取文本数据
# =====================
SUSPECTS = set("åæçøØÆÅôöòóäàáêéèíïìúùýßÂÃÄÅÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜàâãäåçèéêëìíîïðñòóôõöùúûüÿ")

def looks_like_mojibake(s: str) -> bool:
    if not s:
        return False
    hits = sum(ch in SUSPECTS for ch in s[:2000])  # 只看前2k字符
    return hits >= 10  # 命中较多时视为 mojibake

def read_txt_robust(url: str, timeout: float = 25.0) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    raw = b""
    # 多策略尝试，尽量拿到正文
    attempts = [
        dict(headers=headers, timeout=timeout, allow_redirects=True, verify=True),
        dict(headers=headers, timeout=timeout, allow_redirects=True, verify=False),
        # 带 referer 再试
        dict(headers={**headers, "Referer": url}, timeout=timeout, allow_redirects=True, verify=True),
        dict(headers={**headers, "Referer": url}, timeout=timeout, allow_redirects=True, verify=False),
    ]
    for kw in attempts:
        try:
            r = requests.get(url, **kw)
            # 不 raise；有些 4xx 也会有正文
            raw = r.content or b""
            if raw:
                break
        except Exception:
            continue

    # 1) 先试utf-8
    try:
        s = raw.decode('utf-8')
        # 如果像“把UTF-8当latin-1解了”的样子，纠偏一次
        if looks_like_mojibake(s):
            try:
                return s.encode('latin1').decode('utf-8')
            except Exception:
                pass
        return s
    except Exception:
        pass

    # 2) 尝试 utf-8-sig, utf-16 系列
    for enc in ('utf-8-sig', 'utf-16-le', 'utf-16-be'):
        try:
            return raw.decode(enc)
        except Exception:
            continue

    # 3) 常见中文编码与其它
    for enc in ('gb18030', 'gbk', 'big5'):
        try:
            return raw.decode(enc)
        except Exception:
            continue

    # 4) 最后兜底：忽略错误的utf-8
    return raw.decode('utf-8', errors='ignore')


def format_text_content(s: str) -> str:
    if not s:
        return s
    # 统一换行
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    # 去除不可见控制字符（保留\n、\t）
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s)
    # 去除每行首尾空白
    lines = [line.strip() for line in s.split('\n')]
    s = '\n'.join(lines)
    # 折叠多余空行为最多1行
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ========= 授权与抓取工具 =========

def authorize_urls(url_list, batch_size=50):
    """
    分批鉴权URL，避免单次请求超过50个URL的限制
    
    Args:
        url_list: URL列表
        batch_size: 每批处理的URL数量，默认50
    
    Returns:
        dict: URL到鉴权后URL的映射
    """
    if not url_list:
        return {}
    
    result_map = {}
    total_urls = len(url_list)
    print(f"开始分批鉴权，总共 {total_urls} 个URL，每批 {batch_size} 个")
    
    # 分批处理
    for i in range(0, total_urls, batch_size):
        batch_urls = url_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_urls + batch_size - 1) // batch_size
        
        print(f"正在处理第 {batch_num}/{total_batches} 批，包含 {len(batch_urls)} 个URL")
        
        try:
            batch_result = fms_client.authorize_file(
                file_url_list=list(batch_urls),
                env='prod',
                zone='cn',
                app_id='100900',
                app_secret='bF9b1c26b129eD63eb445a453B347aAc',
                client_id='nio-ai_cn_80c33b',
                client_secret='prod_ecb8fd2cd2b3bd7d80a6487a85a961ba',
                expire_sec = 1209600  # 14 * 24 * 60 * 60
            )
            result_map.update(dict(batch_result))
            print(f"第 {batch_num} 批鉴权成功，获得 {len(batch_result)} 个授权URL")
        except Exception as e:
            print(f"第 {batch_num} 批鉴权失败: {e}")
            # 继续处理下一批，不中断整个流程
    
    print(f"鉴权完成，总共获得 {len(result_map)} 个授权URL")
    return result_map

def fetch_texts_concurrently(urls, max_workers=5):
    texts = {}
    if not urls:
        return texts
    def _task(u):
        try:
            # 返回原始文本内容，不做任何格式修改
            return u, read_txt_robust(u)
        except Exception:
            return u, "空"
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_task, u) for u in urls]
        for fut in as_completed(futures):
            u, content = fut.result()
            texts[u] = content
    return texts

# ========= DataFrame 列处理 =========

# 配置列名（如与实际不符请修改）
AUDIO_COL = 'audio_url'
TEXT_COL = 'text_url'

# 输出列
AUDIO_AUTH_COL = 'audio_url_authorized'
TEXT_CONTENT_COL = 'text_content'
TEXT_AUTH_COL = 'text_url_authorized'

# URL 合法性校验
def is_valid_http_url(value) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False
    if s.lower() in ('none', 'null', 'nan'):
        return False
    return s.startswith('http://') or s.startswith('https://')

# 音频列：授权 -> 新链接
if AUDIO_COL in pandas_df.columns:
    audio_urls = (
        pandas_df[AUDIO_COL]
        .astype(str)
        .str.strip()
        .replace('', pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    # 过滤非法 URL（例如 'None' 等）
    audio_urls = [u for u in audio_urls if is_valid_http_url(u)]
    audio_map = authorize_urls(audio_urls)
    pandas_df[AUDIO_AUTH_COL] = pandas_df[AUDIO_COL].map(lambda x: audio_map.get(str(x).strip(), '')).fillna('')
else:
    pandas_df[AUDIO_AUTH_COL] = ''

# 文本列：域名转换 -> 授权 -> 抓取文本 -> 新内容
if TEXT_COL in pandas_df.columns:
    # 对鉴权后的URL进行域名转换
    def convert_domain(url: str) -> str:
        """将 cdn-up-private.onvo.cn 转换为 oss-up-private-onvo.nioint.com"""
        if not url:
            return url
        return url.replace('cdn-up-private.onvo.cn', 'oss-up-private-onvo.nioint.com')
    
    text_src_urls = (
        pandas_df[TEXT_COL]
        .astype(str)
        .str.strip()
        .replace('', pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    # 过滤非法 URL
    text_src_urls = [u for u in text_src_urls if is_valid_http_url(u)]
    
    # 先对原始URL进行域名转换
    converted_src_urls = [convert_domain(url) for url in text_src_urls]
    text_auth_map = authorize_urls(converted_src_urls)
    
    # 保存鉴权后的文本链接列
    pandas_df[TEXT_AUTH_COL] = pandas_df[TEXT_COL].map(
        lambda x: str(text_auth_map.get(convert_domain(str(x).strip()), '')).strip()
    ).fillna('')
    
    # 获取鉴权后的URL列表用于抓取文本
    authorized_urls = [str(v).strip() for v in text_auth_map.values() if str(v).strip()]
    url_to_content = fetch_texts_concurrently(authorized_urls, max_workers=5)

    def map_text_content(original_url: str) -> str:
        if not isinstance(original_url, str) or not original_url:
            return ''
        normalized_url = original_url.strip()
        if not normalized_url:
            return ''
        # 先转换域名，再获取鉴权后的URL
        converted_url = convert_domain(normalized_url)
        auth_url = str(text_auth_map.get(converted_url, '')).strip()
        if not auth_url:
            return ''
        return url_to_content.get(auth_url, '')

    pandas_df[TEXT_CONTENT_COL] = pandas_df[TEXT_COL].map(map_text_content)
else:
    pandas_df[TEXT_CONTENT_COL] = ''
    pandas_df[TEXT_AUTH_COL] = ''

print('*'*20, pandas_df['text_content'].head(5))
print(pandas_df.columns)


# =====================
# 生成质检结果
# =====================
print("开始生成销售电话建议...")
def extract_think_and_notes(text):
    think_match = re.search(r'<think>\n?(.*?)</think>\n?', text, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""
    call_notes = re.sub(r'<think>\n?.*?</think>\n?', '', text, flags=re.DOTALL).strip()
    return pd.Series({'think': think, 'call_notes': call_notes})


result_df = batch_process_users(pandas_df)
# result_df[['think', 'call_notes']] = result_df['call_notes'].apply(extract_think_and_notes)

print('*'*10,)
print(result_df.head(5))


# =====================
# 时间戳格式化函数
# =====================
def format_timestamp_to_hms(timestamp_value):
    """
    将任意时间戳字段统一转换为 HH:MM:SS 字符串。
    
    支持并自动纠正常见情况：
    - 纯数字（毫秒或秒）：如 456000, "2816000", "62" 等
    - 已有人类可读格式：HH:MM、HH:MM:SS、带多余空格/中文/括号等
    - 带异常字符：如 "1319000:", "开始1319000ms-1324000ms"
    
    无法解析或为空时，返回空字符串。
    """
    if not timestamp_value or pd.isna(timestamp_value):
        return ''
    
    timestamp_str = str(timestamp_value).strip()
    if not timestamp_str or timestamp_str.lower() in ('none', 'null', 'nan'):
        return ''
    # 1) 先清洗掉除数字和冒号外的所有字符（防止 "5627812820ms"、"1319000:" 等）
    cleaned = re.sub(r'[^0-9:]', '', timestamp_str)
    if not cleaned:
        return ''

    # 2) 如果是 HH:MM 或 HH:MM:SS 这种「看起来已经是时间」的格式，直接标准化
    if ':' in cleaned:
        parts = cleaned.split(':')
        # 只保留前 3 段，多余的丢弃
        parts = parts[:3]
        # 不足 3 段的补到 3 段
        while len(parts) < 3:
            parts.append('00')
        try:
            h, m, s = [int(p or '0') for p in parts[:3]]
            # 归一化分钟/秒，防止出现 00:75:80 这类情况
            total_seconds = h * 3600 + m * 60 + s
        except ValueError:
            return ''
    else:
        # 3) 纯数字：既可能是毫秒也可能是秒
        try:
            value = int(cleaned)
        except ValueError:
            return ''

        # 经验规则：长度 >= 4 基本可以认为是毫秒，否则认为是秒
        if len(cleaned) >= 4:
            total_seconds = value // 1000
        else:
            total_seconds = value

    # 4) 统一转换为 HH:MM:SS（不做 24 小时取模，保持绝对时长）
    if total_seconds < 0:
        return ''

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# =====================
# 上传结果到Hive
# =====================
print('开始上传结果')
# 确保所有字段都存在，处理可能的 NULL 值
# 注意：所有字段必须从 result_df 中获取，不能新增字段
result_df_upload = result_df.copy()
# 确保所有必需字段存在，如果不存在则填充空字符串
required_fields = {
    'fl_role': '',
    'test_drive_summary': '',
    'has_quality_issue': '',
    'problem_classification': '',
    'specific_problem': '',
    'ai_judgment_problem': '',
    'start_timestamp': '',
    'end_timestamp': '',
    'text_content': ''
}
for field, default_value in required_fields.items():
    if field not in result_df_upload.columns:
        result_df_upload[field] = default_value
    else:
        # 将 NULL/NaN 填充为空字符串
        result_df_upload[field] = result_df_upload[field].fillna('')

# 格式化时间戳字段为 HH:MM:SS 格式
if 'start_timestamp' in result_df_upload.columns:
    result_df_upload['start_timestamp'] = result_df_upload['start_timestamp'].apply(format_timestamp_to_hms)
if 'end_timestamp' in result_df_upload.columns:
    result_df_upload['end_timestamp'] = result_df_upload['end_timestamp'].apply(format_timestamp_to_hms)

spark_df = spark.createDataFrame(result_df_upload)
spark_df.createOrReplaceTempView("temp_view")

write_sql = (f"""
  INSERT OVERWRITE ads.drive_test_asr_llm_detect_fl_feedback_1d_i
  PARTITION (datetime='{dt}', model_version='{model_version}')
  SELECT 
	brand as td_brand_code, -- td_brand_code
	order_no as td_order_no, -- td_order_no
	fl_ad_account, -- fl_ad_account
	
	fl_role, -- fl角色（第几个说话人是销售）
	test_drive_summary, -- 试驾总结
	has_quality_issue, -- 试驾是否有质检问题 
	problem_classification, -- 问题分类（安全问题/智驾问题）
	specific_problem, -- 具体问题（问题的简洁描述）
	ai_judgment_problem, -- AI判断问题（判断依据和详细说明）
	text_content as speech_to_text_content, -- 语音转文本内容（ASR原始文本）
	start_timestamp, -- 开始时间戳（时分秒格式：HH:MM:SS）
	end_timestamp, -- 结束时间戳（时分秒格式：HH:MM:SS）
	'' as created_at, -- 创建时间（系统字段，保留为空）
	'' as updated_at, -- 更新时间（系统字段，保留为空）
    
	text_url, -- text_url
	text_url_authorized, -- text_url_authorized
	audio_url, -- audio_url
	audio_url_authorized, -- audio_url_authorized
	duration -- duration 

  FROM temp_view
""")
print('*'*20, write_sql)
spark.sql(write_sql)


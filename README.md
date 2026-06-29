# 遣唐使交流网络的知识图谱分析

基于日本六国史、新旧唐书、圆仁《入唐求法巡礼行记》等七种文献，构建遣唐使（630–894）知识图谱并进行社会网络分析。

## 安装

```bash
pixi install
```

## 运行

```bash
# 完整流水线
pixi run python -m embedding_pipeline.orchestrator

# 各阶段单独运行
pixi run python -m embedding_pipeline.step1_chunk
pixi run python -m embedding_pipeline.step3_encode
pixi run python -m embedding_pipeline.step4_index
pixi run python -m embedding_pipeline.step5_retrieve
pixi run python -m embedding_pipeline.step6_extract
pixi run python -m embedding_pipeline.step7_merge
pixi run python -m embedding_pipeline.step8_kg
pixi run python -m embedding_pipeline.enhance_events

# 分析脚本
pixi run python scripts/analysis_network.py
pixi run python scripts/analysis_temporal.py
pixi run python scripts/analysis_source_robustness.py
```

## 目录结构

```
├── data/                         # 原始文献
├── output/                       # 最终交付物
├── scripts/
│   ├── embedding_pipeline/       # ETL 流水线
│   │   ├── config.py             # 配置
│   │   ├── step1_chunk.py        # 文本分段
│   │   ├── step3_encode.py       # 向量编码
│   │   ├── step4_index.py        # 索引构建
│   │   ├── step5_retrieve.py     # 语义检索
│   │   ├── step6_extract.py      # 事件抽取（批处理准备）
│   │   ├── step7_merge.py        # 合并去重
│   │   ├── step8_kg.py           # 知识图谱构建
│   │   ├── enhance_events.py     # 时间标注
│   │   └── orchestrator.py       # 流水线编排
│   └── analysis_*.py             # 分析脚本
```
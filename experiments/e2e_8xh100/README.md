# 8×H100 端到端实验 (e2e_8xh100)

QwQ-32B 在 8×H100 集群上的端到端 replay 压测实验。客户端按真实线上日志(flowgpt)回放请求,
测量不同「引擎调度策略 × 集群路由策略」下的 TTFT / TPOT / E2E 延迟与 SLO 达成率。

> 来源:整理自 `/home/ubuntu/zhangy/llm-inference-benchmarking/e2e_8xh100_exp/`,
> 数据集来自 `/home/ubuntu/qq/llm-inference-benchmarking/replay-logs-origin.log`。

## 目录结构

```
experiments/e2e_8xh100/
├── launch_engines.sh        # ① 启动 8 个 QwQ-32B 引擎(每卡 1 个, SLA + ELRAR Agent)
├── launch_router.sh         # ② 启动 production-stack vllm_router(ELRAR 路由, 自动起 Gateway)
├── launch_gateway.sh        # (可选)独立启动 State Gateway;ELRAR 路由已自动起,一般不用
├── stop_all.sh              # 停止引擎 / router / gateway
├── online_replay.py         # ③ 压测主程序(replay 客户端)
├── run_all_clients.sh       # 客户端总入口:依次跑下面 10 个 split 脚本
├── run_client_split*.sh     # 10 个客户端分片脚本(见下方配置表)
├── data/
│   └── replay-logs-origin.log   # 原始请求日志数据集(~4GB, 235k 行, git 已忽略)
├── logs/                    # 运行时生成:engine_*.log/.pid, router.log, gateway.log
├── exp/                     # 历史实验结果(timestamp 模式:sla/sarathi × elrar/roundrobin)
└── exp_new/                 # 历史实验结果(qps + timestamp:native/sla × roundrobin/qps/elrar)
```

## 完整复现链路(8×H100 + ELRAR)

```
8×H100 单机:
  引擎0..7 (vllm serve QwQ-32B, GPU 0..7, :8000..8007)   ← SLA-aware 调度 + ELRAR Agent
        │  UDP:9999 推送 EngineState
        ▼
  vllm_router (production-stack exp 分支, ELRARRouter, :8888)  ← 内含 State Gateway
        ▲  HTTP
        │
  online_replay.py 客户端  (API_BASE=localhost:8888)
```

前置:
1. 当前环境的 `vllm` 是本仓库 **exp-v0.9.1** 编译安装版(含 `sla_aware` + `engine_agent`);
2. production-stack 已切到 **exp 分支**(含 `ELRARRouter` + `state_gateway`),
   路径默认 `/home/ubuntu/zhangy/production-stack`,可用 `PRODUCTION_STACK=` 覆盖。

三步启动:
```bash
cd experiments/e2e_8xh100
bash launch_engines.sh          # ① 起 8 引擎;等所有 :8000..8007 /health 返回 200
bash launch_router.sh           # ② 起 ELRAR router(:8888),自动拉起 Gateway(UDP:9999)
bash run_all_clients.sh         # ③ 跑客户端压测
# 结束后
bash stop_all.sh
```

换路由策略做对比(engine 侧不用动):
```bash
ROUTING_LOGIC=qps        bash launch_router.sh   # 或 roundrobin / least_loaded / latency_based ...
```

关键可调参数(均在脚本顶部,或用环境变量覆盖):
- 引擎:`MODEL` / `NUM_ENGINES` / `PORT_BASE` / `GPU_BASE` / `MAX_MODEL_LEN` / `GPU_MEM_UTIL` / `SLO_TPOT_MS` / `SLO_TTFT_MS` / `PRETRAINED_MODEL`
- 路由:`ROUTING_LOGIC` / `ROUTER_PORT` / ELRAR 打分权重 `VLLM_ELRAR_W1..W4` / `VLLM_ELRAR_SLO_MS` / `VLLM_ELRAR_STALE_MS`

`exp/` 和 `exp_new/` 每个子目录形如 `flowgpt_<mode>_<engine>_<router>_<lo>_<hi>/`,内含每个进程的
`*.json`(汇总指标)与 `*.csv`(逐请求明细),即 `paper_figs/e2e_exp` 画图脚本的输入来源。

## 前置条件(脚本不含 server 启动)

这些脚本只是**客户端**。运行前需自行拉起:
1. 8×H100 上的 vLLM 推理引擎,OpenAI 兼容 API 监听在 `http://localhost:8888/v1`,模型 `Qwen/QwQ-32B`;
2. 对应的 ELRAR 集群路由 / State Gateway(`ENGINE=sla` 且 `ROUTER=qps/elrar` 时需要)。

## 运行

```bash
cd experiments/e2e_8xh100
# 单个分片
bash run_client_split.sh
# 全部分片(依次跑,每个间隔 30s,单个超时 2h)
bash run_all_clients.sh
```

数据集路径已改为脚本自定位(`${SCRIPT_DIR}/data/replay-logs-origin.log`),无需再手改。
其余可调项在每个脚本顶部:`MODEL_NAME` / `API_BASE` / `API_KEY` / `E2E_SLO` / `TTFT_SLO` /
`TPOT_SLO` / `MAX_TOKENS` / `MAX_ROUNDS`。

### Replay 模式
- `REPLAY_MODE="timestamp"`:按日志真实时间戳回放,用 `[LOWER_BOUND, UPPER_BOUND)` 采样区间控制负载;
  脚本每 0.1 区间起一个进程、相隔 10s。
- `REPLAY_MODE="qps"`:按固定 `TARGET_QPS` 回放;脚本每 5 QPS 起一个进程、相隔 10s。

### 各分片当前配置(均 ENGINE=sla, ROUTER=qps)

| 脚本 | 模式 | 采样区间 / QPS |
|---|---|---|
| run_client_split_.sh  | timestamp | 0.0–0.12 |
| run_client_split.sh   | timestamp | 0.0–0.16 |
| run_client_split_1.sh | timestamp | 0.0–0.20 |
| run_client_split_2.sh | timestamp | 0.0–0.24 |
| run_client_split_3.sh | timestamp | 0.0–0.28 |
| run_client_split_4_.sh | qps | TARGET_QPS=8 |
| run_client_split_4.sh  | qps | TARGET_QPS=10 |
| run_client_split_5.sh  | qps | TARGET_QPS=12 |
| run_client_split_6.sh  | qps | TARGET_QPS=14 |
| run_client_split_7.sh  | qps | TARGET_QPS=16 |

输出写到 `exp_qwen32b/<dataset>_<mode>_<engine>_<router>_<lo>_<hi>/` 下(运行时自动创建)。

## 数据集

`data/replay-logs-origin.log`(~4GB,235,341 行)是 flowgpt 线上服务的原始请求日志,含时间戳、
requestId、prompt 等字段。**已被仓库 `.gitignore` 的 `*.log` 规则忽略,不会提交进 git。**
若换机器,需自行同步该文件到 `data/` 下。脱敏工具见
`/home/ubuntu/zhangy/llm-inference-benchmarking/replay_logs_desensitize/`。

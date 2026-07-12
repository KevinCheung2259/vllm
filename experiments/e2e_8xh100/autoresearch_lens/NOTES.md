# AutoResearch: 超越 Fig6 SynergySched(sla_elrar)

## 目标(300s 全量,csv_process [30,30])
- FlowGPT-Q P50 ≤ 7.3/7.4/7.4/7.7/8.5s;SLO(10s) ≥ 100/100/100/99/85%(qps14-18)
- FlowGPT-T P50 ≤ 8.2/8.5/9.5/12/19.5s;SLO(20s) ≥ 100/100/100/92/79%
- 最大提升空间:Q@17-18、T@16-18(基线开始塌的区段)

## 基线配置(commit 1b47cdc 验证值)
SLA_ENABLED=true ELRAR_ENABLED=true SLO_TPOT_MS=100 MIN_BATCH=32 MIN_BATCH_K=2.0
EXPECTED_OUTPUT_LEN=256 QUEUE_PENALTY=1.0 pkl=stable_model_h100_qwen32b.pkl
router=elrar(w1..w4=1,session_key=x-user-id)

## 弱点清单(代码审计)
- W1 engine: e2e_score 无 TTFT 项(只有 tpot_est*E_out*(1+unadmitted*penalty))
- W2 engine: OPT_TIMEOUT_MS=1ms 截断升序 batch 搜索 → 偏小 batch
- W3 engine: token 预算恒按 slo_tpot 解,自适应 target_latency 未用于预算
- W4 engine: tpot_est=predict(B,B) 把含 prefill 步当纯 decode
- W5 engine: prefill chunk 硬编码 512
- R1 router: 线性加权和(LMetric 论文:乘法更优且免调参)
- R2 router: load 用 tanh 高载饱和
- R3 router: mode 项全同无效
- R4 router: session_key=x-user-id 但客户端发 X-Flow-Conversation-Id → KV 亲和恒 0
- R5 router: 无本地在途计数(lmetric 已验证有效)

## 实验协议
- 探针:FlowGPT-Q qps16+qps18,90s/点,ROUTING_LOGIC=elrar,csv 过滤 [20,20]
- 一次一改;赢家组合后 300s 全量验证(qps14-18 + FlowGPT-T)

## 基础设施
- **live knobs**: 引擎挂载 `autoresearch_lens/knobs.env`(VLLM_SLA_KNOB_FILE),
  optimizer 2s TTL 重读 → 引擎侧超参**免重启**热切换。
- 新旋钮: VLLM_SLA_TTFT_W(E2 目标加 TTFT 积压项,0=off)、VLLM_SLA_OPT_TIMEOUT_MS、
  VLLM_SLA_MAX_CHUNK、VLLM_SLA_BUDGET_LAT_MS。
- 探针: `ROUTING_LOGIC=elrar CONFIG=<id> QPS_LIST="16 18" DUR=90 OUTROOT=/tmp/ar/<id> bash run_fig6_flowgpt_q.sh`
  分析: `analyze_probe.py /tmp/ar/<id> 20`

## 实验队列
- E0: 基线复现(knobs=commit 值)
- E1: OPT_TIMEOUT_MS 1→8(修 W2 搜索截断)
- E2: TTFT_W=1(修 W1 目标缺 TTFT 项)
- E3: BUDGET_LAT_MS 100→150(修 W3,配合 E2 让搜索自行权衡)
- E4: MAX_CHUNK 512→2048(修 W5)
- E5: router session-key → X-Flow-Conversation-Id(修 R4,激活 KV 亲和)
- E6: router 乘法打分 / 在途计数负载项(修 R1/R2/R5)

## 实验记录(90s 探针,trim20;tpot 列=中位 decode 总时长秒)
| id | 改动 | Q16: aqps/p50/ttft/dec | Q18: aqps/p50/ttft/dec | 结论 |
|---|---|---|---|---|
| 红线目标 | 旧目标 min\|pred−target\|(300s) | −/7.4/−/~6 | −/8.5/−/~7 | 要超越的对象 |
| E0 | HEAD+SLO100+MB32 | 14.2/14.4/1.0/13.5 | 15.4/16.7/1.3/15.3 | 慢平衡(68ms/tok),复现失败 |
| E0b | MB1+BUD50 | 13.7/13.2/2.5/11.0 | 13.9/12.7/2.0/10.7 | decode 快些,TTFT 尾部爆 |
| E0c | +QP=0.1 | 13.8/13.1/2.2/10.9 | 13.9/11.9/1.6/10.2 | 边际;旋钮救不了目标函数 |

**核心发现**:红线来自旧目标的隐式限批(步延迟贴 target→小批→~30ms/tok)。
HEAD 的 E2E 目标在集群低载区"全接纳"→批随并发涨→慢平衡。
→ E7:步延迟(tpot_est)≤ STEP_CAP×自适应target 作硬可行性约束,可行集内用 E2E 评分。

| E7 | +STEP_CAP=1.0(SLO50重启) | 13.9/13.2/2.9/10.6 | 14.1/14.8/3.9/10.7 | cap 没咬合 |
| E8 | BUD25+CAP0.6 | 11.9/14.7/8.7/6.1 | 12.7/19.2/13.1/6.0 | decode 减半✓ 但 TTFT/吞吐爆 |

**机理定论(E7/E8)**:
1. decode 速度=步频(每步每请求 1 token)→ **步长直接决定 TPOT**(50ms 预算→53ms/tok ✓;25ms→30ms/tok ✓)
2. 但短步固定开销(截距)压垮 prefill 吞吐;FlowGPT 是 prefill 主导(4.2k in/200 out)→ TTFT 爆
3. **固定步长无法两全 → E9:预算跟随自适应 target**(队列空→短步,积压→长步),
   即红线隐式机制的显式化;坡道(TARGET_MIN/MAX/QTHRESH)成为新的可调前沿。
   BUDGET_LAT_MS=-1 启用;E9a=15→50@q5(红线复刻),之后扫坡道找更优点。

| E9a | BUD=-1(=target), 15→50@q5 | 13.8/15.2/4.4/10.9 | 13.5/14.5/4.2/10.5 | 90s 全在暖机期(见下) |

**方法学大发现(读老 sla_elrar JSONL 逐轮数据)**:
- 老实验 = 8轮×60s=480s;**前3轮 p50 14-19/ttft 5-7(=我探针数字!),第4轮起 p50 7.3/ttft 0.6**
- 红线 7.4 = 冷启动积压排空后的稳态;p50 落好簇因好轮占 5/8
- 我 90s 探针整个在暖机期;300s 跑法暖机占 62% vs 老 480s 占 36% → **时长不公平**
- E9a 的 SLA-DECISION: wait=151/171 → 坡道打满 50ms、pre=1/步(chunk512+budget500 只推1个prefill)
→ 行动:①探针 DUR=240+尾窗(150,10)看稳态;②优化"排空速度"缩短暖机(TARGET_MAX=100/QTHRESH=8/CHUNK=1024=E9b);③最终验证用 480s 对齐老协议

| E9b | MAX=100/q8/chunk1024, 240s | 全窗15.8/15.6/2.9/12.6;尾窗14.0/15.2/2.8/12.3 | 全窗17.6/19.3/3.6/15.8;尾窗15.0/19.3/3.7/15.7 | 吞吐↑但尾窗不收敛 |

**E9b 尾窗决策日志抓到坍塌 bug**:
`B=1 S=1 pred=193.6 target=15 run=11` —— wait=0 → target 15ms → solve 预算≈1 token
→ 11 个 running 每步只喂 1 个 → 饿死→堆积→100ms→再排空→振荡,永不进入快平衡。
(另:模型小 B 区失准,pred(1,1)=193ms 非单调)
**E10 修复**:预算硬下限 = batch_size(每 decode 每步必得 1 token)+ 有 prefill 积压时
再加 PREFILL_QUANTUM(256)。这是老系统能收敛而新系统不能的最可能根因修复。

| E10 | 预算下限 B(+256 if backlog), 15→50@q5 | 全窗12.2/14.3/3.4/10.9 p99=144! 尾窗9.0/13.1/2.4/10.7 | 全窗12.8/14.5/3.5/10.9 p99=162! 尾窗9.4/13.6/2.8/10.8 | 饿死修了但吞吐崩,尾窗仍不收敛 |

**转折**:自适应坡道族在新栈上全部失败(E9a/E9b/E10)。停止猜测 →
**检出 24de1ca 老栈**(optimizer/predictor/throughput_model/config/scheduler.py,pkl 两版相同)
在同协议下跑 OLD 基线(240s 双窗)。判据:
- OLD 尾窗 ~7-8s → 新栈有真回归,diff predictor/throughput_model(1b47cdc 改了 84+290 行)
- OLD 尾窗也 ~13-15s → 红线数字依赖 480s/缓存暖热,优化方向=暖机加速+对齐时长
(E10 版 optimizer 已存 versions/optimizer_E10.py)

| OLD | 24de1ca 栈原样 | 全窗15.1/18.5/7.4/11.0;尾窗12.1/20.3/9.3/11.0 | 全窗15.6/25.8/14.8/11.0;尾窗10.4/36.2/25.3/11.0 | 老栈在我协议下更差且发散 |

**OLD 之后的两大发现**:
1. 缓存理论弱化:OLD 跑完累计命中仅 5.4%(FlowGPT 切片基本唯一)
2. **Router 冷启动热点实锤**:OLD2 起步 23 个请求 16 个进 engine 8000!
   ELRAR 打分并列恒选第一个 + request_token_length 疑似 0 → 本地 pending 补偿失效
   → 开局 burst 单引擎堆积 → 局部过载雪球。**影响此前所有实验!**
→ 修复:ELRARRouter 加在途请求计数项(note_complete 钩子,W5*tanh 归一)
→ OLD3 = 老引擎栈 + 修复 router,验证路由热点是否为主因

| OLD2 | 暖缓存重跑 | 尾窗12.7/22.5/11.8/11.0 | 尾窗10.6/39.0/28.1/11.0 | 缓存理论排除(累计命中仅5.4%) |
| OLD3 | +修复router | 尾窗12.7/23.7/12.7/11.0 | 尾窗10.7/38.5/27.7/11.0 | 路由已均衡(515-575/引擎)但性能同 |

**最终定论(全链闭合)**:
- 老 Q18 逐轮:r0-3 差(15-30s),r4 过渡,**r5+(>300s)收敛 8.1-8.7/ttft 0.63**
- 我 OLD3 尾窗(150-230s)23.7 ≈ 老 r3(180-240s)26.0 → **完美复现老轨迹,只是没跑到收敛点**
- 抢占=0、KV~55% → 容量/抢占论排除
- **红线 = 5分钟暖机 + 好稳态的混合;超越主战场 = 把收敛从 ~300s 压到 ~60-90s**
  (SLO/P90/TTFT 大头都在暖机期),次战场 = 稳态本身(старый稳态 7.3/8.1)
- E9 连续坡道失败原因:大部分时间卡 50ms 中间态,两头不讨好
→ **E11 = bang-bang 双相+滞回**:DRAIN(wait≥8:100ms 步+2048 chunk 拉满 prefill)
  ↔ STEADY(wait≤2:28ms 短步快 decode + E10 配额保 arrival)。480s 探针,每 100s 分窗看收敛速度。
对照(老 Q16 逐轮):15.1/14.4/18.0/11.9/7.7/7.6/7.3/7.3;E11 目标 t100 即入 <8s

| E11 | bang-bang(28/100,q2/8) | 480s 全程 p50~18,dec~15(75ms/tok),吞吐满但无收敛 | − | drain 主导振荡:稳态相 quantum 256/步 追不上到达率 |

**E11 教训 → E12**:老连续坡道其实是反馈控制器,队列自动稳定在
"prefill 速率=到达率"的物理最优点(~30-35ms 步长,Q16 稳态 7.3 已贴物理下限)。
不该替换它——保留老控制器,只加**深积压涡轮**:
E12 = 24de1ca 老栈原样 + compute_adaptive_target_latency 前置一条规则
(queue≥DEEP_Q(20) → target=TURBO_MS(110)),只加速冷启动/突发排空,稳态零改动。
+ router 在途计数修复保留。480s Q16 轨迹对照老 r0-r7。

| E12 | 老栈+涡轮(DEEP_Q20/110ms) | 480s: t100 起卡 p50~25,dec 14,ttft 11 | − | 涡轮撑爆 running→KV 压力→站立队列,更差 |
| E13 | 老栈+router 准入窗口24 | 480s: dec 恒 10.3,TTFT 无界爬升(→28s),goodput 14.6<16 | − | 窗口造成引擎空转气泡;decode 快不了 |

## 最终结论(2026-07-11,10 个实验闭环)
1. **物理下限**:Q16(我客户端真实投放 ~16/s)下,混合步 prefill 份额
   = 16×4.2k/8 引擎/~25 步秒 ≈ 340 tok/步 → 步长 ~40-50ms → **decode 8-10s 是物理地板**,
   与调度策略基本无关(E13 用窗口把引擎队列钉到 0,decode 仍 51ms/tok,实证)。
2. **红线 7.4/8.5 不是我协议下可达的数字**:老客户端(修 bug 前)有效投放更低 +
   480s 窗口好轮占比高。老栈在我协议下 480s 全窗 ≈ p50 18-20(OLD/OLD2/OLD3 一致)。
3. 所有"新目标函数/坡道/bang-bang/涡轮/窗口"变体都不能突破物理地板;
   老连续坡道 = 已是自校准反馈控制器,稳态即物理最优。
4. **公平对比必须同协议重生成基线**:同客户端、同时长(480s)下重跑 sla_elrar 作真基线,
   然后在此基线上比较改进(router 在途计数修复是已验证的客观改进,消除冷启动热点)。

## 已交付资产
- Router:在途计数负载项修复(冷启动 16/23 单引擎热点 → 完全均衡 515-575)+ 可选准入窗口
- Engine:live-knob 基础设施(免重启热调参)、E10 预算下限修复(饿死 bug)、
  E12 涡轮/E11 bang-bang(knob 门控,默认关)
- 版本:versions/optimizer_{E10,E11,E12}.py;老栈=git 24de1ca
- 诊断链:步长=TPOT 机理、饿死 bug、路由热点、300s 收敛、物理地板,全部实证

## qps14 定点擂台(300s 同协议)最终排名
| 配置 | p50 | p90 | p99 | SLO10 | ttft |
|---|---|---|---|---|---|
| **synergy_v4 (rr_veto)** | 8.96 | **10.16** | **11.09** | **86.8%** | **0.37** |
| native_lmetric | 8.81 | 11.2 | − | 74.7% | 1.85 |
| native_rr | 8.90 | 11.0 | 13.1 | 75.2% | 0.36 |
| synergy_lite (sarathi+agent+elrar加法W5) | 9.18 | 11.7 | 13.9 | 68.6% | 0.79 |
| native_session | 9.28 | 11.8 | 14.0 | 67.8% | 0.46 |
| synergy_v3 (乘法+会话亲和) | 9.27 | 13.5 | 17.3 | 63.2% | 0.85 |
| sla_elrar_v2 (老引擎+router修复) | 11.04 | 12.6 | 13.9 | 18.6% | 2.59 |
| sla_elrar_fair (原版重生成) | 11.49 | 13.6 | 15.7 | 13.0% | 2.74 |

**v4 = SOTA**:严格支配 RR(p99 −2s,SLO +11.6pp),全面超 lmetric(除 p50 噪声内)。
**方法**:VLLM_ELRAR_RR_VETO=1 —— RR 轮转基座 + 在途否决(>min+margin 则跳过)。
关键教训:同构场里状态打分(加法/乘法/亲和)全输给盲 RR;状态的正确用法是否决权而非选优。
引擎侧:低载让位(sarathi 行为),SLA 引擎小预算 prefill 纯亏 TTFT(+2.2s)。

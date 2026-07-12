"""三阶段SLA感知优化算法模块

该模块实现了论文中描述的三阶段SLA感知调度优化算法：
1. 全局配置优化：搜索最优batch size
2. 运行队列调度：优先处理decode请求，合理分配prefill请求
3. 等待队列选择：基于优先级选择新请求

算法设计确保与现有vLLM调度器完全兼容，同时提供SLA感知的调度决策。
"""

import os
import time
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass


class _LiveKnobs:
    """研究用 live 旋钮:优先从 VLLM_SLA_KNOB_FILE 指向的 KEY=VAL 文件读参数
    (带 2s TTL 缓存,按 mtime 失效),否则回退 os.environ,再回退默认值。
    使得引擎侧调度超参可以在【不重启引擎】的情况下热切换,加速实验迭代。
    文件缺失/解析失败一律 fail-open 回退 env/默认值。
    """

    def __init__(self):
        self._path = os.getenv("VLLM_SLA_KNOB_FILE", "")
        self._cache: Dict[str, str] = {}
        self._mtime = 0.0
        self._checked_at = 0.0

    def _refresh(self) -> None:
        now = time.monotonic()
        if now - self._checked_at < 2.0:
            return
        self._checked_at = now
        if not self._path:
            return
        try:
            mtime = os.path.getmtime(self._path)
            if mtime == self._mtime:
                return
            cache: Dict[str, str] = {}
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    cache[k.strip()] = v.strip()
            self._cache = cache
            self._mtime = mtime
            logger.info(f"[LiveKnobs] reloaded {len(cache)} knobs from {self._path}: {cache}")
        except OSError:
            pass

    def get(self, name: str, default: str) -> str:
        self._refresh()
        if name in self._cache:
            return self._cache[name]
        return os.getenv(name, default)

    def get_float(self, name: str, default: float) -> float:
        try:
            return float(self.get(name, str(default)))
        except ValueError:
            return default

    def get_int(self, name: str, default: int) -> int:
        try:
            return int(float(self.get(name, str(default))))
        except ValueError:
            return default


_KNOBS = _LiveKnobs()

# 导入vLLM相关类型
from vllm.v1.request import Request

from .performance_predictor import PerformancePredictor
from .config import SLASchedulerConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """优化结果数据类
    
    包含优化算法的所有输出结果，用于调度器进行资源分配决策。
    """
    optimal_batch_size: int                 # 最优batch size
    optimal_token_budget: int              # 最优token预算
    allocation: Dict[str, int]             # request_id -> tokens的分配映射
    predicted_latency: float               # 预测延迟
    optimization_time_ms: float            # 优化算法执行时间
    target_latency: float                  # 目标延迟
    actual_batch_size: int                 # 实际分配的batch size
    decode_count: int                      # decode请求数量
    prefill_count: int                     # prefill请求数量


class SLAOptimizer:
    """SLA感知三阶段优化器
    
    实现论文中描述的延迟导向统一调度算法，通过三个阶段的优化
    实现最佳的资源分配和SLA保证。
    """
    
    def __init__(self, config: SLASchedulerConfig, predictor: PerformancePredictor):
        """初始化优化器
        
        Args:
            config: SLA调度器配置
            predictor: 性能预测器实例
        """
        self.config = config
        self.predictor = predictor
        
        # 优化统计信息
        self.stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'timeout_count': 0,
            'avg_optimization_time_ms': 0.0,
        }
    
    def optimize_schedule(self, 
                         running_requests: List[Request],
                         waiting_requests: List[Request],
                         target_latency: float,
                         max_batch_size: int,
                         max_tokens: int) -> Optional[OptimizationResult]:
        """三阶段SLA感知调度优化
        
        实现Algorithm 1: Latency-Guided Unified Scheduling
        
        Args:
            running_requests: 运行中的请求列表
            waiting_requests: 等待中的请求列表
            target_latency: 目标延迟(ms)
            max_batch_size: 最大batch size限制
            max_tokens: 最大token数限制
            
        Returns:
            优化结果，失败时返回None
        """
        start_time = time.perf_counter()
        self.stats['total_optimizations'] += 1
        
        try:
            # 限制搜索范围以确保实时性
            batch_search_limit = len(running_requests) + len(waiting_requests)
            min_batch_size = max(1, len(running_requests))
            
            # 目标(对齐 E2E=TTFT+TPOT*output_len):最小化预测 E2E 代理。
            # 每个候选:decode 阶段代价 = TPOT_est * E_out;并对未接纳的等待请求加排队惩罚。
            # 低载(无排队)→ 偏小 batch → 低 TPOT;高载(排队多)→ 惩罚推动多接纳 → 保吞吐。
            # 所有研究旋钮走 _KNOBS(live 文件 > env > 默认),支持不重启热切换。
            slo_cap = float(self.config.slo_tpot_ms)
            E_out = _KNOBS.get_float("VLLM_SLA_EXPECTED_OUTPUT_LEN", 256.0)  # 预估输出长度
            q_penalty = _KNOBS.get_float("VLLM_SLA_QUEUE_PENALTY", 1.0)      # 未接纳请求排队惩罚系数
            # [E2/W1] TTFT 项权重:>0 时在 e2e 代理里加入
            #   ttft_w * (prefill 积压 tokens / 该候选的 prefill 速率),
            # 显式建模"把积压 prefill 排空需要多久"→ 有积压时奖励高 prefill 吞吐的候选。
            # 0 = 关闭(与原实现一致)。
            ttft_w = _KNOBS.get_float("VLLM_SLA_TTFT_W", 0.0)
            # [E1/W2] batch 搜索超时(ms),1ms 会截断升序搜索、系统性偏向小 batch
            opt_timeout_ms = _KNOBS.get_float(
                "VLLM_SLA_OPT_TIMEOUT_MS", float(self.config.optimization_timeout_ms))
            # [E4/W5] prefill 单请求单步 chunk 上限(原硬编码 512)
            max_chunk = _KNOBS.get_int("VLLM_SLA_MAX_CHUNK", 512)
            # [E3/W3+E9] token 预算求解所用的步延迟上限。
            # 关键机理:decode 速度=步频(每步每 decode 1 token),故步长直接决定 TPOT;
            # 而延迟模型近似线性于 S → 缩短步长几乎不损 prefill 吞吐。
            # 取值<=0 时 = 跟随自适应 target(队列空→短步快 decode;积压→长步猛 prefill)。
            budget_lat_ms = _KNOBS.get_float("VLLM_SLA_BUDGET_LAT_MS", slo_cap)
            if budget_lat_ms <= 0:
                budget_lat_ms = target_latency
            # [E7/H-cap] 步延迟(纯decode代理 tpot_est)可行性上限 = step_cap × 自适应
            # target。旧目标 min|pred-target| 的隐式限批正是红线快 decode 的来源;
            # 此处将其还原为【硬可行性约束】,可行集内仍用 E2E 代理评分,
            # 全不可行时回退到 tpot_est 最小的候选。0=关闭。
            step_cap = _KNOBS.get_float("VLLM_SLA_STEP_CAP", 0.0)
            step_cap_ms = step_cap * target_latency if step_cap > 0 else float("inf")
            n_waiting = len(waiting_requests)
            # 自适应 batch 下限——延迟↔吞吐前沿旋钮,随等待队列(积压)增长:
            #   低载(队列空)→ 下限≈1(小 batch、低 TPOT,匹配基线);
            #   高载(积压多)→ 下限升到上限 MIN_BATCH(大 batch、保吞吐,拿饱和区优势)。
            # MIN_BATCH=1 即关闭自适应(退化为原行为)。
            min_batch_cap = _KNOBS.get_int("VLLM_SLA_MIN_BATCH", 1)   # 自适应下限的上限
            min_batch_k = _KNOBS.get_float("VLLM_SLA_MIN_BATCH_K", 2.0)  # 随队列增长的斜率
            adaptive_floor = max(1, min(min_batch_cap, int(1 + min_batch_k * n_waiting)))
            min_batch_size = max(min_batch_size, min(adaptive_floor, batch_search_limit))
            self._max_chunk = max_chunk  # 供 _greedy_allocation 使用
            # [E10] prefill 配额与积压判断(budget 下限用)
            prefill_quantum = _KNOBS.get_int("VLLM_SLA_PREFILL_QUANTUM", 256)
            prefill_backlog_exists = bool(waiting_requests) or any(
                not self._is_decode_phase(r) for r in running_requests)
            best_result = None
            best_e2e = float('inf')
            fallback_result = None       # 兜底:取预测步延迟最低者
            fallback_pred = float('inf')
            min_error = 0.0              # 仅用于日志兼容

            if self.config.verbose_logging:
                logger.debug(f"Starting optimization(E2E): slo_cap={slo_cap:.1f}ms, E_out={E_out}, "
                           f"running={len(running_requests)}, waiting={n_waiting}")

            # Phase 1: 穷举搜索batch size并为每个候选配置执行贪心调度
            for batch_size in range(min_batch_size, batch_search_limit + 1):
                if (time.perf_counter() - start_time) * 1000 > opt_timeout_ms:
                    self.stats['timeout_count'] += 1
                    break

                # Phase 2: token 预算(prefill 用较大预算以压低 TTFT,受 SLO 与 max_tokens 约束)
                optimal_tokens = self.predictor.solve_for_token_budget(batch_size, budget_lat_ms)
                # [E10] 预算硬下限:B 个请求的步至少要 B 个 token(每个 decode 每步 1 token),
                # 否则短 target(如 15ms)下 solve 会解出 ~1 token → running 请求轮流饿死
                # (实测:B=1/S=1/pred=193ms 的坍塌步,系统在过载与饿死间振荡不收敛)。
                # 若还有 prefill 积压(running prefill 或 waiting),再加一个 prefill 配额
                # 保证 prefill 永续推进,避免 wait=0 但 running prefill 卡住。
                floor = batch_size
                if prefill_backlog_exists:
                    floor += prefill_quantum
                optimal_tokens = max(optimal_tokens, floor)
                optimal_tokens = min(optimal_tokens, max_tokens)
                if optimal_tokens <= 0:
                    continue

                # Phase 3: 贪心分配资源
                allocation = self._greedy_allocation(
                    running_requests, waiting_requests, batch_size, optimal_tokens
                )
                if not allocation:
                    continue

                scheduled_requests = [rid for rid, t in allocation.items() if t > 0]
                actual_batch_size = len(scheduled_requests)
                actual_tokens = sum(allocation.values())
                if actual_batch_size == 0 or actual_tokens <= 0:
                    continue

                predicted_latency = self.predictor.predict_latency(actual_batch_size, actual_tokens)
                # decode TPOT 代理:把该 batch 视作纯 decode 步(每序列 1 token)估每步时间
                tpot_est = self.predictor.predict_latency(actual_batch_size, actual_batch_size)
                # 本步接纳的等待请求数
                admitted_wait = sum(1 for req in waiting_requests
                                    if allocation.get(getattr(req, "request_id", None), 0) > 0)
                unadmitted = max(0, n_waiting - admitted_wait)
                decode_count, prefill_count = self._count_request_types(
                    running_requests, waiting_requests, allocation
                )
                # 预测 E2E 代理(ms):解码 TPOT*输出 + 未接纳请求排队惩罚
                e2e_score = tpot_est * E_out + unadmitted * q_penalty * tpot_est * E_out
                # [E2/W1] 显式 TTFT 项:预计排空当前 prefill 积压所需时间。
                # 积压 = running 未算完的 prefill + 所有 waiting 的 prompt;
                # 该候选的 prefill 速率 = 本步分给 prefill 的 tokens / 预测步时。
                # 有积压时奖励高 prefill 吞吐候选,无积压时该项为 0(退化为原目标)。
                if ttft_w > 0.0:
                    backlog = 0
                    for req in running_requests:
                        backlog += self._get_remaining_prefill_tokens(req)
                    for req in waiting_requests:
                        backlog += max(0, getattr(req, 'num_prompt_tokens', 0) or 256)
                    prefill_toks_this_step = max(0, actual_tokens - decode_count)
                    if backlog > 0:
                        if prefill_toks_this_step > 0 and predicted_latency > 0:
                            prefill_rate = prefill_toks_this_step / predicted_latency  # tok/ms
                            ttft_proxy = backlog / prefill_rate
                        else:
                            ttft_proxy = 10 * slo_cap * E_out  # 有积压却不做 prefill:重罚
                        e2e_score = ttft_w * ttft_proxy + tpot_est * E_out \
                            + unadmitted * q_penalty * tpot_est * E_out
                cand = OptimizationResult(
                    optimal_batch_size=batch_size,
                    optimal_token_budget=actual_tokens,
                    allocation=allocation,
                    predicted_latency=predicted_latency,
                    optimization_time_ms=0,
                    target_latency=target_latency,
                    actual_batch_size=actual_batch_size,
                    decode_count=decode_count,
                    prefill_count=prefill_count
                )
                # [E7/H-cap] 违反步延迟上限的候选不进入最优集(仍可作兜底)
                if tpot_est <= step_cap_ms and e2e_score < best_e2e:
                    best_e2e = e2e_score
                    best_result = cand
                if tpot_est < fallback_pred:
                    fallback_pred = tpot_est
                    fallback_result = cand

            if best_result is None:
                best_result = fallback_result
            
            # 设置优化时间
            optimization_time_ms = (time.perf_counter() - start_time) * 1000
            
            if best_result:
                best_result.optimization_time_ms = optimization_time_ms
                self.stats['successful_optimizations'] += 1
                self._update_stats(optimization_time_ms)

                # [诊断] 每 ~200 次记录一次决策与模型预测,供校准分析
                if self.stats['total_optimizations'] % 200 == 0:
                    logger.info(
                        "[SLA-DECISION] B=%d S=%d pred=%.1fms tpot_est=%.1fms "
                        "target=%.1fms cap=%.1fms run=%d wait=%d dec=%d pre=%d",
                        best_result.actual_batch_size,
                        best_result.optimal_token_budget,
                        best_result.predicted_latency,
                        self.predictor.predict_latency(
                            best_result.actual_batch_size,
                            best_result.actual_batch_size),
                        target_latency, step_cap_ms if step_cap > 0 else -1.0,
                        len(running_requests), n_waiting,
                        best_result.decode_count, best_result.prefill_count)
                
                if self.config.verbose_logging:
                    logger.debug(f"Optimization success: B={best_result.actual_batch_size}, "
                               f"S={sum(best_result.allocation.values())}, "
                               f"T_pred={best_result.predicted_latency:.1f}ms, "
                               f"error={min_error:.1f}ms, time={optimization_time_ms:.2f}ms")
                
                return best_result
            else:
                if self.config.verbose_logging:
                    logger.info(f"Optimization failed: no valid allocation found")
                
                return None
            
        except Exception as e:
            logger.error(f"Optimization failed with exception: {e}")
            return None
    
    def _greedy_allocation(self, 
                          running_requests: List[Request],
                          waiting_requests: List[Request],
                          batch_size: int,
                          token_budget: int) -> Dict[str, int]:
        """贪心分配算法
        
        实现论文中描述的三阶段贪心策略：
        1. Running中的decode请求（每个需要1 token）
        2. Running中的prefill请求
        3. Waiting中的新请求（按优先级排序）
        
        Args:
            running_requests: 运行中的请求
            waiting_requests: 等待中的请求
            batch_size: 目标batch size
            token_budget: token预算
            
        Returns:
            request_id -> tokens的分配字典
        """
        allocation = {}
        remaining_budget = token_budget
        remaining_slots = batch_size
        
        # Phase 1: 分类running请求
        decode_requests = []
        prefill_requests = []
        
        for req in running_requests:
            if self._is_decode_phase(req):
                decode_requests.append(req)
            else:
                prefill_requests.append(req)
        
        # Phase 2: 优先分配decode请求（每个1 token）
        for req in decode_requests:
            if remaining_budget >= 1 and remaining_slots > 0:
                allocation[req.request_id] = 1
                remaining_budget -= 1
                remaining_slots -= 1
            else:
                allocation[req.request_id] = 0
        
        # Phase 3: 分配running prefill请求
        # 按剩余token数排序，优先处理即将完成的请求
        prefill_requests.sort(key=lambda req: self._get_remaining_prefill_tokens(req))
        
        for req in prefill_requests:
            if remaining_budget <= 0 or remaining_slots <= 0:
                allocation[req.request_id] = 0
                continue
            
            remaining_tokens = self._get_remaining_prefill_tokens(req)
            # 限制chunk大小以避免过度占用资源(可经 VLLM_SLA_MAX_CHUNK 热调)
            chunk_cap = getattr(self, "_max_chunk", 512)
            max_chunk = min(remaining_tokens, chunk_cap)
            chunk_size = min(max_chunk, remaining_budget)
            
            allocation[req.request_id] = chunk_size
            remaining_budget -= chunk_size
            remaining_slots -= 1
        
        # Phase 4: 选择waiting请求
        if remaining_budget > 0 and remaining_slots > 0:
            selected_waiting = self._select_waiting_requests(
                waiting_requests, remaining_slots, remaining_budget
            )
            
            for req, tokens in selected_waiting:
                allocation[req.request_id] = tokens
                remaining_budget -= tokens
                remaining_slots -= 1
        
        return allocation
    
    def _select_waiting_requests(self, 
                               waiting_requests: List[Request],
                               remaining_slots: int,
                               remaining_budget: int) -> List[Tuple[Request, int]]:
        """选择等待队列中的请求
        
        Args:
            waiting_requests: 等待中的请求列表
            remaining_slots: 剩余slot数
            remaining_budget: 剩余token预算
            
        Returns:
            选中的请求及其token分配列表
        """
        if not waiting_requests or remaining_slots <= 0 or remaining_budget <= 0:
            return []
        
        # 按优先级排序（如果有优先级字段）
        # 注意：vLLM Request可能没有priority字段，需要兼容处理
        try:
            sorted_waiting = sorted(waiting_requests, 
                                  key=lambda req: getattr(req, 'priority', 0), 
                                  reverse=True)
        except AttributeError:
            # 如果没有priority字段，按FIFO顺序
            sorted_waiting = waiting_requests
        
        selected = []
        
        for req in sorted_waiting:
            if remaining_slots <= 0 or remaining_budget <= 0:
                break
            
            # 计算启动该请求需要的最小token数
            prompt_tokens = getattr(req, 'num_prompt_tokens', 0)
            if prompt_tokens <= 0:
                # 如果无法获取prompt长度，使用默认最小值
                min_startup_tokens = 16
            else:
                min_startup_tokens = min(16, prompt_tokens)
            
            if remaining_budget < min_startup_tokens:
                # 剩余预算不足以启动新请求
                break
            
            # 计算该请求的token分配(chunk 上限可经 VLLM_SLA_MAX_CHUNK 热调)
            chunk_cap = getattr(self, "_max_chunk", 512)
            max_chunk = min(prompt_tokens, chunk_cap) if prompt_tokens > 0 else 256
            chunk_size = min(max_chunk, remaining_budget)
            
            if chunk_size >= min_startup_tokens:
                selected.append((req, chunk_size))
                remaining_budget -= chunk_size
                remaining_slots -= 1
        
        return selected
    
    def _is_decode_phase(self, request: Request) -> bool:
        """判断请求是否处于decode阶段"""
        try:
            return request.num_computed_tokens >= request.num_prompt_tokens
        except AttributeError:
            # 如果字段不存在，假设是prefill阶段
            return False
    
    def _get_remaining_prefill_tokens(self, request: Request) -> int:
        """获取prefill请求的剩余token数"""
        try:
            return max(0, request.num_prompt_tokens - request.num_computed_tokens)
        except AttributeError:
            # 如果字段不存在，返回默认值
            return 256
    
    def _count_request_types(self, 
                           running_requests: List[Request],
                           waiting_requests: List[Request],
                           allocation: Dict[str, int]) -> Tuple[int, int]:
        """统计分配结果中的decode和prefill请求数量"""
        decode_count = 0
        prefill_count = 0
        
        for req in running_requests:
            if req.request_id in allocation and allocation[req.request_id] > 0:
                if self._is_decode_phase(req):
                    decode_count += 1
                else:
                    prefill_count += 1
        
        for req in waiting_requests:
            if req.request_id in allocation and allocation[req.request_id] > 0:
                prefill_count += 1  # 新请求都是prefill
        
        return decode_count, prefill_count
    
    def compute_adaptive_target_latency(self, queue_length: int) -> float:
        """计算自适应目标延迟
        
        基于队列长度的线性插值计算目标延迟，实现负载感知调度。
        
        Args:
            queue_length: 等待队列长度
            
        Returns:
            目标延迟(ms)
        """
        # 基于队列长度的线性插值(坡道三参可经 live knobs 热调)
        t_min = _KNOBS.get_float("VLLM_SLA_TARGET_MIN_MS",
                                 float(self.config.min_batch_time_ms))
        t_max = _KNOBS.get_float("VLLM_SLA_TARGET_MAX_MS",
                                 float(self.config.slo_tpot_ms))
        q_thresh = _KNOBS.get_float("VLLM_SLA_TARGET_QTHRESH",
                                    float(self.config.queue_threshold))
        if q_thresh > 0:
            k = (t_max - t_min) / q_thresh
            target = t_min + k * queue_length
        else:
            target = t_min

        # 确保有效下界：至少能处理1个token
        min_effective = max(
            t_min,
            self.predictor.fallback_intercept + self.predictor.fallback_slope
        )

        adaptive_target = max(min_effective, min(t_max, target))
        
        if self.config.verbose_logging:
            logger.debug(f"Adaptive target latency: Q={queue_length} -> {adaptive_target:.1f}ms")
        
        return adaptive_target
    
    def _update_stats(self, optimization_time_ms: float) -> None:
        """更新优化器统计信息"""
        # 计算移动平均
        alpha = 0.1  # 指数移动平均的权重
        if self.stats['avg_optimization_time_ms'] == 0:
            self.stats['avg_optimization_time_ms'] = optimization_time_ms
        else:
            self.stats['avg_optimization_time_ms'] = (
                alpha * optimization_time_ms + 
                (1 - alpha) * self.stats['avg_optimization_time_ms']
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取优化器统计信息"""
        total = self.stats['total_optimizations']
        success_rate = (self.stats['successful_optimizations'] / total) if total > 0 else 0.0
        
        return {
            'total_optimizations': total,
            'successful_optimizations': self.stats['successful_optimizations'],
            'success_rate': success_rate,
            'timeout_count': self.stats['timeout_count'],
            'avg_optimization_time_ms': self.stats['avg_optimization_time_ms'],
        }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'timeout_count': 0,
            'avg_optimization_time_ms': 0.0,
        }

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
ELRAR Engine Agent - 引擎状态采集与推送组件
负责将引擎内部调度状态实时推送到State Gateway
"""

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import logging
import statistics
import socket
import struct

logger = logging.getLogger(__name__)


@dataclass
class ELRARConfig:
    """ELRAR Engine Agent 配置类"""
    enabled: bool = True
    # 网络配置
    network_mode: str = "unicast"  # "unicast" | "broadcast"
    # UDP 点播配置
    gateway_host: Optional[str] = None  # State Gateway 主机地址
    gateway_port: int = 9999  # State Gateway 端口
    # UDP 广播配置（向后兼容）
    broadcast_port: int = 9999
    push_interval: int = 200  # ms
    engine_id: Optional[str] = None


@dataclass
class EngineState:
    """引擎状态向量"""
    engine_id: str
    timestamp_ms: int
    latency_pred_ms: float          # 延迟预测（下一调度周期）
    scheduling_mode: str            # "latency_optimized" | "throughput_optimized"
    # 前瞻性/容量信息
    pending_tokens_total: int       # 等待队列token总数
    kv_cache_free_blocks: int       # KV可用块数
    kv_cache_total_blocks: int      # KV总块数
    engine_capacity: float    # 引擎能力基线（tokens/s 或标准化分数）


class EngineAgent:
    """
    ELRAR Engine Agent
    
    仅支持通过环境变量进行配置：
    
    环境变量控制：
    - VLLM_ENABLE_ELRAR: 启用ELRAR状态采集
    - VLLM_ELRAR_NETWORK_MODE: 网络模式 ("unicast" | "broadcast")
    - VLLM_ELRAR_GATEWAY_HOST: State Gateway 主机地址（点播模式）
    - VLLM_ELRAR_GATEWAY_PORT: State Gateway 端口（点播模式）
    - VLLM_ELRAR_BROADCAST_PORT: 广播端口（广播模式）
    - VLLM_ELRAR_PUSH_INTERVAL: 推送间隔(ms)
    - VLLM_ELRAR_ENGINE_ID: 引擎标识符（推荐设置为该引擎对外服务的URL，便于Router对齐）
    - VLLM_ENGINE_URL/VLLM_API_BASE: 可用于自动推断URL
    - VLLM_PORT/PORT/VLLM_SCHEME: 用于自动拼装URL（默认http与8000）
    """
    
    def __init__(self):
        # 只使用环境变量初始化
        self._init_from_env()
        
        if not self.enabled:
            logger.info("ELRAR Engine Agent disabled")
            return
            
        if self.network_mode == "unicast":
            logger.info(f"ELRAR Engine Agent enabled: UDP unicast to {self.gateway_host}:{self.gateway_port}, "
                       f"interval={self.push_interval}ms, engine_id={self.engine_id}")
        else:
            logger.info(f"ELRAR Engine Agent enabled: UDP broadcast on port {self.broadcast_port}, "
                       f"interval={self.push_interval}ms, engine_id={self.engine_id}")
        
        # 状态缓存
        self._state_token_history: List[int] = []  # scheduled_tokens历史
        self._state_latency_history: List[float] = []  # latency历史
        self._last_push_time = 0
        
        # UDP Socket 配置（零阻塞）
        self._udp_socket = None
        self._setup_udp_socket()
        
        # 推送线程
        self._push_thread = None
        self._push_queue = []
        self._queue_lock = threading.Lock()
        
        if self.enabled:
            self._start_push_thread()
            # 立即发送初始化状态，让 cluster router 感知到引擎启动
            self._send_initialization_state()

    def _init_from_env(self):
        """从环境变量初始化"""
        self.enabled = os.getenv('VLLM_ENABLE_ELRAR', 'true').lower() == 'true'
        self.network_mode = os.getenv('VLLM_ELRAR_NETWORK_MODE', 'unicast')
        self.gateway_host = os.getenv('VLLM_ELRAR_GATEWAY_HOST')
        self.gateway_port = int(os.getenv('VLLM_ELRAR_GATEWAY_PORT', '9999'))
        self.broadcast_port = int(os.getenv('VLLM_ELRAR_BROADCAST_PORT', '9999'))
        self.push_interval = int(os.getenv('VLLM_ELRAR_PUSH_INTERVAL', '200'))  # ms
        # 优先显式指定，其次自动推断为可达URL，最后退回到进程标识
        self.engine_id = (
            os.getenv('VLLM_ELRAR_ENGINE_ID')
            or self._infer_engine_url()
            or f'engine_id-{os.getpid()}'
        )

    def _infer_engine_url(self) -> Optional[str]:
        """推断当前引擎对外可达的URL，优先读取环境变量，其次用Pod IP与端口拼装。
        返回形如 http://<ip>:<port> 的字符串；若无法推断，返回None。
        """
        # 显式提供的完整URL
        url = os.getenv('VLLM_ENGINE_URL') or os.getenv('VLLM_API_BASE')
        if url:
            return url
        try:
            host = socket.gethostbyname(socket.gethostname())
        except Exception:
            host = None
        port = os.getenv('VLLM_PORT') or os.getenv('PORT') or '8000'
        scheme = os.getenv('VLLM_SCHEME', 'http')
        if host:
            return f"{scheme}://{host}:{port}"
        return None
    
    def _setup_udp_socket(self):
        """设置UDP Socket（零阻塞）"""
        try:
            self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            if self.network_mode == "broadcast":
                # 广播模式设置
                self._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                self._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                logger.info(f"UDP Socket configured for broadcast on port {self.broadcast_port}")
            else:
                # 点播模式设置
                self._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                logger.info(f"UDP Socket configured for unicast to {self.gateway_host}:{self.gateway_port}")
            
            # 设置发送缓冲区大小，避免阻塞
            self._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
            # 设置非阻塞模式
            self._udp_socket.setblocking(False)
        except Exception as e:
            logger.warning(f"Failed to setup UDP socket: {e}")
            self._udp_socket = None    
    
    # 个人认为这里最好在调度器中判断
    def _determine_scheduling_mode(self, 
                                 scheduled_tokens_hist: List[int]) -> str:
        """判断调度模式"""
        if not scheduled_tokens_hist:
            return "balanced"

        hist_avg_tokens = statistics.mean(scheduled_tokens_hist)
        
        # 启发式规则：小chunk + 少token = 延迟优化
        if hist_avg_tokens < 256:
            return "latency_optimized"
        elif hist_avg_tokens > 1024:
            return "throughput_optimized"
        else:
            return "balanced"
    
    def collect_state(self, scheduler_output,
                      pending_tokens_total: Optional[int] = None,
                      kv_cache_free_blocks: Optional[int] = None,
                      kv_cache_total_blocks: Optional[int] = None,
                      latency_pred_ms: Optional[float] = None,
                      scheduling_mode: Optional[str] = None,
                      engine_capacity: Optional[float] = None) -> None:
        """
        采集引擎状态（在调度循环中调用）
        
        Args:
            scheduler_output: 调度器输出
            pending_tokens_total: 等待队列token总数（前瞻性工作量）
            kv_cache_free_blocks: KV缓存可用块数
            kv_cache_total_blocks: KV缓存总块数
            latency_pred_ms: 调度器给出的下一周期延迟预测
            scheduling_mode: 调度器判定的工作模式
            engine_capacity: 引擎能力基线（tokens/s）
        """
        if not self.enabled:
            return
        
        try:
            # 采集基础指标（当前批）
            total_tokens = scheduler_output.total_num_scheduled_tokens
            active_batches = len(scheduler_output.scheduled_new_reqs) + \
                           len(scheduler_output.scheduled_cached_reqs)
            
            # 计算总token数
            chunk_sizes = []
            for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
                if num_tokens > 0:
                    chunk_sizes.append(num_tokens)
            total_tokens = sum(chunk_sizes)
            # 更新token历史
            self._state_token_history.append(total_tokens)
            if len(self._state_token_history) > 10:  # 保持最近10次历史
                self._state_token_history.pop(0)
            
            # 预测延迟：若调度器未提供，使用简单启发式
            if latency_pred_ms is None:
                logger.warning("ELRAR: No latency prediction provided, using simple heuristic")
                base_latency = 8.7
                token_factor = 0.02
                batch_penalty = active_batches * 0.5
                latency_pred_ms = base_latency + total_tokens * token_factor + batch_penalty
            # 更新latency历史
            self._state_latency_history.append(latency_pred_ms)
            if len(self._state_latency_history) > 10:  # 保持最近10次历史
                self._state_latency_history.pop(0)
            
            
            # 工作模式：优先使用调度器判定；否则用本地启发式
            if scheduling_mode is not None:
                scheduling_mode = scheduling_mode
            else:
                scheduling_mode = self._determine_scheduling_mode(self._state_token_history)
            
            # KV容量：若未提供则用占位（-1）
            kv_free = int(kv_cache_free_blocks) if kv_cache_free_blocks is not None else -1
            kv_total = int(kv_cache_total_blocks) if kv_cache_total_blocks is not None else -1
            
            # 等待token总数：若未提供则用0占位
            pending_tokens = int(pending_tokens_total) if pending_tokens_total is not None else 0
            
            # 构建状态向量
            current_time = int(time.time() * 1000)
            state = EngineState(
                engine_id=self.engine_id,
                timestamp_ms=current_time,
                latency_pred_ms=float(statistics.mean(self._state_latency_history)),
                scheduling_mode=scheduling_mode,
                pending_tokens_total=pending_tokens,
                kv_cache_free_blocks=kv_free,
                kv_cache_total_blocks=kv_total,
                engine_capacity=float(engine_capacity),
            )
            
            # 周期性/事件触发推送（UDP广播，零阻塞）
            should_push = (current_time - self._last_push_time) >= self.push_interval
            if should_push:
                # 直接推送，不检查网络状态（UDP广播零阻塞）
                self._async_push_state(state)
                self._last_push_time = current_time
        
        except Exception as e:
            logger.warning(f"ELRAR state collection failed: {e}")
    
    def _async_push_state(self, state: EngineState) -> None:
        """异步推送状态（UDP广播，零阻塞）"""
        try:
            # 直接添加到推送队列，不检查任何网络状态
            with self._queue_lock:
                if len(self._push_queue) < 10:  # 限制队列大小
                    self._push_queue.append(state)
                # 队列满时直接丢弃，不阻塞
        except Exception as e:
            logger.debug(f"ELRAR state queue failed: {e}")
    
    def _start_push_thread(self) -> None:
        """启动后台推送线程（UDP广播，零阻塞）"""
        def push_worker():
            last_heartbeat_time = 0
            heartbeat_interval = 2000  # 2秒发送一次心跳状态
            
            while self.enabled:
                try:
                    current_time = int(time.time() * 1000)
                    
                    # 从队列获取状态（非阻塞）
                    states_to_send = []
                    with self._queue_lock:
                        if self._push_queue:
                            states_to_send = self._push_queue.copy()
                            self._push_queue.clear()
                    
                    # UDP发送所有状态（零阻塞）
                    for state in states_to_send:
                        try:
                            if self._udp_socket:
                                # 序列化状态
                                payload = json.dumps(asdict(state), ensure_ascii=False)
                                data = payload.encode('utf-8')
                                
                                if self.network_mode == "unicast":
                                    # UDP点播到指定的 State Gateway
                                    if self.gateway_host:
                                        try:
                                            self._udp_socket.sendto(data, (self.gateway_host, self.gateway_port))
                                            logger.debug(f"ELRAR state sent to {self.gateway_host}:{self.gateway_port}")
                                        except Exception as e:
                                            logger.debug(f"ELRAR UDP unicast failed: {e}")
                                    else:
                                        logger.warning("ELRAR: gateway_host not configured for unicast mode")
                                else:
                                    # UDP广播到所有可用地址
                                    broadcast_addresses = [
                                        ('<broadcast>', self.broadcast_port),
                                        ('127.0.0.1', self.broadcast_port),
                                        ('localhost', self.broadcast_port)
                                    ]
                                    
                                    for addr, port in broadcast_addresses:
                                        try:
                                            self._udp_socket.sendto(data, (addr, port))
                                        except Exception:
                                            continue  # 忽略单个地址的发送失败
                                    
                                    logger.debug(f"ELRAR state broadcasted: {state.engine_id}")
                        except Exception as e:
                            logger.debug(f"ELRAR UDP send failed: {e}")
                    
                    # 如果没有调度数据，定期发送心跳状态
                    if not states_to_send and (current_time - last_heartbeat_time) >= heartbeat_interval:
                        try:
                            heartbeat_state = EngineState(
                                engine_id=self.engine_id,
                                timestamp_ms=current_time,
                                latency_pred_ms=0.0,  # 心跳时延迟为0
                                scheduling_mode="latency_optimized",  # 心跳时使用延迟优化模式
                                pending_tokens_total=0,  # 心跳时无等待任务
                                kv_cache_free_blocks=-1,  # 心跳时KV缓存信息未知
                                kv_cache_total_blocks=-1,
                                engine_capacity=0.0,  # 心跳时能力基线为0
                            )
                            
                            # 发送心跳状态
                            self._send_state_immediately(heartbeat_state)
                            last_heartbeat_time = current_time
                            logger.debug(f"ELRAR heartbeat state sent: {self.engine_id}")
                            
                        except Exception as e:
                            logger.debug(f"ELRAR heartbeat send failed: {e}")
                    
                    # 短暂休眠，避免CPU占用过高
                    time.sleep(0.01)
                        
                except Exception as e:
                    # 静默处理异常，不影响主线程
                    logger.debug(f"ELRAR push worker error (silent): {e}")
                    time.sleep(0.01)
        
        self._push_thread = threading.Thread(target=push_worker, daemon=True)
        self._push_thread.start()
        logger.info("ELRAR push thread started (UDP broadcast mode)")

    def _send_initialization_state(self) -> None:
        """发送初始化状态，让 cluster router 感知到引擎启动"""
        try:
            current_time = int(time.time() * 1000)
            init_state = EngineState(
                engine_id=self.engine_id,
                timestamp_ms=current_time,
                latency_pred_ms=0.0,  # 初始化时延迟为0
                scheduling_mode="latency_optimized",  # 初始化时使用平衡模式
                pending_tokens_total=0,  # 初始化时无等待任务
                kv_cache_free_blocks=-1,  # 初始化时KV缓存信息未知
                kv_cache_total_blocks=-1,
                engine_capacity=0.0,  # 初始化时能力基线为0
            )
            
            # 直接发送初始化状态，不经过队列
            self._send_state_immediately(init_state)
            logger.info(f"ELRAR initialization state sent: {self.engine_id}")
            
        except Exception as e:
            logger.warning(f"ELRAR initialization state send failed: {e}")

    def _send_state_immediately(self, state: EngineState) -> None:
        """立即发送状态，不经过队列（用于初始化状态）"""
        if not self._udp_socket:
            return
            
        try:
            # 序列化状态
            payload = json.dumps(asdict(state), ensure_ascii=False)
            data = payload.encode('utf-8')
            
            if self.network_mode == "unicast":
                # UDP点播到指定的 State Gateway
                if self.gateway_host:
                    try:
                        self._udp_socket.sendto(data, (self.gateway_host, self.gateway_port))
                        logger.debug(f"ELRAR state sent immediately to {self.gateway_host}:{self.gateway_port}")
                    except Exception as e:
                        logger.debug(f"ELRAR UDP unicast failed: {e}")
                else:
                    logger.warning("ELRAR: gateway_host not configured for unicast mode")
            else:
                # UDP广播到所有可用地址
                broadcast_addresses = [
                    ('<broadcast>', self.broadcast_port),
                    ('127.0.0.1', self.broadcast_port),
                    ('localhost', self.broadcast_port)
                ]
                
                for addr, port in broadcast_addresses:
                    try:
                        self._udp_socket.sendto(data, (addr, port))
                    except Exception:
                        continue  # 忽略单个地址的发送失败
                
                logger.debug(f"ELRAR state broadcasted immediately: {state.engine_id}")
                
        except Exception as e:
            logger.debug(f"ELRAR immediate UDP send failed: {e}")
    
    def shutdown(self) -> None:
        """关闭Agent"""
        self.enabled = False
        if self._push_thread:
            self._push_thread.join(timeout=1.0)  # 减少等待时间
        if self._udp_socket:
            self._udp_socket.close()
        logger.info("ELRAR Engine Agent shutdown") 
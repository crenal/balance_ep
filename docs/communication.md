# 跨节点转发通信设计

## 通信流程
1. 同节点内通信直接写入目标 rank 的输出缓冲
2. 跨节点且 local_rank 相同：源 rank 直接写入目标 rank 输出缓冲
3. 跨节点且 local_rank 不同：源 rank 写入目标节点同号 local_rank 的 mid_buf，并置位 flag
4. 目标节点同号 local_rank 轮询 flag，读取 mid_buf，将数据转发到本节点内其它 local_rank
5. 转发完成后清零 flag

## mid_buf 与 flag 布局
- 每个 rank 分配 (nnodes-1) 个 mid_buf 块
- 每个 mid_buf 块大小为 input buffer 大小
- 每个 mid_buf 块配套一段 flag 数组，长度为 num_tokens_per_rank
- flag 索引与 token 索引一一对应

## intranode_index 扩展
intranode_index 以全局 token 为索引，记录：
- index：目标 rank 输出中的紧凑写入位置
- target_rank：目标 rank 编号
- route：路由类型，0 同节点直写，1 跨节点同号直写，2 跨节点转发

## API 变更
- DispatchConfig 增加 node_npes、nnodes、mid_buf、mid_flags
- pre_process 与 dispatch_tokens 使用 IntranodeIndex 数组作为索引结果

## 使用示例
```cpp
DispatchConfig cfg;
cfg.num_tokens_per_rank = num_tokens_per_rank;
cfg.expert_num = expert_num;
cfg.hidden_size = hidden_size;
cfg.bytes_per_elem = sizeof(float);
cfg.blocks_per_kernel = 8;
cfg.node_npes = node_npes;
cfg.nnodes = nnodes;
cfg.mid_buf = mid_buf;
cfg.mid_flags = mid_flags;

pre_process(routing_map, intranode_index, &cfg);
dispatch_tokens(input_tokens, output_tokens, intranode_index, &cfg);
```

## 限制与特征
- 依赖 rank 在节点内的连续映射关系
- 需要全局一致的 routing_map
- 转发路径使用 flag 同步，存在轮询开销

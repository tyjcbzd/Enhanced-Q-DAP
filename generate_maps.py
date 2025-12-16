import os
import json
import random
import numpy as np
from collections import deque

# ================= 配置区域 =================
# 地图尺寸
HEIGHT = 50
WIDTH = 50

# 起点和终点 [row, col]
START_POS = [0, 0]
GOAL_POS = [HEIGHT - 1, WIDTH - 1]

# 输出文件夹
OUTPUT_DIR = "maps"

# 生成任务配置：(密度, 数量)
TASKS = [
    (0.10, 10),  # 10% 密度，生成 10 个
    (0.25, 20),  # 25% 密度，生成 20 个
    (0.40, 20)   # 40% 密度，生成 20 个
]

# 基础随机种子（确保每次运行脚本生成的序列起点一致，后续会自动递增）
BASE_SEED = 2025
# ===========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def in_bounds(r, c, h, w):
    return 0 <= r < h and 0 <= c < w

def bfs_reachable(grid, start, goal):
    """
    使用 BFS 检查 grid 上 start 到 goal 是否可达。
    grid: 0 表示障碍物, 1 表示空地
    """
    h, w = grid.shape
    sr, sc = start
    gr, gc = goal
    
    # 检查起点终点是否被占用
    if grid[sr, sc] == 0 or grid[gr, gc] == 0:
        return False
        
    if (sr, sc) == (gr, gc):
        return True

    queue = deque([(sr, sc)])
    visited = np.zeros_like(grid, dtype=bool)
    visited[sr, sc] = True
    
    # 上下左右
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r, c = queue.popleft()
        if (r, c) == (gr, gc):
            return True
            
        for dr, dc in actions:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc, h, w) and not visited[nr, nc] and grid[nr, nc] == 1:
                visited[nr, nc] = True
                queue.append((nr, nc))
                
    return False

def generate_valid_map(h, w, density, start, goal, seed_start):
    """
    尝试生成一个可达的地图。
    如果 seed_start 生成的地图不可达，则 seed + 1 继续尝试，直到成功。
    返回: (obstacles_list, used_seed)
    """
    current_seed = seed_start
    max_retries = 10000
    
    for _ in range(max_retries):
        # 设置随机种子
        np.random.seed(current_seed)
        random.seed(current_seed)
        
        # 生成随机矩阵 (0.0 ~ 1.0)
        rand_grid = np.random.rand(h, w)
        
        # 生成障碍物掩码: True 为障碍物
        obstacle_mask = (rand_grid < density)
        
        # 强制清理起点和终点
        obstacle_mask[start[0], start[1]] = False
        obstacle_mask[goal[0], goal[1]] = False
        
        # 生成用于 BFS 的 grid (1: Free, 0: Obstacle)
        # numpy 取反用 ~
        free_grid = (~obstacle_mask).astype(int)
        
        # 验证可达性
        if bfs_reachable(free_grid, tuple(start), tuple(goal)):
            # 提取障碍物坐标列表
            # np.argwhere 返回 [[r1, c1], [r2, c2]...]
            obstacles = np.argwhere(obstacle_mask)
            # 转换为 Python list 且元素为 int (JSON 不支持 int64)
            obstacles_list = [[int(o[0]), int(o[1])] for o in obstacles]
            return obstacles_list, current_seed
        
        # 如果不可达，改变种子重试
        current_seed += 1
        
    raise RuntimeError(f"在 {max_retries} 次尝试后无法生成密度为 {density} 的可达地图。")

def main():
    ensure_dir(OUTPUT_DIR)
    
    global_seed_counter = BASE_SEED
    
    for density, count in TASKS:
        print(f"--- 开始生成密度 {density} 的地图，共 {count} 张 ---")
        
        for i in range(count):
            # 生成地图数据
            # 注意：我们将 global_seed_counter 传入，如果在函数内部重试了多次，
            # 我们需要更新 counter 以免下一张地图重复使用相同的无效种子序列
            try:
                obstacles, valid_seed = generate_valid_map(
                    HEIGHT, WIDTH, density, START_POS, GOAL_POS, global_seed_counter
                )
            except RuntimeError as e:
                print(e)
                break

            # 准备 JSON 数据
            map_data = {
                "height": HEIGHT,
                "width": WIDTH,
                "obstacle_density": density,
                "seed": valid_seed, # 保存实际生成成功的那个种子
                "start": START_POS, # 为了方便后续读取，建议也保存起点
                "goal": GOAL_POS,   # 为了方便后续读取，建议也保存终点
                "obstacles": obstacles
            }
            
            # 生成文件名，例如: map_0.25_15.json
            filename = f"map_den{str(density).replace('.', 'p')}_id{i}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(map_data, f, indent=2)
                
            print(f"  [完成] {filename} (Seed: {valid_seed}, Obstacles: {len(obstacles)})")
            
            # 更新全局种子计数器，确保下一张图从新的种子开始
            global_seed_counter = valid_seed + 1

    print(f"\n所有地图生成完毕，保存在 '{OUTPUT_DIR}' 目录下。")

if __name__ == "__main__":
    main()
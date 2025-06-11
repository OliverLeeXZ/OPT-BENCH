import random
import json
import numpy as np
from collections import defaultdict, Counter

def generate_random_graph(nodes, edge_probability=0.5):
    """
    生成一个具有指定节点数的随机无向图
    
    参数:
    nodes: 图中的节点数
    edge_probability: 在两个节点之间创建边的概率 (0.0 到 1.0)
    
    返回:
    dict: 包含节点数和邻接列表的字典
    """
    # 创建空邻接列表
    adjacency_list = {str(i): [] for i in range(nodes)}
    
    # 随机添加边
    for i in range(nodes):
        for j in range(i+1, nodes):  # 只处理上三角，确保无向图
            if random.random() < edge_probability:
                # 添加双向边
                adjacency_list[str(i)].append(j)
                adjacency_list[str(j)].append(i)
    
    # 创建图对象
    graph = {
        "nodes": nodes,
        "adjacency_list": adjacency_list
    }
    
    return graph

def generate_mst_based_graph(nodes, extra_edges_ratio=0.3):
    """
    基于最小生成树(MST)生成一个连通图
    首先生成一个MST确保图是连通的，然后随机添加额外的边
    
    参数:
    nodes: 图中的节点数
    extra_edges_ratio: 相对于MST的额外边的比例 (0.0到1.0)
    
    返回:
    dict: 包含节点数和邻接列表的字典
    """
    # 创建空邻接列表
    adjacency_list = {str(i): [] for i in range(nodes)}
    
    # 使用Kruskal算法的思想构建MST
    
    # 初始化并查集
    parent = list(range(nodes))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        parent[find(x)] = find(y)
    
    # 生成所有可能的边并随机排序
    all_edges = [(i, j) for i in range(nodes) for j in range(i+1, nodes)]
    random.shuffle(all_edges)
    
    # MST将包含n-1条边
    mst_edges = []
    for i, j in all_edges:
        if find(i) != find(j):
            union(i, j)
            mst_edges.append((i, j))
            # 添加到邻接列表
            adjacency_list[str(i)].append(j)
            adjacency_list[str(j)].append(i)
        
        # 当我们有n-1条边时，MST完成
        if len(mst_edges) == nodes - 1:
            break
    
    # 计算可以添加的额外边数量
    max_possible_edges = nodes * (nodes - 1) // 2  # 完全图的边数
    mst_edges_count = nodes - 1  # MST的边数
    remaining_edges = max_possible_edges - mst_edges_count  # 可以额外添加的最大边数
    extra_edges_count = int(mst_edges_count * extra_edges_ratio)  # 要添加的额外边数
    extra_edges_count = min(extra_edges_count, remaining_edges)  # 确保不超过可能的最大值
    
    # 从剩余的边中随机选择一些添加
    remaining_edges_list = [edge for edge in all_edges if edge not in mst_edges]
    extra_edges = random.sample(remaining_edges_list, extra_edges_count)
    
    # 添加额外的边到邻接列表
    for i, j in extra_edges:
        adjacency_list[str(i)].append(j)
        adjacency_list[str(j)].append(i)
    
    # 创建图对象
    graph = {
        "nodes": nodes,
        "adjacency_list": adjacency_list
    }
    
    return graph

def generate_degree_distribution_graph(nodes, degree_distribution, ensure_connected=True):
    """
    生成一个具有指定度数分布的随机图
    
    参数:
    nodes: 图中的节点数
    degree_distribution: 度数分布字典 {度数: 概率} 或列表 [(度数1, 概率1), (度数2, 概率2), ...]
    ensure_connected: 是否确保图是连通的
    
    返回:
    dict: 包含节点数和邻接列表的字典
    """
    # 将度数分布转换为字典
    if isinstance(degree_distribution, list):
        dist_dict = {d: p for d, p in degree_distribution}
    else:
        dist_dict = degree_distribution
    
    # 标准化概率
    total_prob = sum(dist_dict.values())
    norm_dist = {d: p/total_prob for d, p in dist_dict.items()}
    
    # 计算每个节点的度数
    degrees = list(norm_dist.keys())
    probabilities = list(norm_dist.values())
    node_degrees = np.random.choice(degrees, size=nodes, p=probabilities)
    
    # 确保总度数是偶数(每条边贡献两个度数)
    if sum(node_degrees) % 2 == 1:
        # 随机选择一个节点并增加或减少其度数
        idx = random.randrange(nodes)
        if node_degrees[idx] > min(degrees):
            node_degrees[idx] -= 1
        else:
            node_degrees[idx] += 1
    
    # 创建"桩"（每个节点的连接点）
    stubs = []
    for i in range(nodes):
        stubs.extend([i] * node_degrees[i])
    
    # 随机匹配桩来创建边
    random.shuffle(stubs)
    
    # 创建邻接列表
    adjacency_list = {str(i): [] for i in range(nodes)}
    
    # 配对桩形成边
    for i in range(0, len(stubs), 2):
        if i+1 < len(stubs):  # 确保有一对
            u, v = stubs[i], stubs[i+1]
            # 避免自环和重复边
            if u != v and v not in adjacency_list[str(u)]:
                adjacency_list[str(u)].append(v)
                adjacency_list[str(v)].append(u)
    
    # 如果需要确保连通性
    if ensure_connected:
        # 检查连通性
        visited = set()
        
        def dfs(node):
            visited.add(node)
            for neighbor in adjacency_list[str(node)]:
                if neighbor not in visited:
                    dfs(neighbor)
        
        # 从节点0开始深度优先搜索
        if nodes > 0:
            dfs(0)
        
        # 如果图不是连通的
        if len(visited) < nodes:
            # 找出不连通的组件
            unvisited = set(range(nodes)) - visited
            
            # 连接每个未访问的节点到已访问的节点集合
            for node in unvisited:
                # 随机选择一个已访问的节点连接
                connect_to = random.choice(list(visited))
                adjacency_list[str(node)].append(connect_to)
                adjacency_list[str(connect_to)].append(node)
                visited.add(node)
    
    # 创建图对象
    graph = {
        "nodes": nodes,
        "adjacency_list": adjacency_list
    }
    
    # 计算实际的度数分布
    actual_degrees = [len(neighbors) for neighbors in adjacency_list.values()]
    degree_counts = Counter(actual_degrees)
    graph["degree_distribution"] = {d: count/nodes for d, count in degree_counts.items()}
    
    return graph

def generate_connected_graph(nodes, min_degree=2, max_degree=None):
    """
    生成一个连通的随机无向图，确保每个节点至少有min_degree条边
    
    参数:
    nodes: 图中的节点数
    min_degree: 每个节点的最小度数
    max_degree: 每个节点的最大度数 (默认为节点数-1)
    
    返回:
    dict: 包含节点数和邻接列表的字典
    """
    if max_degree is None:
        max_degree = nodes - 1
    
    # 确保参数有效
    min_degree = max(1, min(min_degree, nodes-1))
    max_degree = max(min_degree, min(max_degree, nodes-1))
    
    # 创建空邻接列表
    adjacency_list = {str(i): [] for i in range(nodes)}
    
    # 确保图是连通的，通过创建一个环
    for i in range(nodes):
        next_node = (i + 1) % nodes
        adjacency_list[str(i)].append(next_node)
        adjacency_list[str(next_node)].append(i)
    
    # 随机添加更多的边，确保每个节点至少有min_degree条边
    for i in range(nodes):
        current_degree = len(adjacency_list[str(i)])
        # 如果当前度数小于最小度数，添加更多边
        while current_degree < min_degree:
            # 找一个不是自身且还没连接的节点
            possible_nodes = [j for j in range(nodes) if j != i and j not in adjacency_list[str(i)]]
            if not possible_nodes:  # 如果没有可能的节点
                break
            
            # 随机选择一个节点连接
            j = random.choice(possible_nodes)
            adjacency_list[str(i)].append(j)
            adjacency_list[str(j)].append(i)
            current_degree += 1
        
        # 可能随机添加更多边，但不超过最大度数
        while current_degree < max_degree and random.random() < 0.3:  # 30%几率添加更多边
            possible_nodes = [j for j in range(nodes) if j != i and j not in adjacency_list[str(i)]]
            if not possible_nodes:
                break
            
            j = random.choice(possible_nodes)
            adjacency_list[str(i)].append(j)
            adjacency_list[str(j)].append(i)
            current_degree += 1
    
    # 创建图对象
    graph = {
        "nodes": nodes,
        "adjacency_list": adjacency_list
    }
    
    return graph

def generate_power_law_graph(nodes, alpha=2.5, min_degree=1, max_degree=None):
    """
    生成一个具有幂律度数分布的随机图
    
    参数:
    nodes: 图中的节点数
    alpha: 幂律指数 (通常在2-3之间)
    min_degree: 最小度数
    max_degree: 最大度数 (默认为节点数的平方根)
    
    返回:
    dict: 包含节点数和邻接列表的字典
    """
    if max_degree is None:
        max_degree = int(np.sqrt(nodes))
    
    # 确保参数有效
    min_degree = max(1, min_degree)
    max_degree = min(max_degree, nodes-1)
    
    # 生成幂律分布
    degrees = range(min_degree, max_degree + 1)
    weights = [d ** (-alpha) for d in degrees]
    
    # 标准化
    total = sum(weights)
    normalized_weights = [w / total for w in weights]
    
    # 创建度数分布字典
    degree_distribution = {d: w for d, w in zip(degrees, normalized_weights)}
    
    # 使用前面定义的函数生成图
    return generate_degree_distribution_graph(nodes, degree_distribution)

def generate_barabasi_albert_graph(nodes, m=2):
    """
    使用Barabási–Albert模型生成具有优先连接特性的图
    新节点倾向于连接到已有的高度节点
    
    参数:
    nodes: 最终图中的节点数
    m: 每个新节点连接到现有节点的边数
    
    返回:
    dict: 包含节点数和邻接列表的字典
    """
    if nodes <= m:
        raise ValueError(f"节点数(nodes={nodes})必须大于每个新节点的边数(m={m})")
    
    # 创建邻接列表
    adjacency_list = {str(i): [] for i in range(nodes)}
    
    # 首先创建一个完全图，包含m个节点
    for i in range(m):
        for j in range(i+1, m):
            adjacency_list[str(i)].append(j)
            adjacency_list[str(j)].append(i)
    
    # 跟踪每个节点的度数，用于优先连接
    degrees = [len(adjacency_list[str(i)]) for i in range(m)]
    
    # 添加剩余节点，每个连接到m个现有节点
    for i in range(m, nodes):
        # 创建与现有节点度数成比例的概率分布
        total_degree = sum(degrees)
        if total_degree == 0:
            # 如果所有节点的度数都为0，则使用均匀分布
            probs = [1.0 / i for _ in range(i)]
        else:
            probs = [d / total_degree for d in degrees]
        
        # 选择m个不重复的节点连接
        connected = set()
        while len(connected) < m:
            # 按度数比例随机选择
            j = np.random.choice(range(i), p=probs)
            if j not in connected:
                connected.add(j)
                # 更新邻接列表
                adjacency_list[str(i)].append(j)
                adjacency_list[str(j)].append(i)
                # 更新度数
                degrees[j] += 1
        
        # 更新新节点的度数
        degrees.append(m)
    
    # 创建图对象
    graph = {
        "nodes": nodes,
        "adjacency_list": adjacency_list
    }
    
    # 计算实际的度数分布
    actual_degrees = [len(neighbors) for neighbors in adjacency_list.values()]
    degree_counts = Counter(actual_degrees)
    graph["degree_distribution"] = {d: count/nodes for d, count in degree_counts.items()}
    
    return graph

def save_graph_to_file(graph, filename):
    """将图保存到文件中"""
    # 创建一个可以被JSON序列化的副本
    serializable_graph = {
        "nodes": int(graph["nodes"]),  # 确保是Python原生int类型
        "adjacency_list": {}
    }
    
    # 处理邻接列表
    for node, neighbors in graph["adjacency_list"].items():
        # 确保所有的节点ID和邻居节点ID都是Python原生int类型
        serializable_graph["adjacency_list"][node] = [int(n) for n in neighbors]
    
    # 处理度数分布(如果存在)
    if "degree_distribution" in graph:
        serializable_graph["degree_distribution"] = {
            str(int(d)): float(freq) for d, freq in graph["degree_distribution"].items()
        }
    
    with open(filename, 'w') as f:
        json.dump(serializable_graph, f, indent=2)

def load_graph_from_file(filename):
    """从文件中加载图"""
    with open(filename, 'r') as f:
        graph = json.load(f)
    
    # 确保所有的节点ID都是字符串类型，邻居节点ID都是整数类型
    adjacency_list = {}
    for node, neighbors in graph["adjacency_list"].items():
        adjacency_list[str(node)] = [int(n) for n in neighbors]
    
    graph["adjacency_list"] = adjacency_list
    
    # 处理度数分布(如果存在)
    if "degree_distribution" in graph:
        degree_distribution = {}
        for degree, freq in graph["degree_distribution"].items():
            degree_distribution[int(degree)] = float(freq)
        graph["degree_distribution"] = degree_distribution
    
    return graph

def print_graph(graph):
    """打印图的信息"""
    print(f"节点数: {graph['nodes']}")
    print("邻接列表:")
    for node, neighbors in sorted(graph['adjacency_list'].items(), key=lambda x: int(x[0])):
        print(f"  {node}: {neighbors}")
    
    # 如果有度数分布信息，也打印出来
    if "degree_distribution" in graph:
        print("\n度数分布:")
        for degree, freq in sorted(graph["degree_distribution"].items()):
            print(f"  度数 {degree}: {freq:.2f} ({int(freq * graph['nodes'])} 个节点)")

if __name__ == "__main__":
    # 定义输出路径
    output_dir = "."  # 当前目录
    
    # 生成一个有20个节点的随机图
    node_count = 20
    random_graph = generate_random_graph(node_count, edge_probability=0.3)
    print("随机图:")
    print_graph(random_graph)
    # 保存随机图
    save_graph_to_file(random_graph, f"{output_dir}/random_graph.json")
    print(f"随机图已保存到 {output_dir}/random_graph.json")
    
    print("\n基于MST的连通图:")
    # 生成一个基于MST的图
    mst_graph = generate_mst_based_graph(node_count, extra_edges_ratio=0.5)
    print_graph(mst_graph)
    # 保存MST图
    save_graph_to_file(mst_graph, f"{output_dir}/mst_graph.json")
    print(f"MST图已保存到 {output_dir}/mst_graph.json")
    
    print("\n具有特定度数分布的图:")
    # 生成一个具有特定度数分布的图 - 例如幂律分布
    # 度数2: 60%, 度数3: 25%, 度数4: 10%, 度数5: 5%
    degree_dist = {2: 0.6, 3: 0.25, 4: 0.1, 5: 0.05}
    dist_graph = generate_degree_distribution_graph(node_count, degree_dist)
    print_graph(dist_graph)
    # 保存度数分布图
    save_graph_to_file(dist_graph, f"{output_dir}/degree_dist_graph.json")
    print(f"度数分布图已保存到 {output_dir}/degree_dist_graph.json")
    
    print("\n幂律分布图:")
    # 生成一个具有幂律度数分布的图
    power_law_graph = generate_power_law_graph(node_count, alpha=2.1, min_degree=2)
    print_graph(power_law_graph)
    # 保存幂律分布图
    save_graph_to_file(power_law_graph, f"{output_dir}/power_law_graph.json")
    print(f"幂律分布图已保存到 {output_dir}/power_law_graph.json")
    
    print("\nBarabási–Albert优先连接模型:")
    # 生成一个基于BA模型的图
    ba_graph = generate_barabasi_albert_graph(node_count, m=2)
    print_graph(ba_graph)
    # 保存BA模型图
    save_graph_to_file(ba_graph, f"{output_dir}/ba_graph.json")
    print(f"BA模型图已保存到 {output_dir}/ba_graph.json")
    
    print("\n连通图(传统方法):")
    # 生成一个有20个节点的连通图
    connected_graph = generate_connected_graph(node_count, min_degree=2, max_degree=5)
    print_graph(connected_graph)
    # 保存连通图
    save_graph_to_file(connected_graph, f"{output_dir}/connected_graph.json")
    print(f"连通图已保存到 {output_dir}/connected_graph.json")
    
    # 也保存到example_graph.json作为示例
    save_graph_to_file(ba_graph, f"{output_dir}/example_graph.json")
    print(f"\n图已保存到 {output_dir}/example_graph.json") 
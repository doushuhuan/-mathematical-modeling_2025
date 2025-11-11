% 定义性能数据
performance_data=[0.7667,0.7667,0.3833,1
    0.3833,0.3833,0.7667,0.3833
    0.7667,0.7667,0.3833,1
    0.3833,0.3833,1,0.7667
    1,0.3833,1,0.7667
    0.3833,0.3833,0.3833,0.7667
    0.3833,0.7667,0.7667,0.7667]


% 定义模型标签
model_labels={'随机森林', '梯度提升', '极端随机树', '支持向量机', '逻辑回归', 'K近邻', '朴素贝叶斯'}

% 定义性能特征标签
performance_labels={'B', 'IR', 'N', 'OR'}

% 自定义绿色系颜色映射
num_colors=256
green_start=[0.8, 1, 0.8] % 浅绿起始颜色
green_end=[0, 0.4, 0]   % 深绿结束颜色
r=linspace(green_start(1), green_end(1), num_colors)
g=linspace(green_start(2), green_end(2), num_colors)
b=linspace(green_start(3), green_end(3), num_colors)
cmap=[r', g', b']

% 绘制热力图
heatmap(performance_data, 'Colormap', cmap, 'XDisplayLabels', performance_labels, 'YDisplayLabels', model_labels)

% 设置标题
title('7种模型在4个类别上的性能热力图对比')

% 添加颜色条
colorbar
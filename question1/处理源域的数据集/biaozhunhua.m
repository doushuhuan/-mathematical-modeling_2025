%% 清除工作区并关闭所有图形
clear all close all clc

%% 1. 加载48kHz原始数据并截取前1024点
% ====================== 关键参数设置 =====================
filepath_48k='D:\比赛\9.21-9.24\数据集\源域数据集\48kHz_Normal_data\N_2_(1750rpm).mat'
segment_len=1024
excel_save_path='NormalN3.xlsx'
% ==========================================================


try
    % 加载数据
    data_struct_48k=load(filepath_48k)
    
    % 自动适配变量名
    if isfield(data_struct_48k, 'data')
        data_48k=data_struct_48k.data
    elseif isfield(data_struct_48k, 'signal')
        data_48k=data_struct_48k.signal
    elseif isfield(data_struct_48k, 'x')
        data_48k=data_struct_48k.x
    else
        field_names=fieldnames(data_struct_48k)
        data_48k=data_struct_48k.(field_names{1})
        fprintf('使用变量 "%s" 作为48kHz原始数据\n', field_names{1})
    end
    
    % 数据预处理
    data_48k=data_48k(:)  % 转为列向量
    if size(data_48k, 2) > 1
        data_48k=data_48k(:, 1)
        disp('48kHz数据为多通道，已提取第一通道')
    end
    
    % 截取前1024点
    if length(data_48k) < segment_len
        error('48kHz数据长度不足1024点！实际长度：%d点', length(data_48k))
    end
    data_48k_cut=data_48k(1:segment_len)
    fs_48k=48000
    t_48k_cut=(0:segment_len-1)/fs_48k
    
    % % 输出信息
    % fprintf('48kHz数据截取完成：\n')
    % fprintf('- 截取长度：%d点\n', segment_len)
    % fprintf('- 时间范围：0 ~ %.4f秒\n', t_48k_cut(end))
    % fprintf('- 原始数据幅值范围：%.6f ~ %.6f\n', min(data_48k_cut), max(data_48k_cut))

catch ME
    error('数据加载/截取失败：%s', ME.message)
end

%% 2. 归一化处理（修复核心部分）
target_range=[-1, 1]

% 执行归一化（现在可以安全获取两个输出参数）
[data_48k_norm, norm_params]=minmax_norm_cut(data_48k_cut, target_range)

%% 3. 可视化对比
figure('Position', [100, 100, 1000, 600], 'Name', '归一化前后对比')
subplot(2,1,1)
plot(t_48k_cut, data_48k_cut, 'b', 'LineWidth', 1.2)
title('归一化前（前1024点）')
xlabel('时间 [s]') ylabel('原始幅度')
ylim([-1,1])
grid on

subplot(2,1,2)
plot(t_48k_cut, data_48k_norm, 'r', 'LineWidth', 1.2)
title('归一化后（[-1,1]）')
xlabel('时间 [s]') ylabel('归一化幅度')
grid on 
ylim([-1,1])

writematrix(t_48k_cut, 'N_x.xlsx', ...
    'WriteMode', 'overwrite')
writematrix(data_48k_cut, 'N_signal_y.xlsx', ...
    'WriteMode', 'overwrite')
writematrix(data_48k_norm, 'N_y.xlsx', ...
    'WriteMode', 'overwrite')

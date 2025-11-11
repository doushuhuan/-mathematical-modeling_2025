%% 清除工作区并关闭所有图形
clear all
close all
clc

%% 导入.mat文件数据
% 请将以下文件路径替换为您的实际.mat文件路径
filepath_48k='48kHz数据文件路径'  % 48kHz数据文件路径
filepath_12k='12kHz数据文件路径'  % 12kHz数据文件路径

% 读取.mat文件
try
    % 加载48kHz数据
    data_struct_48k=load(filepath_48k)
    
    % 尝试获取常见的变量名，如果变量名不同，请修改下面的变量名
    if isfield(data_struct_48k, 'data')
        data_48k=data_struct_48k.data
    elseif isfield(data_struct_48k, 'signal')
        data_48k=data_struct_48k.signal
    elseif isfield(data_struct_48k, 'x')
        data_48k=data_struct_48k.x
    else
        % 获取第一个数值型变量
        field_names=fieldnames(data_struct_48k)
        data_48k=data_struct_48k.(field_names{1})
        fprintf('使用变量 "%s" 作为48kHz数据\n', field_names{1})
    end
    
    % 加载12kHz数据
    data_struct_12k=load(filepath_12k)
    
    % 尝试获取常见的变量名，如果变量名不同，请修改下面的变量名
    if isfield(data_struct_12k, 'data')
        data_12k=data_struct_12k.data
    elseif isfield(data_struct_12k, 'signal')
        data_12k=data_struct_12k.signal
    elseif isfield(data_struct_12k, 'x')
        data_12k=data_struct_12k.x
    else
        % 获取第一个数值型变量
        field_names=fieldnames(data_struct_12k)
        data_12k=data_struct_12k.(field_names{1})
        fprintf('使用变量 "%s" 作为12kHz数据\n', field_names{1})
    end
    
catch ME
    % 如果文件不存在或加载失败，使用示例数据
    % warning('文件加载失败，使用示例数据代替。错误信息: %s', ME.message)
    print("222222")
    % 生成示例数据：1kHz + 3kHz正弦波 + 噪声
    fs_48k=48000
    fs_12k=12000
    
    t_48k=(0:fs_48k*2-1)/fs_48k
    t_12k=(0:fs_12k*2-1)/fs_12k
    
    data_48k=sin(2*pi*1000*t_48k) + 0.5*sin(2*pi*3000*t_48k) + 0.1*randn(size(t_48k))
    data_12k=sin(2*pi*1000*t_12k) + 0.5*sin(2*pi*3000*t_12k) + 0.1*randn(size(t_12k))
    
    % 确保数据是列向量
    data_48k=data_48k(:)
    data_12k=data_12k(:)
end

% 设置采样率（从.mat文件中可能无法获取，需要用户确认或设置）
fs_48k=48000  % 请根据实际情况修改
fs_12k=12000  % 请根据实际情况修改

% 显示数据基本信息
fprintf('48kHz数据: 采样率=%dHz, 长度=%d样本, 持续时间=%.2f秒\n', ...
        fs_48k, length(data_48k), length(data_48k)/fs_48k)
fprintf('12kHz数据: 采样率=%dHz, 长度=%d样本, 持续时间=%.2f秒\n', ...
        fs_12k, length(data_12k), length(data_12k)/fs_12k)

%% 将48kHz数据降采样至12kHz
% 方法1: 使用decimate函数
data_48k_downsampled=decimate(data_48k, decimation_factor, 'fir')

% 方法2: 手动实现滤波+抽取（备用方法）
% 设计抗混叠低通滤波器
nyq_ratio=0.8 % 通常取0.8左右，留有余量
cutoff_freq=nyq_ratio * target_fs/2 % 截止频率

% 设计FIR滤波器
filter_order=100 % 滤波器阶数
b=fir1(filter_order, cutoff_freq/(fs_48k/2), 'low')

% 应用滤波器
filtered_data=filter(b, 1, data_48k)

% 抽取
data_48k_downsampled_manual=filtered_data(1:decimation_factor:end)

% 确保降采样后的数据长度与12kHz数据匹配
min_length=min([length(data_12k), length(data_48k_downsampled), length(data_48k_downsampled_manual)])
data_12k=data_12k(1:min_length)
data_48k_downsampled=data_48k_downsampled(1:min_length)
data_48k_downsampled_manual=data_48k_downsampled_manual(1:min_length)


% %% 时域比较
% t_compare=(0:min_length-1)/target_fs
% 
% figure('Position', [100, 100, 1000, 800], 'Name', '时域比较')
% subplot(3,1,1)
% plot(t_compare, data_12k, 'b', 'LineWidth', 1.5)
% title('原始12kHz数据')
% xlabel('时间 [s]')
% ylabel('幅度')
% grid on
% xlim([0, t_compare(end)])
% 
% subplot(3,1,2)
% plot(t_compare, data_48k_downsampled, 'r', 'LineWidth', 1.5)
% title('48kHz数据降采样至12kHz (使用decimate函数)')
% xlabel('时间 [s]')
% ylabel('幅度')
% grid on
% xlim([0, t_compare(end)])
% 
% subplot(3,1,3)
% plot(t_compare, data_48k_downsampled_manual, 'g', 'LineWidth', 1.5)
% title('48kHz数据降采样至12kHz (手动滤波+抽取)')
% xlabel('时间 [s]')
% ylabel('幅度')
% grid on
% xlim([0, t_compare(end)])

%% 频域比较
figure('Position', [100, 100, 1000, 800], 'Name', '频域比较')

% 计算功率谱密度
[Pxx_12k, f_12k]=pwelch(data_12k, 1024, 512, 1024, target_fs)
[Pxx_ds, f_ds]=pwelch(data_48k_downsampled, 1024, 512, 1024, target_fs)
[Pxx_ds_manual, f_ds_manual]=pwelch(data_48k_downsampled_manual, 1024, 512, 1024, target_fs)

subplot(2,2,[1,2])
semilogy(f_12k, Pxx_12k, 'b', 'LineWidth', 1.5)
hold on
semilogy(f_ds, Pxx_ds, 'r--', 'LineWidth', 1.5)
semilogy(f_ds_manual, Pxx_ds_manual, 'g:', 'LineWidth', 1.5)
xlim([0, target_fs/2])
title('功率谱密度比较')
xlabel('频率 [Hz]')
ylabel('功率谱密度')
legend('原始12kHz数据', '48kHz降采样数据 (decimate)', '48kHz降采样数据 (手动)')
grid on

% 计算频谱差异
subplot(2,2,3)
plot(f_12k, 10*log10(Pxx_ds) - 10*log10(Pxx_12k), 'r', 'LineWidth', 1.5)
hold on
plot(f_12k, 10*log10(Pxx_ds_manual) - 10*log10(Pxx_12k), 'g', 'LineWidth', 1.5)
xlim([0, target_fs/2])
title('频谱差异 (降采样数据 - 原始数据)')
xlabel('频率 [Hz]')
ylabel('差异 [dB]')
legend('decimate方法', '手动方法')
grid on
yline(0, 'k--')
ylim([-3, 3])

% 计算累积频谱差异
subplot(2,2,4)
cumulative_diff_decimate=cumsum(abs(10*log10(Pxx_ds) - 10*log10(Pxx_12k)))
cumulative_diff_manual=cumsum(abs(10*log10(Pxx_ds_manual) - 10*log10(Pxx_12k)))

plot(f_12k, cumulative_diff_decimate, 'r', 'LineWidth', 1.5)
hold on
plot(f_12k, cumulative_diff_manual, 'g', 'LineWidth', 1.5)
xlim([0, target_fs/2])
title('累积频谱差异')
xlabel('频率 [Hz]')
ylabel('累积差异 [dB]')
legend('decimate方法', '手动方法')
grid on

% %% 计算误差指标
% % 计算均方误差 (MSE)
% mse_decimate=mean((data_12k - data_48k_downsampled).^2)
% mse_manual=mean((data_12k - data_48k_downsampled_manual).^2)
% 
% % 计算信噪比 (SNR)
% signal_power=mean(data_12k.^2)
% noise_power_decimate=mean((data_12k - data_48k_downsampled).^2)
% noise_power_manual=mean((data_12k - data_48k_downsampled_manual).^2)
% 
% snr_decimate=10 * log10(signal_power / noise_power_decimate)
% snr_manual=10 * log10(signal_power / noise_power_manual)
% 
% % 计算相关系数
% corr_coef_decimate=corrcoef(data_12k, data_48k_downsampled)
% corr_coef_manual=corrcoef(data_12k, data_48k_downsampled_manual)
% 
% % 计算最大绝对误差
% max_abs_error_decimate=max(abs(data_12k - data_48k_downsampled))
% max_abs_error_manual=max(abs(data_12k - data_48k_downsampled_manual))

% %% 显示部分时域数据的差异
% figure('Position', [100, 100, 1000, 600], 'Name', '时域细节比较')
% plot_range=1:min(1000, min_length) % 只显示前1000个点
% 
% subplot(2,1,1)
% plot(t_compare(plot_range), data_12k(plot_range), 'b', 'LineWidth', 1.5)
% hold on
% plot(t_compare(plot_range), data_48k_downsampled(plot_range), 'r--', 'LineWidth', 1.5)
% title('时域数据对比 (前1000个点)')
% xlabel('时间 [s]')
% ylabel('幅度')
% legend('原始12kHz数据', '48kHz降采样数据 (decimate)')
% grid on
% 
% subplot(2,1,2)
% plot(t_compare(plot_range), data_12k(plot_range) - data_48k_downsampled(plot_range), 'k', 'LineWidth', 1.5)
% title('时域差异 (原始数据 - 降采样数据)')
% xlabel('时间 [s]')
% ylabel('幅度差异')
% grid on
% yline(0, 'r--')

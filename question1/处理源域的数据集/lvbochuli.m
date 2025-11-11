%% 清除环境并加载数据
clear all close all clc

%% 1. 数据加载与预处理（48kHz驱动轴数据）
filepath='D:\比赛\9.21-9.24\数据集\源域数据集\48kHz_DE_data\B\0007\B007_2.mat'
data_struct=load(filepath)

% 提取数据
if isfield(data_struct, 'data')
    raw_signal=data_struct.data
elseif isfield(data_struct, 'vibration')
    raw_signal=data_struct.vibration
else
    fields=fieldnames(data_struct)
    raw_signal=data_struct.(fields{1}) % 自动取第一个字段
    fprintf('使用变量 "%s" 作为数据\n', fields{1})
end
raw_signal=raw_signal(:) % 转为列向量
fs=48000 % 采样率48kHz

%% 2. 数据段选取（稳定运行阶段，2048点）
% 假设从第10000点开始截取稳定段（可根据实际数据调整起始位置）
start_idx=10000 
segment_len=2048 % 数据长度（包含足够多故障周期）
signal=raw_signal(start_idx:start_idx+segment_len-1)
t=(0:segment_len-1)/fs % 时间向量

%% 3. 去噪滤波处理
% 步骤1：带通滤波（保留2-10kHz，抑制台架低频和高频电子噪声）
low_cut=2000   % 低截止频率（Hz）
high_cut=10000 % 高截止频率（Hz）
order=150      % 高阶滤波器保证陡峭过渡带
nyquist=fs/2
Wn=[low_cut, high_cut]/nyquist % 归一化截止频率
window=hamming(order+1) % 窗函数
[b, a]=fir1(order, Wn, 'bandpass', window) % 设计带通滤波器
filtered_signal=filtfilt(b, a, signal) % 零相位滤波

% 步骤2：包络解调（凸显故障周期性冲击）
analytic_signal=hilbert(filtered_signal) % 解析信号
envelope=abs(analytic_signal) % 提取包络（故障冲击的低频特征）

%% 4. 结果可视化
figure('Position', [100, 100, 1200, 900])

% 原始信号及时域去噪效果
subplot(3,1,1)
plot(t, signal, 'b')
title('原始48kHz驱动轴振动信号（稳定段）')
xlabel('时间（s）') ylabel('加速度')
grid on 
%ylim([-1, 1])

subplot(3,1,2)
plot(t, filtered_signal, 'r')
title('带通滤波后信号（2-10kHz）')
xlabel('时间（s）') ylabel('加速度')
% grid on 
%ylim([-1, 1])

% subplot(3,1,3)
% plot(t, envelope, 'k', 'LineWidth', 1.2)
% title('包络信号（凸显故障周期性冲击）')
% xlabel('时间（s）') ylabel('包络幅度')
% grid on

% % 频谱对比（验证噪声抑制效果）
% figure('Position', [1300, 100, 1200, 600])
% % 原始信号频谱
% [Pxx_raw, f_raw]=pwelch(signal, [], [], 8192, fs)
% subplot(2,1,1)
% plot(f_raw, 10*log10(Pxx_raw), 'b')
% hold on
% % 标记噪声频段
% area([0, low_cut], [-60, -60], 'FaceColor',[0.8,0.8,1], 'EdgeColor','none', 'DisplayName','台架低频噪声区')
% area([high_cut, nyquist], [-60, -60], 'FaceColor',[1,0.8,0.8], 'EdgeColor','none', 'DisplayName','高频电子噪声区')
% title('原始信号频谱（含噪声）')
% xlabel('频率（KHz）') ylabel('功率谱密度（dB）')
% xlim([0, nyquist]) grid on legend

% % 滤波后信号频谱
% [Pxx_filtered, f_filtered]=pwelch(filtered_signal, [], [], 8192, fs)
% subplot(2,1,2)
% plot(f_filtered, 10*log10(Pxx_filtered), 'r')
% title('带通滤波后频谱（2-10kHz）')
% xlabel('频率（kHz）') ylabel('功率谱密度（dB）')
% xlim([0, nyquist]) grid on
writematrix(t, 'B_x.xlsx', ...
    'WriteMode', 'overwrite')
writematrix(signal, 'B_signal_y.xlsx', ...
    'WriteMode', 'overwrite')
writematrix(filtered_signal, 'B_y.xlsx', ...
    'WriteMode', 'overwrite')
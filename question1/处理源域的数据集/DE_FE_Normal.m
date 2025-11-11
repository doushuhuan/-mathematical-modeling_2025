%% DE&NORMAL  FE&NORMAL  对比

A12FE='D:\比赛\9.21-9.24\数据集\源域数据集\12kHz_FE_data\B\0007\B007_2.mat' 
data_struct_12FE=load(A12FE)
A12DE='D:\比赛\9.21-9.24\数据集\源域数据集\12kHz_DE_data\B\0007\B007_2.mat' 
data_struct_12DE=load(A12DE)
filepath_Normal='D:\比赛\9.21-9.24\数据集\源域数据集\48kHz_Normal_data\N_2_(1750rpm).mat' 
data_struct_Normal=load(filepath_Normal)

% 采样率设置
fs_12=12000       % FE和DE数据采样率
fs_normal=48000   % 正常数据采样率
target_fs=12000   % 统一目标采样率
decimation_factor=fs_normal / target_fs  % 48k->12k的降采样因子

% 数据加载函数
function data=load_signal(data_struct)
    % 尝试常见变量名
    if isfield(data_struct, 'data')
        data=data_struct.data
    elseif isfield(data_struct, 'signal')
        data=data_struct.signal
    elseif isfield(data_struct, 'x')
        data=data_struct.x
    else
        % 自动获取第一个字段
        fields=fieldnames(data_struct)
        data=data_struct.(fields{1})
        fprintf('使用变量 "%s" 作为数据\n', fields{1})
    end
    % 确保为列向量且单通道
    data=data(:)
    if size(data, 2) > 1
        data=data(:, 1)
        disp('数据为多通道，已取第一通道')
    end
end

try
    % 加载数据
    data_struct_12FE=load(A12FE)
    data_12FE=load_signal(data_struct_12FE)
    
    data_struct_12DE=load(A12DE)
    data_12DE=load_signal(data_struct_12DE)
    
    data_struct_Normal=load(filepath_Normal)
    data_Normal=load_signal(data_struct_Normal)
    
    % 将正常数据降采样至12kHz
    data_Normal_downsampled=decimate(data_Normal, decimation_factor, 'fir')
    
    % 数据长度检查与截取（1024点）
    segment_len=1024
    start_idx=1  % 可修改起始位置，如改为随机起始: randi(length(data_12FE)-segment_len+1)
    
    % 截取1024点数据段
    data_12FE_seg=data_12FE(start_idx:start_idx+segment_len-1)
    data_12DE_seg=data_12DE(start_idx:start_idx+segment_len-1)
    data_Normal_seg=data_Normal_downsampled(start_idx:start_idx+segment_len-1)
    
    % 确保截取后长度一致
    if length(data_12FE_seg) ~= segment_len || ...
       length(data_12DE_seg) ~= segment_len || ...
       length(data_Normal_seg) ~= segment_len
        error('数据长度不足，无法截取1024点')
    end

catch ME
    error('数据处理失败：%s', ME.message)
end

%% 时域对比
t=(0:segment_len-1)/target_fs  % 时间向量

% FE vs Normal
figure('Position', [100, 100, 1000, 400], 'Name', 'FE与Normal时域对比')
subplot(2,1,1)
plot(t, data_12FE_seg, 'b', 'LineWidth', 1.2)
title('12kHz FE数据 (1024点)')
xlabel('时间 [s]')
ylabel('幅度')
ylim([-0.5,0.5])
grid on

subplot(2,1,2)
plot(t, data_Normal_seg, 'r', 'LineWidth', 1.2)
title('降采样至12kHz的Normal数据 (1024点)')
xlabel('时间 [s]')
ylabel('幅度')
ylim([-0.5,0.5])
grid on


% DE vs Normal
figure('Position', [100, 500, 1000, 400], 'Name', 'DE与Normal时域对比')
subplot(2,1,1)
plot(t, data_12DE_seg, 'b', 'LineWidth', 1.2)
title('12kHz DE数据 (1024点)')
xlabel('时间 [s]')
ylabel('幅度')
ylim([-0.5,0.5])
grid on

subplot(2,1,2)
plot(t, data_Normal_seg, 'r', 'LineWidth', 1.2)
title('降采样至12kHz的Normal数据 (1024点)')
xlabel('时间 [s]')
ylabel('幅度')
ylim([-0.5,0.5])
grid on


% 
% %% 差异分析
% % 计算误差指标函数
% function [mse, snr, corr]=calc_metrics(signal1, signal2)
%     mse=mean((signal1 - signal2).^2)
%     signal_power=mean(signal1.^2)
%     noise_power=mean((signal1 - signal2).^2)
%     snr=10 * log10(signal_power / noise_power)
%     corr=corrcoef(signal1, signal2)
%     corr=corr(1,2)
% end
% 
% % FE与Normal的误差指标
% [fe_mse, fe_snr, fe_corr]=calc_metrics(data_12FE_seg, data_Normal_seg)
% % DE与Normal的误差指标
% [de_mse, de_snr, de_corr]=calc_metrics(data_12DE_seg, data_Normal_seg)

writematrix(t, '12_x.xlsx', ...
    'WriteMode', 'overwrite') 
writematrix(data_12FE_seg, '12_FE_B_y.xlsx', ...
    'WriteMode', 'overwrite') 
writematrix(data_Normal_seg, '12_N_y.xlsx', ...
    'WriteMode', 'overwrite') 
writematrix(data_12DE_seg, '12_DE_B_y.xlsx', ...
    'WriteMode', 'overwrite') 

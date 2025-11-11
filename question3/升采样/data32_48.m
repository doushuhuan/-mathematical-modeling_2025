% 数据处理顺序：先升采样，再截取1024点，最后归一化
% 读取.mat文件
matFileName='D:\比赛\9.21-9.24\数据集\目标域数据集\P.mat'  % 你的文件路径

dataStruct=load(matFileName)

% 获取文件中所有变量名
varNames=fieldnames(dataStruct)

% 自动获取数据变量
if length(varNames)==1
    signal=dataStruct.(varNames{1})  % 直接读取唯一变量
    fprintf('自动识别变量名: %s\n', varNames{1})
else
    fprintf('.mat文件中有多个变量：\n')
    for i=1:length(varNames)
        fprintf('%d. %s\n', i, varNames{i})
    end
    choice=input('请输入数据所在变量的序号: ')
    signal=dataStruct.(varNames{choice})
end

% 确保数据是列向量
if isrow(signal)
    signal=signal'
end

% 定义采样率
original_fs=32000  % 原始采样率32KHz
target_fs=48000    % 目标采样率48KHz

% 先对完整数据进行升采样（32K→48K）
up_factor=3    % 上采样因子（32*3=96）
down_factor=2  % 下采样因子（96/2=48）
signal_upsampled_full=resample(signal, up_factor, down_factor)

% 从升采样后的完整数据中截取前1024点
upsampled_length=length(signal_upsampled_full)
if upsampled_length >= 1024
    signal_upsampled=signal_upsampled_full(1:1024)  % 截取48KHz下的前1024点
else
    signal_upsampled=signal_upsampled_full
    warning('升采样后的数据长度不足1024点，已使用全部数据（%d点）', upsampled_length)
end

% 归一化处理（缩放到[-1, 1]）
max_val=max(abs(signal_upsampled))
if max_val > 0
    signal_normalized=signal_upsampled / max_val
else
    signal_normalized=signal_upsampled
end

writematrix(signal_normalized, 'P.xlsx', ...
    'WriteMode', 'overwrite')  % 覆盖现有文件（默认）
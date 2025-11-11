%% 1. 定义时域特征计算函数（包含公式）
function features=calc_time_features(signal)
    % 峰值：信号绝对值的最大值
    features.peak=max(abs(signal)) % 公式：peak=max|x(i)|
    
    % 均方根（RMS）：信号平方的平均值开根号
    features.rms=sqrt(mean(signal.^2)) % 公式：RMS=√(1/NΣx(i)²)
    
    % 峭度：描述信号峰值特性（减去3使正态分布峭度为0）
    features.kurtosis=kurtosis(signal) % 公式：kurt=(1/NΣx(i)⁴)/(1/NΣx(i)²)² - 3
    
    % 峰值因子：峰值与RMS的比值
    features.crest_factor=features.peak / features.rms % 公式：crest=peak / RMS
    
    % 偏度：描述信号分布的不对称性
    features.skewness=skewness(signal) % 公式：skew=(1/NΣx(i)³)/(1/NΣx(i)²)^(3/2)
end
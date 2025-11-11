% 计算误差指标函数
function [mse, snr, corr]=calc_metrics(signal1, signal2)
    mse=mean((signal1 - signal2).^2)
    signal_power=mean(signal1.^2)
    noise_power=mean((signal1 - signal2).^2)
    snr=10 * log10(signal_power / noise_power)
    corr=corrcoef(signal1, signal2)
    corr=corr(1,2)
end
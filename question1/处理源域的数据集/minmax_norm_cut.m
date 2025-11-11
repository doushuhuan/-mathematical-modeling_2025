% 归一化函数
function [norm_data, norm_params]=minmax_norm_cut(data, target_range)
    % 先计算基本统计量
    data_min=min(data)
    data_max=max(data)
    data_mean=mean(data)
    
    % 初始化参数结构体（关键修复：无论何种情况都先定义）
    norm_params=struct()
    norm_params.original_min=data_min
    norm_params.original_max=data_max
    norm_params.original_mean=data_mean
    norm_params.target_range=target_range
    norm_params.sample_count=length(data)
    
    % 处理数据
    if data_max==data_min
        % 数据无波动情况
        norm_data=zeros(size(data))
        fprintf('警告：数据无波动，归一化后为0向量\n')
    else
        % 正常归一化
        norm_data=(data - data_min) / (data_max - data_min) ...
                    * (target_range(2) - target_range(1)) + target_range(1)
    end
end
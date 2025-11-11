function main_code()
    %% 数据筛选 DE 48KHZ centered 007_2
    X124_cut=X124_DE_time(1:1024)
    X111_cut=X111_DE_time(1:1024)
    X137_cut=X137_DE_time(1:1024)
    X099_cut=X099_DE_time(1:1024)
    combinedTable=table(X124_cut, X111_cut,X137_cut, X099_cut, ...
        'VariableNames', {'B', 'IR', 'OR', 'Normal'})

    %% 2. 噪声处理（带通滤波去除无关频段噪声）
    % 2.1 滤波器参数设置（48KHz采样率下的轴承故障特征频率通常在1-10KHz）
    Fs = 48000         % 采样频率：48KHz
    %计算低频和高频
    frOR=X137RPM%外圈转速n
    frIR=X111RPM%内圈转速n
    frB=X124RPM%滚动体转速n
    n=9%滚动体数
    dB=0.3126%滚动体直径
    D=1.537%轴承节径
    BPFO=frB*(n/2)*(1-(dB/D))%外圈故障特征频率
    BPFI=frIR*(n/2)*(1+(dB/D))%内圈故障特征频率
    BSF=frOR*(D/dB)*(1-(dB/D)^2)%滚动体故障特征频率
    freq_vector=[BPFO,BPFI,BSF]% 将三个频率组成向量
    lowcut=min(freq_vector)%低频截止
    highcut=max(freq_vector)%高频截止
    order = 6          % 滤波器阶数（平衡滤波效果与信号失真）
    
    % 2.2 设计Butterworth带通滤波器
    [bf, af] = butter(order, [lowcut, highcut]/(Fs/2), 'bandpass')
    
    % 2.3 对每类数据进行滤波（filtfilt避免相位偏移）
    B_filtered = filtfilt(bf, af, combinedTable.B)
    IR_filtered = filtfilt(bf, af, combinedTable.IR)
    OR_filtered = filtfilt(bf, af, combinedTable.OR)
    Normal_filtered = filtfilt(bf, af, combinedTable.Normal)
    
    % 2.4 整合滤波后的数据
    filteredTable = table(B_filtered, IR_filtered, OR_filtered, Normal_filtered, ...
        'VariableNames', {'B', 'IR', 'OR', 'Normal'})
    
    
    %% 3. 数据归一化（Z-score标准化，消除量纲影响）
    % 3.1 计算每类数据的均值和标准差（用于归一化）
    stats = struct()
    for col = 1:width(filteredTable)
        data = filteredTable{:, col}
        stats(col).mean = mean(data)   % 均值
        stats(col).std = std(data)     % 标准差
    end
    
    % 3.2 执行Z-score归一化：x_norm = (x - mean) / std
    B_norm = (filteredTable.B - stats(1).mean) / stats(1).std
    IR_norm = (filteredTable.IR - stats(2).mean) / stats(2).std
    OR_norm = (filteredTable.OR - stats(3).mean) / stats(3).std
    Normal_norm = (filteredTable.Normal - stats(4).mean) / stats(4).std
    
    % 3.3 整合归一化后的数据
    normalizedTable = table(B_norm, IR_norm, OR_norm, Normal_norm, ...
        'VariableNames', {'B', 'IR', 'OR', 'Normal'})
    
    
    %% 4. 结果可视化（验证处理效果）
    % 4.1 绘制原始信号与滤波后信号对比（以B故障为例）
    figure('Name', '噪声处理效果对比')
    subplot(2,1,1)
    plot(combinedTable.B).title('原始B故障信号')
    xlabel('采样点').ylabel('振幅')
    subplot(2,1,2)
    plot(filteredTable.B).title('带通滤波后B故障信号')
    xlabel('采样点').ylabel('振幅')
    legend('噪声已去除')
    
    % 4.2 绘制归一化前后信号对比（以正常数据为例）
    figure('Name', '归一化效果对比')
    subplot(2,1,1)
    plot(filteredTable.Normal).title('滤波后正常信号')
    ylim([-0.02, 0.02]).xlabel('采样点').ylabel('振幅')
    subplot(2,1,2)
    plot(normalizedTable.Normal).title('归一化后正常信号')
    ylim([-3, 3]).xlabel('采样点').ylabel('归一化振幅')
    legend('均值=0，标准差=1')
    
    % 4.3 显示最终处理结果
    disp('归一化后的带标签数据表格：')
    normalizedTable
    
    
    
    %% 2. 对每类状态数据计算时域特征
    % 提取表格中的数据（B故障、IR故障、OR故障、正常）
    states = normalizedTable.Properties.VariableNames % 状态名称：{'B', 'IR', 'OR', 'Normal'}
    featureNames = {'peak', 'rms', 'kurtosis', 'crest_factor', 'skewness'} % 特征名称
    
    % 初始化特征结果表（行：状态；列：特征）
    % 1. 先定义每列的数据类型（所有特征都是数值型，用'double'）
    varTypes = repmat({'double'}, 1, length(featureNames)) % 生成和列数一致的类型数组
    
    % 2. 正确创建空表（补充'VariableTypes'）
    featureTable = table('Size', [length(states), length(featureNames)], ...
        'VariableNames', featureNames, ...  % 列名：特征名
        'RowNames', states, ...             % 行名：轴承状态
        'VariableTypes', varTypes)        % 关键补充：列数据类型（全为double）
    
    % 循环计算每个状态的特征
    for i = 1:length(states)
        signal = normalizedTable.(states{i}) % 获取对应状态的信号
        feat = calc_time_features(signal)    % 计算特征
    
        % 存入结果表
        featureTable.peak(i) = feat.peak
        featureTable.rms(i) = feat.rms
        featureTable.kurtosis(i) = feat.kurtosis
        featureTable.crest_factor(i) = feat.crest_factor
        featureTable.skewness(i) = feat.skewness
    end
    
    
    %% 3. 特征结果展示
    disp('各状态的时域特征值：')
    featureTable  % 打印特征表格
    
    
    %% 4. 特征可视化（对比不同状态的特征差异）
    figure('Name', '时域特征对比', 'Position', [100, 100, 1000, 600])
    barWidth = 0.15 % 柱状图宽度
    x = 1:length(states) % x轴坐标（状态索引）
    
    % 绘制每个特征的柱状图
    for i = 1:length(featureNames)
        subplot(2, 3, i) % 2行3列布局（最后一个留空）
        bar(x + (i-1)*barWidth - barWidth*(length(featureNames)-1)/2, ...
            featureTable.(featureNames{i}), barWidth)
        title(featureNames{i}, 'FontWeight', 'bold')
        set(gca, 'XTick', x, 'XTickLabel', states, 'FontName', 'SimHei')
        grid onbox on
        if i >= 4  % 底部特征添加x轴标签
            xlabel('轴承状态')
        end
    end
    
    % 调整布局
    sgtitle('不同轴承状态的时域特征对比', 'FontSize', 14, 'FontWeight', 'bold')
    
    
    %% 5. 特征分析结论（修正条件判断语法）
    disp(' ')
    disp('特征分析结论：')
    for i = 1:length(featureNames)
        % 获取正常状态和故障状态的特征值
        normalVal = featureTable.(featureNames{i})(4) % 正常状态值
        faultVals = featureTable.(featureNames{i})(1:3) % 故障状态值
        maxFaultVal = max(faultVals) % 故障状态最大值
    
        % 计算差异百分比
        diffRatio = abs(maxFaultVal - normalVal) / normalVal * 100
    
        % 修正：用if-else替代三元运算符
        if diffRatio > 50
            sensitivity = '对故障更敏感'
        else
            sensitivity = '对故障敏感性一般'
        end
    
        % 输出结果
        fprintf('%s：故障状态最大值（%.2f）与正常状态（%.2f）差异为%.1f%%，%s\n', ...
            featureNames{i}, maxFaultVal, normalVal, diffRatio, sensitivity)
    end
end
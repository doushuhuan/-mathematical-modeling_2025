function plotTrainingProgress(lossHistory, accuracyHistory)
    % 绘制训练进度
    figure;
    
    subplot(1, 2, 1);
    plot(lossHistory);
    xlabel('迭代次数');
    ylabel('损失值');
    title('训练损失曲线');
    grid on;
    
    if ~isempty(accuracyHistory)
        subplot(1, 2, 2);
        plot(accuracyHistory);
        xlabel('迭代次数');
        ylabel('准确率');
        title('验证准确率曲线');
        grid on;
    end
end
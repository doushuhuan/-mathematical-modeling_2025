% 清除环境
clear clc close all

% 设置数据文件夹路径
data_folder='Excel文件的文件夹路径' % Excel文件的文件夹路径
file_pattern='*.xlsx' % 文件模式，可以修改为*.xls等

% 获取所有数据文件
file_list=dir(fullfile(data_folder, file_pattern))
num_files=length(file_list)

if num_files==0
    error('在文件夹 %s 中没有找到匹配 %s 的文件', data_folder, file_pattern)
end

fprintf('找到 %d 个数据文件:\n', num_files)
for i=1:num_files
    fprintf('%d. %s\n', i, file_list(i).name)
end

% 初始化结果存储结构
all_results=struct()

% 定义模型列表
models={
    'Random Forest',          'rf'
    'Gradient Boosting',      'gb' 
    'Extra Trees',            'et'
    'Support Vector Machine', 'svm'
    'Logistic Regression',    'lr'
    'K-Nearest Neighbor',     'knn'
    'Naive Bayes',            'nb'
}

metrics={'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'}

% 循环处理每个数据文件
for file_idx=1:num_files
    filename=fullfile(data_folder, file_list(file_idx).name)
    fprintf('\n\n=== 处理文件 %d/%d: %s ===\n', file_idx, num_files, file_list(file_idx).name)
    
    try
        % 读取数据
        numericData=readmatrix(filename, 'NumHeaderLines', 1)
        fprintf('数据维度: %d行 × %d列\n', size(numericData, 1), size(numericData, 2))
        
        % 分离特征和标签
        X=numericData(:, 1:end-1)
        y=numericData(:, end)
        
        % 数据预处理
        % 1. 处理缺失值
        X(isnan(X))=0
        
        % 2. 移除方差为零的特征
        zero_var_features=var(X)==0
        if any(zero_var_features)
            fprintf('移除 %d 个常数特征\n', sum(zero_var_features))
            X(:, zero_var_features)=[]
        end
        
        % 3. 检查数据有效性
        if isempty(X)
            fprintf('警告: 文件 %s 所有特征都是常数，跳过处理\n', file_list(file_idx).name)
            continue
        end
        
        % 4. 标准化特征
        X=zscore(X)
        
        % 5. 处理标签
        unique_labels=unique(y)
        if length(unique_labels) > 10
            % 如果唯一值太多，转换为二分类
            median_val=median(y)
            y_binary=y > median_val
            y=categorical(y_binary)
            fprintf('将回归问题转换为二分类问题，阈值: %.2f\n', median_val)
        else
            y=categorical(y)
        end
        
        final_categories=categories(y)
        fprintf('类别数量: %d\n', length(final_categories))
        
        if length(final_categories) < 2
            fprintf('警告: 文件 %s 处理后仍然只有一个类别，跳过处理\n', file_list(file_idx).name)
            continue
        end
        
        % 划分训练集和测试集
        rng(42)
        cv=cvpartition(y, 'HoldOut', 0.3)
        idxTrain=training(cv)
        idxTest=test(cv)
        
        XTrain=X(idxTrain, :)
        yTrain=y(idxTrain)
        XTest=X(idxTest, :)
        yTest=y(idxTest)
        
        fprintf('训练集大小: %d, 测试集大小: %d\n', length(yTrain), length(yTest))
        
        % 为当前文件初始化结果矩阵
        file_results=zeros(length(models), length(metrics))
        
        % 训练和评估每个模型
        for model_idx=1:length(models)
            fprintf('  训练模型: %s...\n', models{model_idx, 1})
            
            try
                switch models{model_idx, 2}
                    case 'rf'
                        model=TreeBagger(50, XTrain, yTrain, 'Method', 'classification')
                        yPred=predict(model, XTest)
                        yPred=categorical(yPred)
                        [~, scores_all]=predict(model, XTest)
                        if iscell(scores_all)
                            scores=zeros(size(scores_all, 1), length(final_categories))
                            for j=1:size(scores_all, 1)
                                scores(j, :)=scores_all{j}
                            end
                        else
                            scores=scores_all
                        end
                        
                    case 'gb'
                        if length(final_categories)==2
                            model=fitcensemble(XTrain, yTrain, 'Method', 'AdaBoostM1', ...
                                               'NumLearningCycles', 100)
                        else
                            model=fitcensemble(XTrain, yTrain, 'Method', 'GentleBoost', ...
                                               'NumLearningCycles', 100)
                        end
                        yPred=predict(model, XTest)
                        [~, scores]=predict(model, XTest)
                        
                    case 'et'
                        model=TreeBagger(50, XTrain, yTrain, 'Method', 'classification', ...
                                         'NumPredictorsToSample', size(XTrain, 2))
                        yPred=predict(model, XTest)
                        yPred=categorical(yPred)
                        [~, scores_all]=predict(model, XTest)
                        if iscell(scores_all)
                            scores=zeros(size(scores_all, 1), length(final_categories))
                            for j=1:size(scores_all, 1)
                                scores(j, :)=scores_all{j}
                            end
                        else
                            scores=scores_all
                        end
                        
                    case 'svm'
                        if length(final_categories)==2
                            model=fitcsvm(XTrain, yTrain, 'Standardize', false, ...
                                          'KernelFunction', 'linear')
                            yPred=predict(model, XTest)
                            [~, scores]=predict(model, XTest)
                        else
                            template=templateSVM('Standardize', false, 'KernelFunction', 'linear')
                            model=fitcecoc(XTrain, yTrain, 'Learners', template)
                            yPred=predict(model, XTest)
                            [~, scores]=predict(model, XTest)
                        end
                        
                    case 'lr'
                        if length(final_categories)==2
                            model=fitclinear(XTrain, yTrain, 'Learner', 'logistic')
                            yPred=predict(model, XTest)
                            [~, scores]=predict(model, XTest)
                        else
                            template=templateLinear('Learner', 'logistic')
                            model=fitcecoc(XTrain, yTrain, 'Learners', template)
                            yPred=predict(model, XTest)
                            [~, scores]=predict(model, XTest)
                        end
                        
                    case 'knn'
                        model=fitcknn(XTrain, yTrain, 'NumNeighbors', min(5, length(yTrain)-1))
                        yPred=predict(model, XTest)
                        [~, scores]=predict(model, XTest)
                        
                    case 'nb'
                        model=fitcnb(XTrain, yTrain, 'DistributionNames', 'kernel')
                        yPred=predict(model, XTest)
                        [~, scores]=predict(model, XTest)
                end
                
                % 确保预测结果格式正确
                if ~iscategorical(yPred)
                    yPred=categorical(yPred)
                end
                
                % 计算指标
                cm=confusionmat(yTest, yPred)
                accuracy=sum(diag(cm)) / sum(cm(:))
                
                % 多分类指标计算
                nClasses=size(cm, 1)
                precision=zeros(nClasses, 1)
                recall=zeros(nClasses, 1)
                f1=zeros(nClasses, 1)
                
                for c=1:nClasses
                    tp=cm(c, c)
                    fp=sum(cm(:, c)) - tp
                    fn=sum(cm(c, :)) - tp
                    
                    precision(c)=tp / (tp + fp + eps)
                    recall(c)=tp / (tp + fn + eps)
                    f1(c)=2 * (precision(c) * recall(c)) / (precision(c) + recall(c) + eps)
                end
                
                % AUC计算
                if length(final_categories)==2
                    if size(scores, 2)==2
                        pos_class_scores=scores(:, 2)
                    else
                        pos_class_scores=scores(:, 1)
                    end
                    [~, ~, ~, auc]=perfcurve(double(yTest), pos_class_scores, 1)
                else
                    auc_sum=0
                    for c=1:length(final_categories)
                        binary_yTest=double(yTest==final_categories(c))
                        if size(scores, 2) >= c
                            [~, ~, ~, auc_c]=perfcurve(binary_yTest, scores(:, c), 1)
                            auc_sum=auc_sum + auc_c
                        end
                    end
                    auc=auc_sum / length(final_categories)
                end
                
                % 存储结果
                file_results(model_idx, 1)=accuracy
                file_results(model_idx, 2)=mean(precision)
                file_results(model_idx, 3)=mean(recall)
                file_results(model_idx, 4)=mean(f1)
                file_results(model_idx, 5)=auc
                
            catch model_error
                fprintf('     模型 %s 失败: %s\n', models{model_idx, 1}, model_error.message)
                file_results(model_idx, :)=NaN
            end
        end
        
        % 存储当前文件的结果
        [~, file_name, ~]=fileparts(file_list(file_idx).name)
        file_name=matlab.lang.makeValidName(file_name) % 确保有效的字段名
        
        all_results.(file_name).results=file_results
        all_results.(file_name).models=models
        all_results.(file_name).metrics=metrics
        all_results.(file_name).data_info=sprintf('%d样本×%d特征, %d类别', ...
            size(numericData, 1), size(X, 2), length(final_categories))
        
        fprintf('文件 %s 处理完成\n', file_list(file_idx).name)
        
    catch file_error
        fprintf('处理文件 %s 时出错: %s\n', file_list(file_idx).name, file_error.message)
    end
end

% 汇总结果显示
fprintf('\n\n=== 所有数据集处理完成 ===\n')

% 获取所有成功处理的数据集名称
processed_files=fieldnames(all_results)
num_processed=length(processed_files)

if num_processed==0
    fprintf('没有成功处理任何数据集\n')
    return
end

fprintf('成功处理了 %d 个数据集:\n', num_processed)
for i=1:num_processed
    fprintf('%d. %s (%s)\n', i, processed_files{i}, all_results.(processed_files{i}).data_info)
end

% 创建汇总表格
summary_tables=cell(num_processed, 1)

for i=1:num_processed
    file_name=processed_files{i}
    results=all_results.(file_name).results
    models=all_results.(file_name).models
    metrics=all_results.(file_name).metrics
    
    % 创建当前数据集的表格
    valid_idx=~any(isnan(results), 2)
    if sum(valid_idx) > 0
        summary_tables{i}=array2table(results(valid_idx, :), ...
            'VariableNames', metrics, ...
            'RowNames', models(valid_idx, 1))
        
        fprintf('\n--- %s 的结果 ---\n', file_name)
        disp(summary_tables{i})
        
        % 找出最佳模型
        [best_acc, best_idx]=max(results(valid_idx, 1))
        best_models=models(valid_idx, 1)
        fprintf('最佳模型: %s (准确率: %.4f)\n', best_models{best_idx}, best_acc)
    else
        fprintf('\n--- %s 的结果 ---\n', file_name)
        fprintf('所有模型都失败了\n')
    end
end

% 可视化比较所有数据集的结果
if num_processed > 1
    % 创建模型在所有数据集上的性能比较图
    figure('Position', [100, 100, 1400, 800])
    
    % 提取每个数据集的准确率
    accuracy_matrix=zeros(length(models), num_processed)
    for i=1:num_processed
        file_name=processed_files{i}
        results=all_results.(file_name).results
        accuracy_matrix(:, i)=results(:, 1)
    end
    
    % 设置无效值为NaN以便可视化
    accuracy_matrix(isnan(accuracy_matrix))=NaN
    
    % 绘制热力图
    subplot(2, 3, 1)
    % 创建热力图数据，将NaN替换为最小值-0.1
    heatmap_data=accuracy_matrix'
    heatmap_data(isnan(heatmap_data))=-0.1
    imagesc(heatmap_data)
    colorbar
    title('各模型在不同数据集上的准确率')
    xlabel('模型')
    ylabel('数据集')
    set(gca, 'XTick', 1:length(models), 'XTickLabel', models(:,1), 'XTickLabelRotation', 45)
    set(gca, 'YTick', 1:num_processed, 'YTickLabel', processed_files)
    
    % 绘制每个模型的平均性能
    subplot(2, 3, 2)
    mean_accuracy=mean(accuracy_matrix, 2, 'omitnan')
    bar(mean_accuracy)
    title('各模型平均准确率')
    set(gca, 'XTickLabel', models(:,1), 'XTickLabelRotation', 45)
    ylabel('平均准确率')
    grid on
    
    % 修复箱线图 - 使用更简单的方法
    subplot(2, 3, 3)
    if num_processed > 1
        % 创建一个适合箱线图的数据结构
        box_data=[]
        group_labels={}
        
        % 为每个数据集收集所有有效的准确率值
        for i=1:num_processed
            valid_accuracies=accuracy_matrix(:, i)
            valid_accuracies=valid_accuracies(~isnan(valid_accuracies))
            
            if ~isempty(valid_accuracies)
                box_data=[box_data valid_accuracies]
                % 为每个数据点创建对应的标签
                group_labels=[group_labels repmat(processed_files(i), length(valid_accuracies), 1)]
            end
        end
        
        if length(unique(group_labels)) > 1
            % 使用分组数据绘制箱线图
            boxplot(box_data, group_labels)
            title('各数据集的模型性能分布')
            set(gca, 'XTickLabelRotation', 45)
            ylabel('准确率')
            grid on
        else
            text(0.5, 0.5, '有效数据集不足，无法绘制箱线图', 'HorizontalAlignment', 'center')
            title('各数据集的模型性能分布')
        end
    else
        text(0.5, 0.5, '需要多个数据集才能绘制箱线图', 'HorizontalAlignment', 'center')
        title('各数据集的模型性能分布')
    end
    
    % 绘制最佳模型比较
    subplot(2, 3, 4)
    best_accuracies=zeros(num_processed, 1)
    for i=1:num_processed
        file_name=processed_files{i}
        results=all_results.(file_name).results
        best_accuracies(i)=max(results(:, 1), [], 'omitnan')
    end
    bar(best_accuracies)
    title('各数据集的最佳模型准确率')
    set(gca, 'XTickLabel', processed_files, 'XTickLabelRotation', 45)
    ylabel('最佳准确率')
    grid on
    
    % 绘制模型稳定性（标准差）
    subplot(2, 3, 5)
    std_accuracy=std(accuracy_matrix, 0, 2, 'omitnan')
    bar(std_accuracy)
    title('各模型在不同数据集上的稳定性（标准差）')
    set(gca, 'XTickLabel', models(:,1), 'XTickLabelRotation', 45)
    ylabel('标准差')
    grid on
    
    % 绘制成功率
    subplot(2, 3, 6)
    success_rate=sum(~isnan(accuracy_matrix), 2)' / num_processed * 100
    bar(success_rate)
    title('各模型成功运行的比例')
    set(gca, 'XTickLabel', models(:,1), 'XTickLabelRotation', 45)
    ylabel('成功率 (%)')
    grid on
    
    % 保存汇总结果到Excel文件
    try
        output_filename=fullfile(data_folder, '模型性能汇总.xlsx')
        
        % 创建汇总表
        summary_data=cell(length(models)+1, num_processed+1)
        summary_data{1,1}='模型'
        for i=1:num_processed
            summary_data{1,i+1}=processed_files{i}
        end
        
        for i=1:length(models)
            summary_data{i+1,1}=models{i,1}
            for j=1:num_processed
                if ~isnan(accuracy_matrix(i,j))
                    summary_data{i+1,j+1}=accuracy_matrix(i,j)
                else
                    summary_data{i+1,j+1}='失败'
                end
            end
        end
        
        writecell(summary_data, output_filename, 'Sheet', '汇总')
        fprintf('汇总结果已保存到: %s\n', output_filename)
    catch
        fprintf('无法保存结果到Excel文件\n')
    end
end

% === 新增部分：详细比较各个文件对应算法处理后的值 ===
fprintf('\n\n=== 开始详细比较各个文件对应算法的性能 ===\n')

if num_processed > 0
    % 创建详细比较图
    figure('Position', [100, 100, 1600, 1000])
    
    % 提取所有指标的数据
    metrics_data=cell(length(metrics), 1)
    for m=1:length(metrics)
        metric_matrix=zeros(length(models), num_processed)
        for i=1:num_processed
            file_name=processed_files{i}
            results=all_results.(file_name).results
            metric_matrix(:, i)=results(:, m)
        end
        metric_matrix(isnan(metric_matrix))=NaN
        metrics_data{m}=metric_matrix
    end
    
    % 1. 各模型在不同数据集上的性能比较（多指标）
    subplot(3, 4, 1)
    colors=lines(length(models))
    for i=1:length(models)
        plot(1:num_processed, metrics_data{1}(i, :), 'o-', 'Color', colors(i, :), ...
            'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', models{i,1})
        hold on
    end
    title('各模型在不同数据集上的准确率比较')
    xlabel('数据集')
    ylabel('准确率')
    set(gca, 'XTick', 1:num_processed, 'XTickLabel', processed_files, 'XTickLabelRotation', 45)
    legend('Location', 'bestoutside')
    grid on
    
    % 2. 各指标的平均性能比较
    subplot(3, 4, 2)
    mean_metrics=zeros(length(metrics), length(models))
    for m=1:length(metrics)
        for i=1:length(models)
            mean_metrics(m, i)=mean(metrics_data{m}(i, :), 'omitnan')
        end
    end
    bar(mean_metrics')
    title('各模型在不同指标上的平均性能')
    ylabel('平均值')
    set(gca, 'XTickLabel', models(:,1), 'XTickLabelRotation', 45)
    legend(metrics, 'Location', 'bestoutside')
    grid on
    
    % 3. 各数据集的模型排名热力图
    subplot(3, 4, 3)
    rank_matrix=zeros(size(metrics_data{1}))
    for i=1:num_processed
        [~, ranks]=sort(metrics_data{1}(:, i), 'descend')
        for j=1:length(models)
            rank_matrix(j, i)=find(ranks==j)
        end
    end
    imagesc(rank_matrix)
    colorbar
    title('各数据集上的模型排名（准确率）')
    xlabel('数据集')
    ylabel('模型')
    set(gca, 'XTick', 1:num_processed, 'XTickLabel', processed_files, 'XTickLabelRotation', 45)
    set(gca, 'YTick', 1:length(models), 'YTickLabel', models(:,1))
    
    % 4. 各模型的平均排名
    subplot(3, 4, 4)
    mean_ranks=mean(rank_matrix, 2, 'omitnan')
    bar(mean_ranks)
    title('各模型的平均排名')
    ylabel('平均排名（越小越好）')
    set(gca, 'XTickLabel', models(:,1), 'XTickLabelRotation', 45)
    grid on
    
    % 5. 各数据集的最佳模型分布
    subplot(3, 4, 5)
    best_model_count=zeros(length(models), 1)
    for i=1:num_processed
        [~, best_idx]=max(metrics_data{1}(:, i), [], 'omitnan')
        if ~isnan(best_idx)
            best_model_count(best_idx)=best_model_count(best_idx) + 1
        end
    end
    bar(best_model_count)
    title('各模型成为最佳模型的次数')
    ylabel('次数')
    set(gca, 'XTickLabel', models(:,1), 'XTickLabelRotation', 45)
    grid on
    
    % 6. 各指标的相关性分析
    subplot(3, 4, 6)
    correlation_matrix=zeros(length(metrics))
    for i=1:length(metrics)
        for j=1:length(metrics)
            valid_idx=~isnan(metrics_data{i}(:)) & ~isnan(metrics_data{j}(:))
            if sum(valid_idx) > 1
                correlation_matrix(i, j)=corr(metrics_data{i}(valid_idx), metrics_data{j}(valid_idx))
            else
                correlation_matrix(i, j)=NaN
            end
        end
    end
    imagesc(correlation_matrix)
    colorbar
    title('各指标之间的相关性')
    set(gca, 'XTick', 1:length(metrics), 'XTickLabel', metrics, 'XTickLabelRotation', 45)
    set(gca, 'YTick', 1:length(metrics), 'YTickLabel', metrics)
    
    % 7. 各数据集的难度分析（所有模型的平均性能）
    subplot(3, 4, 7)
    dataset_difficulty=zeros(num_processed, 1)
    for i=1:num_processed
        dataset_difficulty(i)=mean(metrics_data{1}(:, i), 'omitnan')
    end
    bar(dataset_difficulty)
    title('各数据集的分类难度（平均准确率）')
    ylabel('平均准确率')
    set(gca, 'XTickLabel', processed_files, 'XTickLabelRotation', 45)
    grid on
    
    % 8. 模型稳定性分析（各模型在不同数据集上的变异系数）
    subplot(3, 4, 8)
    cv_values=zeros(length(models), 1)
    for i=1:length(models)
        std_val=std(metrics_data{1}(i, :), 'omitnan')
        mean_val=mean(metrics_data{1}(i, :), 'omitnan')
        cv_values(i)=std_val / (mean_val + eps)
    end
    bar(cv_values)
    title('各模型的稳定性（变异系数）')
    ylabel('变异系数（越小越稳定）')
    set(gca, 'XTickLabel', models(:,1), 'XTickLabelRotation', 45)
    grid on
    
    % 9. 各模型的综合评分（基于排名和性能）
    subplot(3, 4, 9)
    % 计算综合评分：平均准确率 * (1 - 变异系数)
    composite_scores=zeros(length(models), 1)
    for i=1:length(models)
        mean_acc=mean(metrics_data{1}(i, :), 'omitnan')
        std_acc=std(metrics_data{1}(i, :), 'omitnan')
        cv=std_acc / (mean_acc + eps)
        composite_scores(i)=mean_acc * (1 - cv)
    end
    bar(composite_scores)
    title('各模型的综合评分')
    ylabel('综合评分')
    set(gca, 'XTickLabel', models(:,1), 'XTickLabelRotation', 45)
    grid on
    
    % 10. 各数据集的模型性能分布（小提琴图替代箱线图）
    subplot(3, 4, 10)
    if num_processed > 1
        % 使用violin plot替代箱线图
        violin_data=cell(num_processed, 1)
        for i=1:num_processed
            violin_data{i}=metrics_data{1}(:, i)
            violin_data{i}=violin_data{i}(~isnan(violin_data{i}))
        end
        
        % 简单的小提琴图实现
        positions=1:num_processed
        for i=1:num_processed
            if ~isempty(violin_data{i})
                % 计算密度估计
                [f, xi]=ksdensity(violin_data{i})
                f=f / max(f) * 0.4 % 归一化宽度
                
                % 绘制小提琴形状
                fill([positions(i)-f, positions(i)+fliplr(f)], ...
                     [xi, fliplr(xi)], 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'b')
                hold on
                
                % 绘制中位数线
                median_val=median(violin_data{i})
                plot(positions(i)+[-0.4, 0.4], [median_val, median_val], 'r-', 'LineWidth', 2)
            end
        end
        title('各数据集的模型性能分布')
        xlabel('数据集')
        ylabel('准确率')
        set(gca, 'XTick', 1:num_processed, 'XTickLabel', processed_files, 'XTickLabelRotation', 45)
        grid on
    else
        text(0.5, 0.5, '需要多个数据集', 'HorizontalAlignment', 'center')
        title('各数据集的模型性能分布')
    end
    
    % 11. 各模型的最佳适用场景分析
    subplot(3, 4, 11)
    % 分析每个模型在哪些数据集上表现最好
    performance_ratio=zeros(length(models), 1)
    for i=1:length(models)
        model_performances=metrics_data{1}(i, :)
        max_performances=max(metrics_data{1}, [], 1, 'omitnan')
        performance_ratio(i)=mean(model_performances ./ max_performances, 'omitnan')
    end
    bar(performance_ratio)
    title('各模型与最佳性能的接近程度')
    ylabel('性能比率（越接近1越好）')
    set(gca, 'XTickLabel', models(:,1), 'XTickLabelRotation', 45)
    grid on
    
    % 12. 各数据集的模型选择建议
    subplot(3, 4, 12)
    % 为每个数据集推荐最佳模型
    recommendations=cell(num_processed, 1)
    for i=1:num_processed
        [best_val, best_idx]=max(metrics_data{1}(:, i), [], 'omitnan')
        if ~isnan(best_val)
            recommendations{i}=sprintf('%s: %.3f', models{best_idx, 1}, best_val)
        else
            recommendations{i}='无有效模型'
        end
    end
    
    % 显示推荐结果
    text(0.1, 0.5, '数据集最佳模型推荐:', 'FontSize', 12, 'FontWeight', 'bold')
    for i=1:num_processed
        text(0.1, 0.5 - i*0.07, sprintf('%s: %s', processed_files{i}, recommendations{i}), ...
            'FontSize', 10, 'Interpreter', 'none')
    end
    axis off
    title('各数据集的最佳模型推荐')
    
    % 保存详细比较结果
    try
        output_filename=fullfile(data_folder, '详细模型比较结果.xlsx')
        
        % 创建详细比较表
        detailed_data=cell(length(models)+1, num_processed*length(metrics)+1)
        detailed_data{1,1}='模型'
        
        % 设置表头
        for i=1:num_processed
            for m=1:length(metrics)
                col_idx=(i-1)*length(metrics) + m + 1
                detailed_data{1, col_idx}=sprintf('%s_%s', processed_files{i}, metrics{m})
            end
        end
        
        % 填充数据
        for i=1:length(models)
            detailed_data{i+1,1}=models{i,1}
            for j=1:num_processed
                for m=1:length(metrics)
                    col_idx=(j-1)*length(metrics) + m + 1
                    if ~isnan(metrics_data{m}(i, j))
                        detailed_data{i+1, col_idx}=metrics_data{m}(i, j)
                    else
                        detailed_data{i+1, col_idx}='失败'
                    end
                end
            end
        end
        
        writecell(detailed_data, output_filename, 'Sheet', '详细比较')
        
        % 创建排名表
        rank_data=cell(length(models)+1, num_processed+1)
        rank_data{1,1}='模型'
        for i=1:num_processed
            rank_data{1,i+1}=processed_files{i}
        end
        for i=1:length(models)
            rank_data{i+1,1}=models{i,1}
            for j=1:num_processed
                rank_data{i+1,j+1}=rank_matrix(i, j)
            end
        end
        writecell(rank_data, output_filename, 'Sheet', '模型排名')
        
        fprintf('详细比较结果已保存到: %s\n', output_filename)
    catch
        fprintf('无法保存详细比较结果到Excel文件\n')
    end
    
    % 显示总体结论
    fprintf('\n=== 总体分析结论 ===\n')
    
    % 找出总体最佳模型
    [best_overall_acc, best_overall_idx]=max(mean(metrics_data{1}, 2, 'omitnan'))
    fprintf('总体最佳模型: %s (平均准确率: %.4f)\n', models{best_overall_idx,1}, best_overall_acc)
    
    % 找出最稳定的模型
    [most_stable_val, most_stable_idx]=min(cv_values)
    fprintf('最稳定模型: %s (变异系数: %.4f)\n', models{most_stable_idx,1}, most_stable_val)
    
    % 找出综合评分最高的模型
    [best_composite, best_composite_idx]=max(composite_scores)
    fprintf('综合评分最高模型: %s (评分: %.4f)\n', models{best_composite_idx,1}, best_composite)
    
    % 分析数据集难度
    [easiest_val, easiest_idx]=max(dataset_difficulty)
    [hardest_val, hardest_idx]=min(dataset_difficulty)
    fprintf('最容易的数据集: %s (平均准确率: %.4f)\n', processed_files{easiest_idx}, easiest_val)
    fprintf('最困难的数据集: %s (平均准确率: %.4f)\n', processed_files{hardest_idx}, hardest_val)
end

fprintf('\n所有处理完成!\n')
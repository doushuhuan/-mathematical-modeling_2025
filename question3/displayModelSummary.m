function displayModelSummary(model)
    % 显示模型摘要
    fprintf('\n=== 模型架构摘要 ===\n');
    fprintf('共享特征提取器: %d层\n', length(model.sharedLayers));
    fprintf('域特定特征提取器: %d层\n', length(model.domainSpecificLayers));
    fprintf('分类器: %d层\n', length(model.classifierLayers));
    fprintf('域判别器: %d层\n', length(model.domainDiscriminatorLayers));
end


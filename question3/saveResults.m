function saveResults(results, filename)
    % 保存结果到文件
    if nargin < 2
        filename = 'transfer_diagnosis_results.mat';
    end
    
    save(filename, 'results');
    fprintf('结果已保存到: %s\n', filename);
end
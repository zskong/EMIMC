function [Z_final, labels, times] = Incremental_Train4(X_list, cls_num, anchor_num, alpha, beta, delta, lambda)
% 增量学习主函数 (带艾宾浩斯遗忘曲线权重)
% 修改为: 短期只和上一个时刻交互，长期还是融合的交互
% 参数:
%   lambda: 长期记忆的遗忘速率系数 (建议范围0.3~0.7)
%   alpha: 短期记忆权重
%   beta: 长期记忆权重
%  alpha=0; 
%  beta=0;
    tic
    num_views = length(X_list);
    
    % 初始化存储
    Z_history = cell(1, num_views);
    Z_long_term = []; % 长期记忆表示
    time_decay_factors = zeros(1, num_views); % 存储各视图的时间衰减因子
    
    for t = 1:num_views
        fprintf('====== 处理视图 %d / %d ======\n', t, num_views);
        X_t = X_list{t};

        if t == 1
            % 1. 第一个视图初始化
            Z_t = process_first_view(X_t, anchor_num, 20);
            Z_history{t} = Z_t;
            Z_long_term = Z_t; % 初始化长期记忆
            time_decay_factors(t) = 1; % 最新视图权重最大
            
        else
            % 2. 增量学习阶段
            % a. 获取短期记忆(前一时刻)和长期记忆
            Z_short_term = Z_history{t-1}; % 短期记忆: 只考虑前一时刻
            
            % b. 更新长期记忆(加权融合所有历史)
            time_intervals = (t-1:-1:1); % [t-1, t-2, ..., 1]
            weights = exp(-lambda * time_intervals); % 艾宾浩斯衰减
            weights = weights / sum(weights); % 归一化
            Z_long_term = zeros(size(Z_history{1}));
            for k = 1:(t-1)
                Z_long_term = Z_long_term + weights(k) * Z_history{k};
            end
            
            % c. 优化当前视图表示 (同时考虑短期和长期记忆)
            [Z_t] = process_subsequent_view_modified(X_t, Z_short_term, Z_long_term, anchor_num, alpha, beta, delta, 20);
            
            % d. 更新存储
            Z_history{t} = Z_t;
            time_decay_factors(t) = 1; % 当前视图权重设为1
        end
    end
    
    times = toc;
    
    % 3. 最终表示生成（加权融合所有视图）
    time_steps = num_views:-1:1;  % 从当前视图到最早视图的时间步
    %final_weights = exp(-lambda * sqrt(time_steps));  % 使用平方根衰减
    %final_weights = final_weights / sum(final_weights);  % 归一化
    final_weights = 1./(time_steps.^lambda);  % 使用幂律衰减 (t^{-λ})
    final_weights = final_weights / sum(final_weights);  % 归一化
    
    Z_final = zeros(size(Z_history{1}));
    for k = 1:num_views
        Z_final = Z_final + final_weights(k) * Z_history{k};
    end

    % 聚类
    fprintf('====== 对最终表示进行聚类 ======\n');
    [U] = mySVD(Z_final', cls_num);
    rng(5489, 'twister');
    labels = litekmeans(U, cls_num, 'MaxIter', 100, 'Replicates', 10);
end
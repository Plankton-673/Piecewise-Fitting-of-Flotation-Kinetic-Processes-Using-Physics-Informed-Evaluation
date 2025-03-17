clc
clear all
time = [0 0.50 1.00 2.00 3.00 5.00 ];

% data0 = [0 9.99 15.58 24.26 30.13 37.19 
%     0 13.86 23.69 34.89 41.04 48.83 
%     0 18.06 25.71 36.00 43.52 51.68 
%     0 20.67 27.80 38.35 46.02 54.35 
%     0 29.66 38.54 47.39 53.73 61.74 ];

data = [0	22.31 	33.55 	47.27 	53.91 	58.85 
0	28.25 	41.30 	55.29 	61.94 	67.85 
0	32.83 	46.10 	60.40 	66.36 	72.59 
0	35.69 	50.44 	63.86 	70.44 	76.32 
0	43.22 	55.62 	67.92 	73.45 	79.22 ];


n_groups = size(data,1);

% 微分方程模型，p为参数向量[A, n]
ode_model = @(t, y, p) (p(1) - y)^p(2);
% 目标函数，计算残差
objective_function1 = @(p, t, y) dealModel0(t, y, p, ode_model);
objective_function2 = @(p, params,t, y) dealModel1(t, y, p,params, ode_model);


options = optimoptions('lsqnonlin', ...
    'Display', 'iter', ...           % 显示每次迭代的结果
    'Algorithm',  'levenberg-marquardt', ...  % 使用信赖区域反射算法
    'TolFun', 1e-12, ...              % 函数容忍度，降低以提高精度
    'TolX', 1e-12, ...                % X的变化容忍度，降低以提高精度
    'MaxIter', 100);                % 增加最大迭代次数以允许更多的迭代
fitted_params = zeros(n_groups, 4);  % 存放两段的拟合参数 [A1, n1, A2, n2]

% 对每组数据进行分段拟合
for i = 1:n_groups
    % 第1到第3个点进行拟合
    t_group_1 = time(1:3);
    y_group_1 = data(i,1:3);
    initial_guess_1 = [max(y_group_1), 1];  % A1 和 n1 的初始猜测
    [params_1, ~] = lsqnonlin(@(p) objective_function1(p, t_group_1, y_group_1), initial_guess_1, [], [], options);

    % 第3到第6个点进行拟合
    t_group_2 = time(3:6);
    y_group_2 = data(i,3:6);
    initial_guess_2 = [params_1(1), 1];  % 使用第一段的A1作为A2的初始猜测，n2的初始猜测为1
    
    [params_2, ~] = lsqnonlin(@(p) objective_function2(p, params_1 ,t_group_2, y_group_2), initial_guess_2, [], [], options);

    % 存储参数
    fitted_params(i, :) = [params_1, params_2];
    i
    % pause()

end

% 显示拟合结果
disp('Fitted parameters for each group:');
disp(fitted_params); 



figure;
% for i = 1:length(random_indices)
for i = 1:n_groups
    idx = i;
    % 第一段数据
    t_group_1 = time(1:3);
    y_group_1 = data(i,1:3);
    A1 = fitted_params(idx, 1);
    n1 = fitted_params(idx, 2);
    
    % 第二段数据
    t_group_2 = time(3:6);
    y_group_2 = data(i,3:6);
    A2 = fitted_params(idx, 3);
    n2 = fitted_params(idx, 4);
    
    % 计算拟合曲线
    
    t_fine_1 = linspace(t_group_1(1), t_group_1(end), 100);
    [T_fit_1, Y_fit_1] = ode45(@(t, y) ode_model(t, y, [A1, n1]), t_fine_1, y_group_1(1));
    
    %t_fine_2 = linspace(t_group_2(1), t_group_2(end), 100);
    t_fine_2 = linspace(t_group_2(1), 10, 100);
    [T_fit_2, Y_fit_2] = ode45(@(t, y) ode_model(t, y, [A2, n2]), t_fine_2, y_group_2(1));
    % 绘图
    ylim([0, 90])
    hold on;
    plot(t_group_1, y_group_1, 'ro', 'MarkerSize', 6); % 第一段原始数据点
    plot(T_fit_1, Y_fit_1, 'b-', 'LineWidth', 1.5);    % 第一段拟合曲线
    plot(t_group_2, y_group_2, 'go', 'MarkerSize', 6); % 第二段原始数据点
    plot(T_fit_2, Y_fit_2, 'm-', 'LineWidth', 1.5);    % 第二段拟合曲线
    hold off;
    xlabel('Time (t)');
    ylabel('y');
    hold on
    title('Performance of piecewise fitting with ODE estimation');
end
    hold off




function residuals = dealModel0(t, y, p, ode_model)
    % 初始条件
    y0 = y(1);
    
    % 求解微分方程
    [T, Y] = ode45(@(t, y) ode_model(t, y, p), t, y0);
    
    % 插值求解结果以匹配数据点
    y_fit = interp1(T, Y, t);
    
    % 计算残差
    % residuals = [(y_fit(1) - y(1));(y_fit(2) - y(2));(y_fit(3) - y(3));(y_fit(4) - y(4))];
    residuals = [(y_fit(1) - y(1));(y_fit(2) - y(2));(y_fit(3) - y(3))];

end



function res = dealModel1(t, y, p, params ,ode_model)
    % 初始条件
    y0 = y(1);
    
    % 求解微分方程
    [T, Y] = ode45(@(t, y) ode_model(t, y, p), t, y0);
    
    % 插值求解结果以匹配数据点
    y_fit = interp1(T, Y, t);
    
    res1 = [(y_fit(1) - y(1));(y_fit(2) - y(2));(y_fit(3) - y(3));2*(y_fit(4) - y(4))];
    res2 =  2*((p(1)-y(1))^p(2)-(params(1)-y(1))^params(2));
    % 计算残差
    res = [res1;res2];
end



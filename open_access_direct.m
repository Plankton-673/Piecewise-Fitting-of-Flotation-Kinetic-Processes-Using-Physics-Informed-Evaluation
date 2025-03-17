clc
clear all
time = [0 0.50 1.00 2.00 3.00 5.00 ];


data = [0	22.31 	33.55 	47.27 	53.91 	58.85 
0	28.25 	41.30 	55.29 	61.94 	67.85 
0	32.83 	46.10 	60.40 	66.36 	72.59 
0	35.69 	50.44 	63.86 	70.44 	76.32 
0	43.22 	55.62 	67.92 	73.45 	79.22 ];

n_groups = size(data,1);

% 微分方程模型，p为参数向量[A, n]
ode_model = @(t, y, p) (p(1) - y)^p(2);

% 目标函数，计算残差
objective_function1 = @(p, t, y) dealModel1(t, y, p, ode_model);

options = optimoptions('lsqnonlin', ...
    'Display', 'iter', ...           % 显示每次迭代的结果
    'Algorithm',  'levenberg-marquardt', ...  % 使用信赖区域反射算法
    'TolFun', 1e-12, ...              % 函数容忍度，降低以提高精度
    'TolX', 1e-12, ...                % X的变化容忍度，降低以提高精度
    'MaxIter', 100);                % 增加最大迭代次数以允许更多的迭代

fitted_params = zeros(n_groups, 2);  % 存放两段的拟合参数 [A1, n1, A2, n2]


% 对每组数据进行分段拟合
for i = 1:n_groups
    
    t_group_1 = time(1:6);
    y_group_1 = data(i,1:6);
    initial_guess_1 = [max(y_group_1), 1];  % A1 和 n1 的初始猜测
    [params_1, ~] = lsqnonlin(@(p) objective_function1(p, t_group_1, y_group_1), initial_guess_1, [], [], options);
    fitted_params(i, :) = params_1;
    i
    % pause()
    A1 = params_1(1);
    n1 = params_1(2);
    [T1, Y1] = ode45(@(t, y) ode_model(t, y, [A1, n1]),  t_group_1,  y_group_1(1));
end

% 显示拟合结果
disp('Fitted parameters for each group:');
disp(fitted_params); 

figure;
for i=1:n_groups
    t_group_1 = time(1:6);
    y_group_1 = data(i,1:6);
    A1 = fitted_params(i, 1);
    n1 = fitted_params(i, 2);

    % 计算拟合曲线
    t_fine_1 = linspace(t_group_1(1), 10, 1000);
    [T_fit_1, Y_fit_1] = ode45(@(t, y) ode_model(t, y, [A1, n1]), t_fine_1, y_group_1(1));
   
    % 绘图
    ylim([0, 90])
    plot(t_group_1, y_group_1, 'ro', 'MarkerSize', 6); % 原始数据点
    hold on
    plot(T_fit_1, Y_fit_1, 'b-', 'LineWidth', 1.5);    % 拟合曲线
    xlabel('Time (t)');
    ylabel('y');
    title('Performance of single segment direct fitting');
end
hold off

function residuals = dealModel1(t, y, p, ode_model)
    % 初始条件
    y0 = y(1);
    
    % 求解微分方程
    [T, Y] = ode45(@(t, y) ode_model(t, y, p), t, y0);
    
    % 插值求解结果以匹配数据点
    y_fit = interp1(T, Y, t);
    
    % 计算残差
    residuals = y_fit - y;
end












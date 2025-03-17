clc
clear all
time = [0 0.50 1.00 2.00 3.00 5.00 ];



data = [0	22.31 	33.55 	47.27 	53.91 	58.85 
0	28.25 	41.30 	55.29 	61.94 	67.85 
0	32.83 	46.10 	60.40 	66.36 	72.59 
0	35.69 	50.44 	63.86 	70.44 	76.32 
0	43.22 	55.62 	67.92 	73.45 	79.22 ];


n_groups = size(data,1);

% Differential equation model, p is parameter vector [A, n]
ode_model = @(t, y, p) (p(1) - y)^p(2);
% Objective function to calculate residuals
objective_function1 = @(p, t, y) dealModel0(t, y, p, ode_model);
objective_function2 = @(p, params,t, y) dealModel1(t, y, p,params, ode_model);


options = optimoptions('lsqnonlin', ...
    'Display', 'iter', ...           % Show iteration results
    'Algorithm',  'levenberg-marquardt', ...  % Use Levenberg-Marquardt algorithm
    'TolFun', 1e-12, ...              % Lower tolerance to improve accuracy
    'TolX', 1e-12, ...                % Lower tolerance to improve accuracy
    'MaxIter', 100);                % Increase maximum iterations
fitted_params = zeros(n_groups, 4);  % Store fitted parameters for two segments [A1, n1, A2, n2]

% Perform piecewise fitting for each group of data
for i = 1:n_groups
    % Fit first 3 points
    t_group_1 = time(1:3);
    y_group_1 = data(i,1:3);
    initial_guess_1 = [max(y_group_1), 1];  % Initial guess for A1 and n1
    [params_1, ~] = lsqnonlin(@(p) objective_function1(p, t_group_1, y_group_1), initial_guess_1, [], [], options);

    % Fit points 3-6
    t_group_2 = time(3:6);
    y_group_2 = data(i,3:6);
    initial_guess_2 = [params_1(1), 1];  % Use A1 from first segment as initial guess for A2
    
    [params_2, ~] = lsqnonlin(@(p) objective_function2(p, params_1 ,t_group_2, y_group_2), initial_guess_2, [], [], options);

    % Store parameters
    fitted_params(i, :) = [params_1, params_2];
    i
    % pause()

end

% Display fitting results
disp('Fitted parameters for each group:');
disp(fitted_params); 



figure;
% for i = 1:length(random_indices)
for i = 1:n_groups
    idx = i;
    % First segment data
    t_group_1 = time(1:3);
    y_group_1 = data(i,1:3);
    A1 = fitted_params(idx, 1);
    n1 = fitted_params(idx, 2);
    
    % Second segment data
    t_group_2 = time(3:6);
    y_group_2 = data(i,3:6);
    A2 = fitted_params(idx, 3);
    n2 = fitted_params(idx, 4);
    
    % Calculate fitted curves
    
    t_fine_1 = linspace(t_group_1(1), t_group_1(end), 100);
    [T_fit_1, Y_fit_1] = ode45(@(t, y) ode_model(t, y, [A1, n1]), t_fine_1, y_group_1(1));
    
    %t_fine_2 = linspace(t_group_2(1), t_group_2(end), 100);
    t_fine_2 = linspace(t_group_2(1), 10, 100);
    [T_fit_2, Y_fit_2] = ode45(@(t, y) ode_model(t, y, [A2, n2]), t_fine_2, y_group_2(1));
    % Plot results
    ylim([0, 90])
    hold on;
    plot(t_group_1, y_group_1, 'ro', 'MarkerSize', 6); % First segment original data
    plot(T_fit_1, Y_fit_1, 'b-', 'LineWidth', 1.5);    % First segment fitted curve
    plot(t_group_2, y_group_2, 'go', 'MarkerSize', 6); % Second segment original data
    plot(T_fit_2, Y_fit_2, 'm-', 'LineWidth', 1.5);    % Second segment fitted curve
    hold off;
    xlabel('Time (t)');
    ylabel('y');
    hold on
    title('Piecewise fitting with ODE estimation performance');
end
    hold off



function residuals = dealModel0(t, y, p, ode_model)
    % Initial conditions
    y0 = y(1);
    
    % Solve differential equation
    [T, Y] = ode45(@(t, y) ode_model(t, y, p), t, y0);
    
    % Interpolate to match data points
    y_fit = interp1(T, Y, t);
    
    % Calculate residuals
    % residuals = [(y_fit(1) - y(1));(y_fit(2) - y(2));(y_fit(3) - y(3));(y_fit(4) - y(4))];
    residuals = [(y_fit(1) - y(1));(y_fit(2) - y(2));(y_fit(3) - y(3))];

end



function res = dealModel1(t, y, p, params ,ode_model)
    % Initial conditions
    y0 = y(1);
    
    % Solve differential equation
    [T, Y] = ode45(@(t, y) ode_model(t, y, p), t, y0);
    
    % Interpolate to match data points
    y_fit = interp1(T, Y, t);
    
    res1 = [(y_fit(1) - y(1));(y_fit(2) - y(2));(y_fit(3) - y(3));2*(y_fit(4) - y(4))];
    res2 =  2*((p(1)-y(1))^p(2)-(params(1)-y(1))^params(2));
    % Calculate residuals
    res = [res1;res2];
end

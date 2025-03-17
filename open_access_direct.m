clc
clear all
time = [0 0.50 1.00 2.00 3.00 5.00 ];


data = [0	22.31 	33.55 	47.27 	53.91 	58.85 
0	28.25 	41.30 	55.29 	61.94 	67.85 
0	32.83 	46.10 	60.40 	66.36 	72.59 
0	35.69 	50.44 	63.86 	70.44 	76.32 
0	43.22 	55.62 	67.92 	73.45 	79.22 ];

n_groups = size(data,1);

% Differential equation model, p is the parameter vector [A, n]
ode_model = @(t, y, p) (p(1) - y)^p(2);

% Objective function to calculate residuals
objective_function1 = @(p, t, y) dealModel1(t, y, p, ode_model);

options = optimoptions('lsqnonlin', ...
    'Display', 'iter', ...           % Show iteration results
    'Algorithm',  'levenberg-marquardt', ...  % Use Levenberg-Marquardt algorithm
    'TolFun', 1e-12, ...              % Lower to improve accuracy
    'TolX', 1e-12, ...                % Lower to improve accuracy
    'MaxIter', 100);                % Increase maximum iterations

fitted_params = zeros(n_groups, 2);  % Stores fitted parameters [A, n] for each group


% Fit each group of data
for i = 1:n_groups
    
    t_group_1 = time(1:6);
    y_group_1 = data(i,1:6);
    initial_guess_1 = [max(y_group_1), 1];  % Initial guess for A and n
    [params_1, ~] = lsqnonlin(@(p) objective_function1(p, t_group_1, y_group_1), initial_guess_1, [], [], options);
    fitted_params(i, :) = params_1;
    i
    % pause()
    A1 = params_1(1);
    n1 = params_1(2);
    [T1, Y1] = ode45(@(t, y) ode_model(t, y, [A1, n1]),  t_group_1,  y_group_1(1));
end

% Display fitted parameters
disp('Fitted parameters for each group:');
disp(fitted_params); 

figure;
for i=1:n_groups
    t_group_1 = time(1:6);
    y_group_1 = data(i,1:6);
    A1 = fitted_params(i, 1);
    n1 = fitted_params(i, 2);

    % Calculate fitted curve
    t_fine_1 = linspace(t_group_1(1), 10, 1000);
    [T_fit_1, Y_fit_1] = ode45(@(t, y) ode_model(t, y, [A1, n1]), t_fine_1, y_group_1(1));
   
    % Plot results
    ylim([0, 90])
    plot(t_group_1, y_group_1, 'ro', 'MarkerSize', 6); % Original data points
    hold on
    plot(T_fit_1, Y_fit_1, 'b-', 'LineWidth', 1.5);    % Fitted curve
    xlabel('Time (t)');
    ylabel('y');
    title('Single segment direct fitting performance');
end
hold off

function residuals = dealModel1(t, y, p, ode_model)
    % Initial conditions
    y0 = y(1);
    
    % Solve the differential equation
    [T, Y] = ode45(@(t, y) ode_model(t, y, p), t, y0);
    
    % Interpolate solution to match data points
    y_fit = interp1(T, Y, t);
    
    % Calculate residuals
    residuals = y_fit - y;
end

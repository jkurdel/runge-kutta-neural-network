% Function solving Van der Pol equations usign 4-th order Runge-Kutta method
%
% t_end - end of time period for solution (time starts from 0)
% h     - time step
% y0    - init values

function [t,y] = rk_van_der_pol(t_end, h, y0)

    % Runge-Kutta general formula
    % y_{i+1} = y_i + 1/6*(k_1 + 2k_2 + 2k_3 + k_4), i = 0,1,2,...
    % k_1 = hf(x_i,y_1)
    % k_2 = hf(x_i + 1/2h, y_1 + 1/2k_1)
    % k_3 = hf(x_i + 1/2h, y_1 + 1/2k_2)    
    % k_4 = hf(x_i + h, y_1 + k_3)
    
    t = 0:h:t_end;
    n = length(t);
    h = 0.1;

    y = zeros(n, 2);
    y(1, 1) = y0(1);
    y(1, 2) = y0(2);
    
    % Van der Pol equation 
    % y''(t) - (1 - y^2(t)) * y'(t) + y(t) = 0;
    % y'(t)  = y(2)
    % y''(t) = (1-y(1)^2)*y(2)-y(1)

    for i = 1:n-1
        k1 = h*(y(i,2));
        k2 = h*(y(i,2)+0.5*k1);
        k3 = h*(y(i,2)+0.5*k2);
        k4 = h*(y(i,2)+k3);
        y(i+1,1) = y(i,1)+(k1+2*k2+2*k3+k4)/6;
        
        y1 = y(i,1);
        y2 = y(i,2);
        k1 = h*((1-y1^2)*y2-y1);
        y1 = y(i,1)+0.5*k1;
        y2 = y(i,2)+0.5*k1;
        k2 = h*((1-y1^2)*y2-y1);
        y1 = y(i,1)+0.5*k2;
        y2 = y(i,2)+0.5*k2;
        k3 = h*((1-y1^2)*y2-y1);
        y1 = y(i,1)+k3;
        y2 = y(i,2)+k3;
        k4 = h*((1-y1^2)*y2-y1);
        y(i+1,2) = y(i,2)+(k1+2*k2+2*k3+k4)/6;

    end
    
end

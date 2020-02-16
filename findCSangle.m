function [ CS_angle ] = findCSangle( length ,Ex )

% findCSangle : find the polar scattering angle of compton scattering by
% solving the cumulative distribution function (CDF) via Bisection method

% Input:
% Ex: x-ray energy (in keV)
% length: number of polar angles needed to be found
% Output:
% CS_angle : polar scattering angle

% test
% length = 5;
% Ex = 0.3;

TOL = 10^(-6); % set tolarance
N0 = 1000; % set maximum number of iteration
h = 6.626 * 10^(-34); % plank constant
c = 3*10^8; % speed of light
Ex = Ex * 1.6022*10^(-16); % convert to Joule
lambda = h*c/Ex; % wavelength of incident x-ray


% CDF:
f = @(x) (lambda^2./(1+lambda-cos(x)).^2 .* (lambda./(1+lambda-cos(x)) + (1+lambda-cos(x))./lambda - sin(x).^2)) .* sin(x);

r = rand(length,1);
FA = r;
FP = zeros(length,1);
p = zeros(length,1);

for j = 1:length
    
    i = 1;
    a = 0;
    b = pi;
    
    while i<=N0
        p(j) = a + (b-a)/2; % find initial trial value
        f_1 = integral (f,0,p(j));
        f_2 = integral (f,0,pi);
        C = f_1/f_2;
        FP(j) = r(j)-C;
        
        % check tolarance
        if abs(FP) <= TOL %|| (b-a)/2 <= TOL
            break
        end
        
        i=i+1;
        
        if FP(j)*FA(j) > 0
            a = p(j);
            FA(j) = FP(j);
        else
            b = p(j);
        end
    end
    
    
end

CS_angle = p;

end












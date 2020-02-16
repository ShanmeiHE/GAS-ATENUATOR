function y = rayl( n,sigma )
%to produce n random number following Rayleigh distribution
%   此处显示详细说明
randnum=rand(n,1);
y=2^0.5*sigma*(-log(1-randnum)).^0.5;
end


function x = random_number( N,I0 ) %procuce N random number of 0-1,the beginning number of I needs to be input
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
a=16807;m=2147483647;q=floor(m/a);r=mod(m,a); %floor is to get int,mod is to get the remainder
I=zeros(N,1);
x=zeros(N,1);
I(1)=I0;
x(1)=I0/m;
for i=2:N
    I(i)=mod_y(a,I(i-1),m);
		x(i)=I(i)/m;
end
end


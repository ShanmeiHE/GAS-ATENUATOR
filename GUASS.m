function [pp] = GUASS(delta_r,laser_r,sum_cell)

sigma = laser_r/3;
p = [];
fun = @(x) x.*exp(-x.^2/(2*sigma^2));
parfor i = 1:round(laser_r/delta_r)
    p(i) = integral(fun,(i-1)*delta_r,i*delta_r);
end
sum_cell = [0,sum_cell];

for j = 1:length(p)
    pp(sum_cell(j)+1:sum_cell(j+1)) = p(j).*ones(sum_cell(j+1)-sum_cell(j),1)/(sum_cell(j+1)-sum_cell(j));
end

pp = pp'/sum(pp);




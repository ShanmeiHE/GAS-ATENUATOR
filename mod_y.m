function remainder =mod_y (a,z,m)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
q=floor(m/a);r=mod(m,a);
s=a*mod(z,q)-r*floor(z/q);
	if(s>=0) 
        remainder=s;
	else 
        remainder=s+m;
    end

end


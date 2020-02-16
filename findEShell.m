function [E_Bind, shells ,E_Shell ] = findEShell (m_a,Ex)
E_Bind = csvread (['EB_',num2str(m_a),'.csv']);
E_Bind = E_Bind./1000;%change of unit from eV to keV
shells = length (E_Bind);
for i = 1:shells
    if Ex >= E_Bind(i)
        E_Shell = E_Bind(i);
        break
    else
        E_Shell = 0;
    end
end
end

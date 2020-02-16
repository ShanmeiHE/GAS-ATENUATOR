function [mol_rho] = molecule_rho(W,INDEX,VC,cellnumxy,z_divide)
cellnum = round(cellnumxy*z_divide);
mol_rho = [];
parfor i = 1:cellnum
    mol_rho(i) = length(find(INDEX == i))*W/VC(i);
end
mol_rho = mol_rho';
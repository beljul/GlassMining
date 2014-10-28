run import.m % Imports the data to struct data

% This script will do a PCA on the data and then plot the 1st and 2nd
% component against each other, grouping the glass by type

% Fetch the concentration names
names = fieldnames(data);
names = names(2:9);
% Enumerate the types
types = {'building windows, float processed'; 'building windows, non float processed'; 'vehicle windows, float processed'; 'containers'; 'tableware'; 'headlamps'};
% Do PCA of data.all
% Minus the mean of data
mu = mean(data.all(:,2:9));
concentrations = bsxfun(@minus, data.all(:,2:9), mu);

% SVD compute
[U,S,V] = svd(concentrations);
s = diag(S);

% Pourcentage of implication of each attribute
rho = (s.^2)/(sum(s.^2));

% Plot variance explained
mfig('Glass: Var. explained');clf;
plot(rho, 'o-');
title('Variance explained by principal components');
xlabel('Principal component');
ylabel('Variance explained value');

% Compute the projection onto the principal components
Z = U*S;

% Principal components to be plotted
j = 1;
k = 2;

figure;
hold all;
opt={'r' 'g' 'b' };
for i=1:7
    index = find(data.type == i);
    plot(Z(index, j), Z(index, k), 'o');
end
legend(types)
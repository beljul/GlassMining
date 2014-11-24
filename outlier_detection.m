%% Outlier detection
import
attributeNames = fieldnames(data);
attributeNames = attributeNames(2:10);
M = 9;
X = data.all(:, 1:9);

%% Normalize the data
scaledX = (X-min(X(:))) ./ (max(X(:)-min(X(:))));

%% Gaussian Kernel density
% Kernel width
w = 1;

% Estimate optimal kernel density width by leave-one-out cross-validation
widths=2.^[0:1];
for w=1:length(widths)
    [f, log_f] = gausKernelDensity(scaledX, widths(w));
    logP(w)=sum(log_f);
end
[val,ind]=max(logP);
width=widths(ind);
disp(['Optimal estimated width is ' num2str(width)])

% Estimate density for each observation not including the observation
% itself in the density estimate
f = gausKernelDensity(scaledX, width);

% Sort the densities
[y,j] = sort(f);

% Display the index of the 10 lowest data object
disp(j(1:10));

% Plot density estimate outlier scores
mfig('Outlier score'); clf;
bar(y(1:20));

%% KNN density
% Neighbor to use
K = 1;

% Find the k nearest neighbors
[i, D] = knnsearch(scaledX, scaledX, 'K', K+1);

% Compute the density
density = 1./(sum(D,2)/K);

% Sort the densities
[y,j] = sort(density);

% Display the index of the 10 lowest data object
disp(j(1:10));

% Plot density estimate outlier scores
mfig('Outlier score'); clf;
bar(y(1:20));

%% KNN average relative density
% Neighbor to use
K = 1;

% Compute the average relative density
[iX,DX] = knnsearch(scaledX, scaledX, 'K', K+1);
densityX= 1./(sum(DX(:,2:end),2)/K);
avg_rel_density=density./(sum(densityX(i(:,2:end)),2)/K);

% Sort the densities
[y,j] = sort(avg_rel_density);

% Display the index of the 10 lowest data object
disp(j(1:10));

% Plot density estimate outlier scores
mfig('Outlier score'); clf;
bar(y(1:20));
%% Distance to Kth nearest neighbor
% Neighbor to use
K = 1;

% Find the k nearest neighbors
[i, D] = knnsearch(X, X, 'K', K+1);

% Outlier score
f = D(:,K+1);

% Sort the outlier scores
[y,j] = sort(f, 'descend');

% Display the index of the lowest density data object
disp(j(1:10));

% Plot kernel density estimate outlier scores
mfig('Distance: Outlier score'); clf;
bar(y(1:20));

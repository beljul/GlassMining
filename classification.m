%% Get data
attributeNames = fieldnames(data);
attributeNames = attributeNames(2:9);
classNames = {'BW-FP'; 'BW-NFP'; 'VW-FP'; 'VW-NFP'; 'C'; 'T'; 'H'};
y = data.type';
X = data.all(:,2:9);
M = 8;
C = length(classNames);
%% Classification tree 
% Number of folds for crossvalidation
K = 10;

% Create holdout crossvalidation partition
CV = cvpartition(classNames(y), 'Kfold', K);

% Pruning levels
prune = 0:10;

% Variable for classification error
Error_train = nan(K,length(prune));
Error_test = nan(K,length(prune));

for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k))';
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k))';
    
    % Fit classification tree to training set
    T = classregtree(X_train, classNames(y_train), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames, ...
        'prune', 'on', ...
        'minparent', 10);

    % Compute classification error
    for n = 1:length(prune) % For each pruning level
        Error_train(k,n) = sum(~strcmp(classNames(y_train), eval(T, X_train, prune(n))));
        Error_test(k,n) = sum(~strcmp(classNames(y_test), eval(T, X_test, prune(n))));
    end    
end
    
% Plot classification error
mfig('Glass decision tree: K-fold crossvalidatoin'); clf; hold all;
plot(prune, sum(Error_train)/sum(CV.TrainSize));
plot(prune, sum(Error_test)/sum(CV.TestSize));
xlabel('Pruning level');
ylabel('Classification error');
legend('Training error', 'Test error');

%%  With optimize parameter (Prune=5)
% Pruning levels
prune = 5;

% Variable for classification error
Error_train = nan(K,length(prune));
Error_test = nan(K,length(prune));

for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k))';
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k))';
    
    % Fit classification tree to training set
    T = classregtree(X_train, classNames(y_train), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames, ...
        'prune', 'on', ...
        'minparent', 10);

    % Compute classification error
    for n = 1:length(prune) % For each pruning level
        Error_train(k,n) = sum(~strcmp(classNames(y_train), eval(T, X_train, prune(n))));
        Error_test(k,n) = sum(~strcmp(classNames(y_test), eval(T, X_test, prune(n))));
    end    
end
    
fprintf('\n');
fprintf('Decision tree with feature selection:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train)/sum(CV.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test)/sum(CV.TestSize));

%% Plot the tree
% Plot the tree
T = classregtree(X, classNames(y), ...
    'method', 'classification', ...
    'splitcriterion', 'gdi', ...
    'categorical', [], ...
    'names', attributeNames, ...
    'prune', 'on', ...
    'minparent', 10);

view(T)
%% K-nearest neighbors

% Leave-one-out crossvalidation
CV = cvpartition(N, 'Leaveout');
K = CV.NumTestSets;

% K-nearest neighbors parameters
Distance = 'euclidean'; % Distance measure
L = 40; % Maximum number of neighbors

% Variable for classification error
Error = nan(K,L);

for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    for l = 1:L % For each number of neighbors
        
        % Use knnclassify to find the l nearest neighbors
        y_test_est = knnclassify(X_test, X_train, y_train, l, Distance);
        
        % Compute number of classification errors
        Error(k,l) = sum(y_test~=y_test_est); % Count the number of errors
    end
end

% Plot the classification error rate
mfig('Error rate');
plot(sum(Error)./sum(CV.TestSize)*100);
xlabel('Number of neighbors');
ylabel('Classification error rate (%)');

%% With the optimal parameter (K = 1|2)
% K-nearest neighbors parameters

NG = 1; % Number of neighbors
Distance = 'euclidean'; % Distance measure

% Cross-validation
% Leave-one-out crossvalidation
N = 213;
CV = cvpartition(N, 'Leaveout');
K = CV.NumTestSets;
% Variable for classification error
Error = nan(K, 1);

for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test(k) = y(CV.test(k));

    
    % Use knnclassify to find the l nearest neighbors
    y_test_est(k) = knnclassify(X_test, X_train, y_train, NG, Distance);

    % Compute number of classification errors
    Error(k) = sum(y_test(k)~=y_test_est(k)); % Count the number of errors
end

% Plot confusion matrix
mfig('Confusion matrix');
confmatplot(classNames(y_test), classNames(y_test_est));

fprintf('\n');
fprintf('K-nearest neighbors with feature selection (N=1):\n');
fprintf('- Test error:     %8.2f\n', sum(Error)/sum(CV.TestSize));
%% Artificial Neural Networks
% K-fold crossvalidation
K = 10;
N = 213;
CV = cvpartition(N,'Kfold', K);

% Parameters for neural network classifier
NHiddenUnits = 10;  % Number of hidden units
NTrain = 1; % Number of re-trains of neural network

% Variable for classification error
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
bestnet=cell(K,1);

for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k))';
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k))';
    % Fit neural network to training set
    MSEBest = inf;
    for t = 1:NTrain
        netwrk = nr_main(X_train, y_train, X_test, y_test, NHiddenUnits);
        if netwrk.mse_train(end)<MSEBest, bestnet{k} = netwrk; MSEBest=netwrk.mse_train(end); MSEBest=netwrk.mse_train(end); end
    end
    
    % Predict model on test and training data    
    y_train_est = bestnet{k}.t_pred_train;    
    y_test_est = bestnet{k}.t_pred_test;        
    
    % Compute least squares error
    Error_train(k) = sum((y_train-y_train_est).^2);
    Error_test(k) = sum((y_test-y_test_est).^2); 
        
    % Compute least squares error when predicting output to be mean of
    % training data
    Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);            
end

% Print the least squares errors
% Display results
fprintf('\n');
fprintf('Neural network regression without feature selection:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train)/sum(CV.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test)/sum(CV.TestSize));
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));

% Display the trained network 
mfig('Trained Network');
k=1; % cross-validation fold
displayNetwork(bestnet{k});

%% Naive Bayes
% K-fold crossvalidation
K = 10;
CV = cvpartition(y, 'Kfold', K);

% Parameters for naive Bayes classifier
Distribution = 'mvmn';
Prior = 'uniform';

% Variable for classification error
Error = nan(K,1);


for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k))';
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k))';

    % Fit naive Bayes classifier to training set
    NB = NaiveBayes.fit(X_train, y_train, 'Distribution', Distribution, 'Prior', Prior);
    
    % Predict model on test data    
    y_test_est = predict(NB, X_test);
    
    % Compute error rate
    Error(k) = sum(y_test~=y_test_est); % Count the number of errors
end

% Print the error rate
fprintf('Error rate: %.1f%%\n', sum(Error)./sum(CV.TestSize)*100);
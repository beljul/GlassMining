%% Association mining
import
attributeNames = fieldnames(data);
attributeNames = attributeNames(2:10);
M = 9
X = data.all(:, 1:9)
%% Binarized data
[Xbinary,attributeNamesBin]=binarize(X,[2*ones(1,M)],attributeNames);
%% Write into a file
j = 1;
for i = 1:(M*2)
    if(mod(i,2) == 0)
        Xbinaryfinal(:, j) = Xbinary(:, i);
        j = j + 1;
    end
end
writeAprioriFile(Xbinaryfinal, 'DataBinary.txt')

%% Apriori algorithm
MinSupp = 30;
MinConf = 60;
[AssocRules,FreqItemsets]=apriori('DataBinary.txt', MinSupp, MinConf)
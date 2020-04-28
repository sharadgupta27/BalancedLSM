function [boostset, boosttarget] = boost_data(trainset, traintarget,weight)
% To sample a subset according to each example's weight
% Input:
%   trainset: n-by-d training set
%   traintarget: n-by-1 training target {0,1}
%   weight: n-by-1 weight vector, with its sum is 1
% Output:
%   boostset: sampled data set
%   boosttarget: labels for boostset

% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Contact: Xu-Ying Liu (liuxy@lamda.nju.edu.cn)
% Modified by: Nicola Lazzarini, 2012
% Modified by: Sharad Kumar Gupta, 2020
% Contact: sharadgupta27@gmail.com
%-------------------------------------------------------

num_class = size(unique(traintarget)); %count number of classes
n = length(traintarget);
c_sum = cumsum(weight);
cont=0;

%determine if minimum target is 0 or 1
if min(traintarget) == 0
    sub = 1;
else
    sub = 0;
end


while (cont<100) %loop while each class is represented on boostset  
    select = rand(size(traintarget));
    for i = 1:n
        select(i) = find(c_sum>=select(i), 1 );   
    end

    boostset = trainset(select,:);
    boosttarget = traintarget(select);
    
    if size(unique(boosttarget))==num_class %check if each class is represented
        cont = 101;
    else
        cont = cont+1;
    end
end

if (cont ~= 101)
    for i = 1-sub:max(traintarget)
        if isempty(find(boosttarget==i, 1))% i-th is not represented
            idxs = find(traintarget==i);
            random_idxs = idxs(randperm(size(idxs,1)),:);%random order for trainset elements that belong to i-th class             
            freq = histcounts( traintarget, unique(traintarget) );
            [x,j] = max(freq);
            idx = find(traintarget==j-sub);%find elements for most represented class j
            random_idx = idx(randperm(size(idx,1)),:);             
            %random substitute an element of class j with an element of class i
            boostset(random_idx(1),:)= trainset(random_idxs(1),:); 
            boosttarget(random_idx(1),:)= traintarget(random_idxs(1),:);
        end   
    end
end
    
save ('bost','boostset');
save ('train','trainset');
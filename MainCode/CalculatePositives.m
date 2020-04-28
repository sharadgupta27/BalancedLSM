function rates = CalculatePositives(ensemble,test,testtarget,str_class_eval,col,modelName)
% To calculate (fpr,tpr)
% Input:
%   ensemble: AdaBoost/EasyEnsemble/BalanceCascade classifer
%   test: n-by-d test set
%   testtarget: n-by-1 test target {0,1}
%   str_class_eval   : command for classifier evaluation
% Output:
%   rates: (fpr,tpr) vector with fpr in ascending order

% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Contact: Xu-Ying Liu (liuxy@lamda.nju.edu.cn)
% Modified by: Nicola Lazzarini, 2012
% Modified by: Sharad Kumar Gupta, 2020
% Contact: sharadgupta27@gmail.com
%-------------------------------------------------------
n = length(testtarget);
values = EvaluateValue(test,testtarget,ensemble,str_class_eval,col,modelName);
vi = [values testtarget];
vi = sortrows(vi);

fp = zeros(n+1,1);
tp = zeros(n+1,1);

tpc = sum(testtarget);
fpc = n-tpc;
prev = -1;
index = 1;
for i=1:n
    if vi(i,1)~=prev
        prev = vi(i,1);
        tp(index)=tpc;
        fp(index)=fpc;
        index = index+1;
    end
    if vi(i,2)==1
        tpc = tpc - 1;
    else
        fpc = fpc - 1;
    end
end
tp(index)=0;
fp(index)=0;
tp=tp(1:index);
fp=fp(1:index);

rates = [fp tp];
rates = flipud(rates);
rates(:,1) = rates(:,1) / (length(testtarget)-sum(testtarget));
rates(:,2) = rates(:,2) / sum(testtarget);
function thresh = AdjustFPRate(dataset,target,ensemble,fpr,str_class_eval,col,modelName)
% To adjust threshold such that AdaBoost classifier's false positive rate is $fpr$
% Input:
%   dataset: n-by-d a set of negative examples
%   ensemble: AdaBoost classifier
%   fpr: the false positive rate to be achieved
%   str_class_eval   : command for classifier evaluation
%               example: [predict_target]=svmpredict(traintarget, trainset, model);%
% Output:
%   thresh: threshold of current AdaBoost classifier

% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Modified by: Nicola Lazzarini, 2012
% Modified by: Sharad Kumar Gupta, 2020
% Contact: Sharad Kumar Gupta (sharadgupta27@gmail.com)

result = EvaluateValue(dataset,target,ensemble,str_class_eval,col,modelName);
result = sort(result);
thresh = result(max([1 round(size(dataset,1)*(1-fpr))]));
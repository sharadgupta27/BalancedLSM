function [evaltarget1,w,bias] = EvaluateValue(evalset,evaltarget,ensemble,str_class_eval,col,modelName)
% To predict real values for a set of examples using 
% AdaBoost/EasyEnsemble/BalanceCascade classifer
% Input:
%   evalset: n-by-d test set
%   ensemble: AdaBoost/EasyEnsemble/BalanceCascade classifer
%   str_class_eval   : command for classifier evaluation
%               example: [predict_target]=svmpredict(traintarget, trainset, model);%

% Output:
%   evaltarget: predicted real values of test examples
%   weight and bias

% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Modified by: Nicola Lazzarini, 2012
% Modified by: Sharad Kumar Gupta, 2020
%	  Contact: sharadgupta27@gmail.com
%-------------------------------------------------------
evaltarget = evaltarget;
evaltarget1 = zeros(size(evalset,1),1);
noClassifiers = length(ensemble.classifier);
w = zeros(noClassifiers,col);
bias = zeros(noClassifiers,1);
for i=1:noClassifiers
    %test each pattern of dataset using i-th model built with Adaboost
    model = ensemble.classifier{i};
    if modelName == "SVMCode"
        w(i,:) = (model.sv_coef' * full(model.SVs));
        bias(i,1) = -model.rho;
    elseif modelName == "ANNCode"
        net = model;
        wt = net.IW{1,1}';
        for k = 2:length(net.layers)
            wt = wt * net.LW{k,k-1}';
        end
        w(i,:) = wt;
    end
    eval(str_class_eval);%this command need to use evalset and evaltarget; predicted labels MUST TO BE STORED in: predict_target
    evaltarget1 = evaltarget1 + ensemble.alpha(i) * predict_target;
end
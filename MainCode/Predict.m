function result= Predict(dataset,target,ensemble,str_class_eval,col,modelName)
% To predict labels for a set of examples using 
% AdaBoost/EasyEnsemble/BalanceCascade classifer
% Input:
%   dataset: n-by-d test set
%   ensemble: AdaBoost/EasyEnsemble/BalanceCascade classifer
%   str_class_eval   : command for classifier evaluation
%               VARIABLE NAME:
%                                model->classifer/model used for evaluation
%                                evalset-> data for evaluation phase
%                                predict_target->stores predicted
%                                                target/label obtained during validation
%                                                phase
%               example: [predict_target]=svmpredict(traintarget, trainset,
%               model);%
% Output:
%   result: predicted labels of test examples

% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Contact: Xu-Ying Liu (liuxy@lamda.nju.edu.cn)
% Modified by: Nicola Lazzarini, 2012
% Modified by: Sharad Kumar Gupta, 2020
% Contact: Sharad Kumar Gupta (sharadgupta27@gmail.com)

result = EvaluateValue(dataset,target,ensemble,str_class_eval,col,modelName); 
result = (result >= ensemble.thresh);

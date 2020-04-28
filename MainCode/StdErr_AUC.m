function [SE, CILower, CIUpper] = StdErr_AUC(auc, confusionMatrix)
%-------------------------------------------------------
% Code for Standard Error of AUC Computation
% SE = Standard Error
% CILower = Confidence Interval Lower Limit (95% CI)
% CIUpper = Confidence Interval Upper Limit (95% CI)
% Code is based on: Bradley AP (1997), Pattern Recognition 30:1145–1159. https://doi.org/10.1016/S0031-3203(96)00142-2
% Copyright: Sharad Kumar Gupta, 2020
% Contact: Sharad Kumar Gupta (sharadgupta27@gmail.com)
%-------------------------------------------------------
TN = confusionMatrix(1,1);
FP = confusionMatrix(1,2);
FN = confusionMatrix(2,1);
TP = confusionMatrix(2,2);

Np = TP + FN;
Nn = TN + FP;

Q1 = auc/(2 - auc);
Q2 = 2*(auc^2)/(1 + auc);

SqSE = (auc*(1 - auc) + (Np - 1)*(Q1 - auc^2) + (Nn - 1)*(Q2 - auc^2))/(Np*Nn);
SE = sqrt(SqSE);

CILower = auc - (1.96*SE);
CIUpper = auc + (1.96*SE);
end

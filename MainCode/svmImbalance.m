% SVM without data imbalance

%-------------------------------------------------------
% Code for preparing LSI using SVM over imbalanced data - IBSVM
% Copyright: Sharad Kumar Gupta, 2020
% Contact: Sharad Kumar Gupta (sharadgupta27@gmail.com)
%-------------------------------------------------------

load Data_Xy_2.mat
rng(1);
fprintf('The Code Started\n');
model = eval('svmtrain(traintarget,trainX,''-s 0 -t 2 -c 1 -g 1'')');
eval('[predict_target, accuracy, scores] = svmpredict(testtarget, testX, model);');
pt = predict_target;
confusionMatrix = confusionmat(testtarget,predict_target)
TN = confusionMatrix(1,1);
FP = confusionMatrix(1,2);
FN = confusionMatrix(2,1);
TP = confusionMatrix(2,2);
recall = TP/(TP+FN); % or sensitivity = TP/(TP+FN);
precision = TP/(TP+FP);
Accuracy = (TP+TN)/(TP+FP+TN+FN);
f_score = 2*(precision*recall)/(precision+recall);
pacc = recall;
nacc = TN/(TN+FP);
g_mean = sqrt(pacc*nacc);
[xplot,yplot,Threshold,auc] = perfcurve(testtarget,predict_target,1,'XCrit','fpr','YCrit','tpr');
% figure, plot(xplot,yplot)
% xlabel('False Positive Rate');
% ylabel('True Positive Rate');
predict_target = double(logical(predict_target));
confusionchart(testtarget,predict_target)
P = length(predict_target(predict_target==1));
N = length(predict_target(predict_target==0));
BalancedAccuracy = ((TP/P)+(TN/N))/2;
w = (model.sv_coef' * full(model.SVs));
bias = -model.rho;
LSI_temp = d*w' + bias;
mx = max(LSI_temp);
mn = min(LSI_temp);
LSI_norm = (LSI_temp - mn)/(mx - mn);
LSI = reshape(LSI_norm,[rows cols]);
geotiffwrite('.\Results\LSI_DataImbalance_SVM.tif',LSI,Rs,'CoordRefSysCode',32644); % Change the EPSG code for the study area
save('.\Results\DataImbalance_SVM.mat','Threshold','xplot','yplot','LSI','w','bias','auc','confusionMatrix','accuracy','model','f_score','g_mean','precision','recall','testtarget','predict_target');
%clearvars
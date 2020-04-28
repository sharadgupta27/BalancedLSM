% Code for demo of EasyEnsemble
%--------------------------------------
% Dataset format instruction:
%   In sample data file "Data_Xy_2.mat", training 'train' and test set 'test' are n-by-d matrixes,
%   where, d is the number of dimensions. And their labels
%   'traintarget' and 'testtarget' are n-by-1 vectors, with values of 0 (negative class) and
%   1 (positive class).
% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Modified by: Sharad Kumar Gupta, 2020 
%	  Contact: sharadgupta27@gmail.com
%--------------------------------------
currDir = pwd();
fprintf('Current working directory - %s\n',currDir);
load Data_Xy_2.mat
% all_layers = [aspect dem geology ndvi plancurv profilecurv relief slope soil spi stream tangcurv thrust twi];
rng(1)
if ~exist('Results','dir')
    mkdir('Results');
end
inputName = {'Enter Model (ANNCode/SVMCode):'};
dlgtitle = 'Input Model Name';
dims = [1 45];
definput = {'ANNCode'}; % Default Value. Input new values when popup appears.
modelName = inputdlg(inputName,dlgtitle,dims,definput);

%columnNo = [1,2,3,8,9,11,13];
% trainX = trainX(:,columnNo);
% testX = testX(:,columnNo);
[row,col] = size(trainX);
T = 30;% set parameter T (sample T subsets of negtive examples)
si = 30; % set parameter si (use si iterations to train each AdaBoost classifier)
if modelName == "SVMCode"
    str_class='svmtrain(boosttarget,boostset,''-s 0 -t 2 -c 3 -g 2.5 -w1 0.75 -w2 0.25'')';
    str_class_eval='[predict_target, accuracy, scores] = svmpredict(evaltarget, evalset, model);';
elseif modelName == "ANNCode"
    str_class='annCustomTrain(boostset,boosttarget)';
    str_class_eval='[predict_target, accuracy] = annCustomPredict(model,evalset,evaltarget);';
end

ensemble = EasyEnsemble(trainX,traintarget,catidx,T,si,str_class,str_class_eval,modelName);

fprintf('\nTest Cases Executing Now\n');
fprintf('=============================\n');
[f,w,bias] = EvaluateValue(testX,testtarget,ensemble,str_class_eval,col,modelName); % get real valued output
rates = CalculatePositives(ensemble,testX,testtarget,str_class_eval,col,modelName);
plot(rates(:,1),rates(:,2));
auc = CalculateAUC(rates)

predicted = f>=ensemble.thresh;
[fval,gmean,confusionMatrix,accuracy,precision,recall] = ImbalanceEvaluate(testtarget,predicted,1,0)
%data = d(:,columnNo);
data = d;
predicted = double(logical(predicted));
LSI = LSICompute(w,bias,data,rows,cols);

fileLSI = strcat('.\Results\LSI_EasyEnsemble_',string(modelName),'.tif');
fileMAT = strcat('.\Results\LSI_EasyEnsemble_',string(modelName),'.mat');
geotiffwrite(fileLSI,LSI,Rs,'CoordRefSysCode',32644); % Change the EPSG code for the study area
save(fileMAT,'w','bias','auc','fval','gmean','confusionMatrix','accuracy','ensemble','predicted','rates','precision','recall','LSI');
%clearvars
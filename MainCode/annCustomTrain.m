function [net] = annCustomTrain(traindata,trainlabel)
%-------------------------------------------------------
% Custom Code for training of ANN
% Copyright: Sharad Kumar Gupta, 2020
% Contact: Sharad Kumar Gupta (sharadgupta27@gmail.com)
%-------------------------------------------------------
x = traindata';
y = trainlabel';

%% Applying ANN
net = cascadeforwardnet([7 7],'trainbfg');
% training functions:'trainbfg','traincgf','trainbr','trainrp','trainscg','traincgp','traincgb','traingda','traingdx','traingdm'
% 'trainbfg' and 'trainscg' both are good functions
net = configure(net, x, y);
net.divideFcn = '';
%     view(net);
%net.trainFcn = 'traingd';

% net.inputWeights{1,1}.learnFcn = 'learngdm';
% net.layerWeights{2,1}.learnFcn = 'learngdm';
% net.biases.learnFcn{1} = 'learngdm';

% net.performFcn = 'crossentropy';
net.performFcn = 'mse';
% net.performFcn = 'mae';
%net.performParam.regularization = 0.01;

% Setting initial weights
% in_wei = net.IW{1,1};
%net.inputConnect = [1; 0; 0; 0; 0; 0; 0; 0];

% Divide data
net.trainParam.epochs = 300;
net.trainParam.goal = 0.00001;
% net.divideParam.trainRatio = 100/100;
% net.divideParam.valRatio = 0/100;
% net.divideParam.testRatio = 0/100;

% Providing the training
net.trainParam.showWindow = false;
[net,~] = train(net,x,y);
end
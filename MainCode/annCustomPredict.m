function [predict_target, accuracy] = annCustomPredict(net, testdata,testlabel)
%-------------------------------------------------------
% Custom Code for prediction using ANN
% Copyright: Sharad Kumar Gupta, 2020
% Contact: Sharad Kumar Gupta (sharadgupta27@gmail.com)
%-------------------------------------------------------
testX = testdata;
testY = testlabel;

% This is for testing our data
predict_target = net(testX');
predict_target = predict_target';
% Calculating the accuracy of classification
predict_target(predict_target>0.5) = 1;
predict_target(predict_target<=0.5) = 0;
pclass = 1;
nclass = 0;
tp = sum(testY==pclass & predict_target==pclass);
fn = sum(testY==pclass & predict_target==nclass);
tn = sum(testY==nclass & predict_target==nclass);
fp = sum(testY==nclass & predict_target==pclass);


accuracy = 100*(tp+tn)/(tp+fp+tn+fn);
%accuracy = (sum(predict_target == testY)/numel(testY))*100;
lenT = length(testY);
% lenP = length(find(predict_target==1));
lenP = tp+tn;
fprintf('Accuracy on (%d/%d) = %.2f\n',lenP,lenT,accuracy);
%scores = perform(net,testY,predict_target);
end
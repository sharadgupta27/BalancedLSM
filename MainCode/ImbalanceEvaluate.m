function [f,g,confmat,acc,precision,recall] = ImbalanceEvaluate(target,result,pclass,nclass)
%-------------------------------------------------------
% Code for computing F-measure and G-mean
% Modified by: Sharad Kumar Gupta, 2020 (sharadgupta27@gmail.com)
% Contact: Sharad Kumar Gupta (sharadgupta27@gmail.com)
%-------------------------------------------------------
tp = sum(target==pclass & result==pclass);
fn = sum(target==pclass & result==nclass);
tn = sum(target==nclass & result==nclass);
fp = sum(target==nclass & result==pclass);

acc = (tp+tn)/(tp+fp+tn+fn);
confmat = [tn fp;fn tp];

if(tp == 0)
    f = 0;
    g = 0;
else
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
    f = 2*(precision*recall)/(precision+recall);
    pacc = recall;
    nacc = tn/(tn+fp);
    g = sqrt(pacc*nacc);
end

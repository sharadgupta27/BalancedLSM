function [ensemble,training_array,label_array] = EasyEnsemble(trainset, traintarget, catidx, T, rounds, str_class, str_class_eval, modelName)

% Input:
%   trainset: n-by-d training set
%   traintarget: n-by-1 training target {0,1}
%   catidx: indicates which attributes are discrete ones (Note: start from 1)
%   T: sample $T$ subsets of negtive examples
%   rounds: use $rounds$ iterations to train each AdaBoost classifier
%   str_class   : command for classifier
%               VARIABLE NAME:
%                                boostarget-> labels for training data
%                                boostset-> data for training phase
%               example: svmtrain(boosttarget,boostset,parameters);
%
%   str_class_eval   : command for classifier evaluation
%               VARIABLE NAME:
%                                evaltarget-> labels for evaluation data
%                                evalset-> data for evaluation phase
%                                predict_target->stores predicted
%                                                target/label obtained during validation
%                                                phase
%               example: [predict_target]=svmpredict(traintarget, trainset, model);%
% Output:
%   ensemble: EasyEnsemble classifier, a structure variable
%   training_arry: array containing all used training set 
%   label_arry: array containing all label for used training set 

% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Contact: Xu-Ying Liu (liuxy@lamda.nju.edu.cn)
% Modified by: Nicola Lazzarini, 2012
% Modified by: Sharad Kumar Gupta, 2020
%	  Contact: sharadgupta27@gmail.com
%--------------------------------------

training_array={};
label_array={};

poscount = sum(traintarget==1);
negcount = length(traintarget)-poscount;
posset = trainset(traintarget==1,:);
negset = trainset(traintarget==0,:);
negset = negset(randperm(negcount),:);

ensemble = struct('classifier',{},'alpha',{},'thresh',{}); 

for node=1:T 
    fprintf("\nTraining Results for Subset - %d\n",node);
    nset = negset(1:poscount,:); % a random subset of negative examples
    %build current training set
    curtrainset = [posset;nset];
    curtarget = zeros(size(curtrainset,1),1);
    curtarget(1:poscount)=1;
    [ens,train_array,lab_array] = AdaBoost(curtrainset,curtarget,catidx,rounds,str_class,str_class_eval,modelName);% node classifier (ADABOOST)   
    training_array{node}={train_array};
    label_array{node}={lab_array};
    ensemble(node) = ens;
    intermediateFile = strcat('.\Results\IntermediateData_subset-',num2str(node),'.mat');
    save(intermediateFile,'curtrainset','curtarget');
    negset = negset(randperm(negcount),:);  %select a new random subset of negative examples  
end

%combine all weak learners to form the final ensemble
depth = length(ensemble); 
ens= struct('classifier',{},'alpha',{},'thresh',{});
for i=1:depth
   ens(1).classifier = [ens.classifier; ensemble(i).classifier]; 
   ens(1).alpha = [ens.alpha; ensemble(i).alpha];
end
ens.thresh = sum(ens.alpha)/2;
ensemble = ens;
function [ensemble,training_array,label_array] = BalanceCascade(trainset, traintarget, catidx, T, rounds, str_class, str_class_eval, col, modelName)
% -------------------------------------------
% Implements Balance Cascade method
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
%               example: [predict_target]=svmpredict(traintarget,trainset,model);
% Output:
%   ensemble: BalanceCascade classifier, a structure variable
%   training_arry: array containing all used training set 
%   label_arry: array containing all label for used training set 

% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Contact: Xu-Ying Liu (liuxy@lamda.nju.edu.cn)
% Modified by: Nicola Lazzarini, 2012
% Modified by: Sharad Kumar Gupta, 2020
% 	Contact  : sharadgupta27@gmail.com
% -------------------------------------------

training_array={};
label_array={};

poscount = sum(traintarget==1);
negcount = length(traintarget)-poscount;

posset = trainset(traintarget==1,:);
negset = trainset(traintarget==0,:);
negset = negset(randperm(negcount),:);
FP = (poscount/negcount)^(1/(T-1));

%select classifier as learning method
ensemble = struct('classifier',{},'alpha',{},'thresh',{});


for node = 1:T
    fprintf("\nTraining Results for Subset - %d\n",node);
    if isempty(negset)
        training_array{node}={train_array};
        label_array{node}={lab_array};
        ensemble(node) = ens;
    else
        if (size(negset,1)<poscount)
            nset=negset;
        else
            nset = negset(1:poscount,:);
        end
        %build current training set
        curtrainset = [posset;nset];
        curtarget = zeros(size(curtrainset,1),1);
        curtarget(1:poscount) = 1;
        [ens,train_array,lab_array] = AdaBoost(curtrainset,curtarget,catidx,rounds,str_class,str_class_eval,modelName);% node classifier (ADABOOST)
        training_array{node}={train_array};
        label_array{node}={lab_array};
        negtarget = zeros(size(negset,1),1);
        ens.thresh = AdjustFPRate(negset,negtarget,ens,FP,str_class_eval,col,modelName);% base on negset
        ensemble(node) = ens;

        result = Predict(negset,negtarget,ens,str_class_eval,col,modelName); %evaluate negset to select correctly negative examples
        negset = negset(result==1,:); % remove correctly classified negative examples
        negcount = size(negset,1);
        negset = negset(randperm(negcount),:);
    end
end

%combine all weak learners to form the final ensemble
depth = length(ensemble);
%select classifier as learning method
ens= struct('classifier',{},'alpha',{},'thresh',{});
for i=1:depth
   ens(1).classifier = [ens.classifier; ensemble(i).classifier]; 
   ens(1).alpha = [ens.alpha; ensemble(i).alpha];
end
ens.thresh = sum(ens.alpha)/2;
ensemble = ens;
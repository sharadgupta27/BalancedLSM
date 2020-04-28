function [ensemble,training_array,label_array] = AdaBoost(trainset, traintarget, catidx, rounds,str_class,str_class_eval,modelName)

% Input:
%   trainset: n-by-d training set
%   traintarget: n-by-1 training target {0,1}
%   catidx: indicates which attributes are discrete ones (Note: start from 1, used only for certain classifiers)
%   rounds: use $rounds$ iterations to train AdaBoost classifier
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
%               example: [predict_target]=svmpredict(evaltarget, evalset,
%               model);%
% Output:
%   ensemble: AdaBoost classifier, a structure variable
%   training_arry: array containing all used training set
%   label_arry: array containing all label for used training set

% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Modified by: Nicola Lazzarini, 2012
% Modified by: Sharad Kumar Gupta, 2020
% Contact: Sharad Kumar Gupta (sharadgupta27@gmail.com)

fprintf('Total no. of points in balanced training data = %d\n', size(trainset,1));
training_array={};
label_array={};

ensemble.classifier = cell(rounds,1); %(modified for SVM)
ensemble.alpha = zeros(rounds,1);
ensemble.thresh = 0;

conta=0;
NO=0;

while conta==0
    
    weight = zeros(size(traintarget));
    weight(traintarget==1) = 1/sum(traintarget==1);
    weight(traintarget==0) = 1/sum(traintarget==0);
    weight = weight / sum(weight);
    
    result = zeros(size(traintarget));
    
    for i=1:rounds
        if sum(isnan(weight))
            %keyboard
            NO=1;
            break
        end
        [boostset, boosttarget] = boost_data(trainset,traintarget,weight);
        training_array{i}=boostset;
        label_array{i}=boosttarget;
        %train a model using traintarget
        model=eval(str_class);%this command need to use boostset and boostarget
        ensemble.classifier{i} = model;
        evalset=trainset;
        evaltarget=traintarget;
        eval(str_class_eval);%this command need to use evalset and evaltarget; predicted labels MUST TO BE STORED in: predict_target
        %evaluate trainset on the model created to calculate errors and weights
        trainresult=predict_target;
        trainerror = sum(weight.*(trainresult~=traintarget));
        if trainerror==0
            for other=i+1:rounds
                training_array{other}=boostset;
                label_array{other}=boosttarget;
                ensemble.classifier{other} = model;
            end
            break
        end
        beta = (1-trainerror)/trainerror;
        ensemble.alpha(i) = 0.5*log(beta);
        result = result + ensemble.alpha(i) * trainresult;
        weight = weight .* exp(-ensemble.alpha(i)*(trainresult-0.5).*(traintarget-0.5)*4);
        weight = weight / sum(weight);
    end
    
    if NO==0
        conta=1;
        ensemble.thresh = sum(ensemble.alpha)/2;
    else
        NO=0;
    end
end
function [LSI] = LSICompute(w,b,data,rows,cols)
%-------------------------------------------------------
% Code for LSI computation from weight and bias obtained
% from ANN and SVM model.
% Copyright: Sharad Kumar Gupta, 2020
% Contact: Sharad Kumar Gupta (sharadgupta27@gmail.com)
%-------------------------------------------------------
bias = mean(b);
w1 = mean(w);
LSI_temp = data*w1' + bias;
mx = max(LSI_temp);
mn = min(LSI_temp);
LSI_norm = (LSI_temp - mn)/(mx - mn);
LSI = reshape(LSI_norm,[rows cols]);
end
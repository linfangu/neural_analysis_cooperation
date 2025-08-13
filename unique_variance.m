function [dvar,varexp] = unique_variance(x,target_idx,ca,nshift)
% The unique variance in the neural space that can be explained by specific behavioral variables 
% First calulate the total variance that can be explained by all behaviors.
% Then shuffle the target variable, leaving other behaviors intact, and
% calculate variance explained by this new matrix. Unique variance is defined
% as the reduction of variance explained after shuffling the target
% variable.

% Inputs:
%   x: behavioral traces (time x behavior)
%   target_idx: the behavioral column of interest (to calculate unique var)
%   ca: neural traces (time x cell)
%   nshift: number of shuffle 

% Example usage:
%   unique_variance(x,[1,2],ca,100) % if target variable contains two
%   columns (e.g. the x and y coordinate of one animal)
%   unique_variance(x,3,ca,100) %  if target variable is one column

% Outputs:
%  dvar: unique variance explained by the targer variable (percentage in
%  total neural space)
%   varexp: total percentage variance in the neural space that can be captured by the
%   whole behavior space 

y = zscore(ca);
% run plsr y
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(x,y);
varexp = sum(PCTVAR(2,:));
shift = randi([15,length(y)-15],1,nshift);
varshuf = nan(1,nshift);
parfor sh = 1:nshift % shuffle behavior
    shuffled = circshift(x(:,target_idx),shift(sh),1);
    newX = [x(:,setdiff(1:size(x,2),target_idx)),shuffled];
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(newX,y);
    varshuf(sh) = sum(PCTVAR(2,:));
end
dvar = varexp-mean(varshuf);
end
